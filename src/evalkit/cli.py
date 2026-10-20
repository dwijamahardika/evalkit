"""Command-line interface.

Exit codes (stable, for CI):
  0  success / no significant regression
  1  gate failed (pass rate below threshold, significant regression, kappa too low)
  2  invalid input (bad arguments, invalid suite, unreadable run file)
  3  infrastructure errors (provider/judge calls failed after retries)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from evalkit import __version__
from evalkit.cache import CacheMode, ResponseCache
from evalkit.calibrate import (
    CalibrationError,
    calibrate,
    read_labels,
    render_calibration_text,
    write_calibration,
)
from evalkit.config import ConfigError, LoadedSuite, load_suite, pricing_issues
from evalkit.diff import FailOn, diff_runs, render_diff_markdown, render_diff_text
from evalkit.providers.base import Provider
from evalkit.providers.factory import build_provider
from evalkit.providers.openai_compat import MissingCredentialError
from evalkit.report import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    render_junit,
    render_markdown,
    summarize,
)
from evalkit.report import render_text as render_run_text
from evalkit.runner import Runner, RunOptions, SelectionError
from evalkit.runstore import Run, RunFormatError, TrialRecord, read_run, write_run

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="LLM regression testing that runs in CI like unit tests.",
)


def _fail(message: str, code: int = EXIT_USAGE) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code)


def _write(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {path}", err=True)


def _load(path: Path) -> LoadedSuite:
    try:
        return load_suite(path)
    except ConfigError as exc:
        _fail(exc.render())


def _read(path: Path) -> Run:
    try:
        return read_run(path)
    except RunFormatError as exc:
        _fail(str(exc))


def _version(value: bool) -> None:
    if value:
        typer.echo(f"evalkit {__version__}")
        raise typer.Exit(EXIT_OK)


@app.callback()
def _root(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """LLM regression testing that runs in CI like unit tests."""


@app.command()
def validate(
    suites: Annotated[list[Path], typer.Argument(help="Suite YAML file(s).", show_default=False)],
) -> None:
    """Check suite files without calling any model. Reports every error at once."""
    ok = True
    for path in suites:
        try:
            loaded = load_suite(path)
        except ConfigError as exc:
            typer.echo(exc.render(), err=True)
            ok = False
            continue
        s = loaded.suite
        n_asserts = sum(len(s.assertions_for(c)) for c in s.cases)
        typer.echo(
            f"{path}: ok ({len(s.cases)} cases, {n_asserts} assertions, "
            f"{s.provider.type}/{s.provider.model}, {s.run.trials} trials/case)"
        )
    raise typer.Exit(EXIT_OK if ok else EXIT_USAGE)


def _parse_labels(items: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            _fail(f"--label expects key=value, got {item!r}")
        labels[key] = value
    return labels


async def _execute(runner: Runner, providers: list[Provider]) -> Run:
    try:
        return await runner.run()
    finally:
        for provider in providers:
            await provider.aclose()


@app.command()
def run(
    suite: Annotated[Path, typer.Argument(help="Suite YAML file.", show_default=False)],
    out: Annotated[Path, typer.Option("--out", "-o", help="Run file to write (JSONL).")],
    trials: Annotated[int | None, typer.Option(min=1, help="Override trials per case.")] = None,
    model: Annotated[str | None, typer.Option(help="Override the model under test.")] = None,
    concurrency: Annotated[int | None, typer.Option(min=1, help="Max in-flight calls.")] = None,
    case: Annotated[
        list[str] | None, typer.Option("--case", help="Only this case id (repeatable).")
    ] = None,
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Only cases with this tag (repeatable).")
    ] = None,
    cache: Annotated[CacheMode, typer.Option(help="Response cache mode.")] = CacheMode.USE,
    cache_dir: Annotated[
        Path, typer.Option(envvar="EVALKIT_CACHE_DIR", help="Response cache directory.")
    ] = Path(".evalkit-cache"),
    markdown: Annotated[Path | None, typer.Option(help="Also write a Markdown report.")] = None,
    junit: Annotated[Path | None, typer.Option(help="Also write JUnit XML.")] = None,
    label: Annotated[
        list[str] | None, typer.Option("--label", help="key=value stored in the run (repeatable).")
    ] = None,
    allow_errors: Annotated[bool, typer.Option(help="Do not exit 3 on provider errors.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="No per-case progress.")] = False,
) -> None:
    """Run a suite: every case x N trials, assertions, gate. Writes a JSONL run file."""
    loaded = _load(suite)
    if model:
        s = loaded.suite
        s = s.model_copy(update={"provider": s.provider.model_copy(update={"model": model})})
        issues = pricing_issues(s, model)
        if issues:
            _fail("\n".join(str(i) for i in issues))
        loaded = LoadedSuite(suite=s, path=loaded.path, base_dir=loaded.base_dir)
    s = loaded.suite
    try:
        provider = build_provider(s.provider)
        judge_provider = build_provider(s.judge.provider) if s.judge else None
    except MissingCredentialError as exc:
        _fail(str(exc))

    def progress(case_id: str, records: list[TrialRecord]) -> None:
        if not quiet:
            passed = sum(r.passed for r in records)
            errored = sum(r.error is not None for r in records)
            extra = f" ({errored} errored)" if errored else ""
            typer.echo(f"  {case_id}: {passed}/{len(records)}{extra}", err=True)

    options = RunOptions(
        trials=trials,
        concurrency=concurrency,
        case_ids=frozenset(case or []),
        tags=frozenset(tag or []),
        labels=_parse_labels(label or []),
    )
    runner = Runner(
        loaded,
        provider,
        judge_provider,
        ResponseCache(cache_dir, cache),
        options,
        on_case_done=progress,
    )
    providers = [provider] + ([judge_provider] if judge_provider else [])
    try:
        result = asyncio.run(_execute(runner, providers))
    except SelectionError as exc:
        _fail(str(exc))
    write_run(out, result)
    summary = summarize(result)
    typer.echo(render_run_text(summary, allow_errors=allow_errors))
    typer.echo(f"wrote {out}", err=True)
    _write(markdown, render_markdown(summary, allow_errors=allow_errors))
    _write(junit, render_junit(summary))
    raise typer.Exit(summary.exit_code(allow_errors=allow_errors))


@app.command()
def report(
    run_file: Annotated[Path, typer.Argument(help="Run file (JSONL).", show_default=False)],
    markdown: Annotated[Path | None, typer.Option(help="Write a Markdown report.")] = None,
    junit: Annotated[Path | None, typer.Option(help="Write JUnit XML.")] = None,
    allow_errors: Annotated[bool, typer.Option(help="Do not exit 3 on errored trials.")] = False,
) -> None:
    """Summarise an existing run and re-apply its gate."""
    summary = summarize(_read(run_file))
    typer.echo(render_run_text(summary, allow_errors=allow_errors))
    _write(markdown, render_markdown(summary, allow_errors=allow_errors))
    _write(junit, render_junit(summary))
    raise typer.Exit(summary.exit_code(allow_errors=allow_errors))


@app.command()
def diff(
    base: Annotated[Path, typer.Argument(help="Baseline run file.", show_default=False)],
    candidate: Annotated[Path, typer.Argument(help="Candidate run file.", show_default=False)],
    alpha: Annotated[float, typer.Option(min=0.0001, max=0.5, help="Significance level.")] = 0.05,
    fail_on: Annotated[FailOn, typer.Option(help="What counts as a regression.")] = FailOn.ANY,
    markdown: Annotated[Path | None, typer.Option(help="Write a Markdown report.")] = None,
    resamples: Annotated[int, typer.Option(min=100, help="Bootstrap resamples.")] = 10_000,
    seed: Annotated[int, typer.Option(help="Bootstrap seed (results are deterministic).")] = 0,
    allow_errors: Annotated[bool, typer.Option(help="Do not exit 3 on errored trials.")] = False,
) -> None:
    """Per-case regressions/improvements with significance tests, and the suite delta."""
    result = diff_runs(_read(base), _read(candidate), alpha=alpha, resamples=resamples, seed=seed)
    typer.echo(render_diff_text(result, fail_on, allow_errors))
    _write(markdown, render_diff_markdown(result, fail_on, allow_errors))
    raise typer.Exit(result.exit_code(fail_on, allow_errors=allow_errors))


@app.command(name="calibrate")
def calibrate_cmd(
    run_file: Annotated[
        Path, typer.Argument(help="Run with llm_judge results.", show_default=False)
    ],
    labels: Annotated[
        Path, typer.Argument(help="CSV: case_id,trial,label[,assertion].", show_default=False)
    ],
    out: Annotated[Path | None, typer.Option(help="Write the calibration file here.")] = None,
    min_kappa: Annotated[
        float | None, typer.Option(min=-1.0, max=1.0, help="Exit 1 if any judge is below this.")
    ] = None,
) -> None:
    """Compare judge verdicts with human labels: agreement, Cohen's kappa, confusion."""
    run_data = _read(run_file)
    try:
        result = calibrate(run_data, read_labels(labels))
    except CalibrationError as exc:
        _fail(str(exc))
    typer.echo(render_calibration_text(result))
    if out is not None:
        write_calibration(out, result)
        typer.echo(f"wrote {out}", err=True)
    if min_kappa is not None:
        worst = result.worst_kappa()
        if worst is None or worst < min_kappa:
            shown = "undefined" if worst is None else f"{worst:.3f}"
            typer.echo(f"result: FAIL (kappa {shown} < {min_kappa})")
            raise typer.Exit(EXIT_GATE_FAILED)
        typer.echo(f"result: PASS (kappa {worst:.3f} >= {min_kappa})")
    raise typer.Exit(EXIT_OK)


def main() -> None:
    app()
