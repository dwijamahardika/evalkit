"""Execute a suite: cases x trials -> assertion results -> a Run."""

from __future__ import annotations

import asyncio
import datetime as dt
import random
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from evalkit import __version__, judge, template
from evalkit.assertions import AssertionResult, TrialContext, evaluate
from evalkit.cache import CacheMode, ResponseCache, cache_key
from evalkit.config import Case, LlmJudge, LoadedSuite
from evalkit.pricing import cost_usd
from evalkit.providers.base import (
    ChatMessage,
    Completion,
    CompletionRequest,
    Provider,
    ProviderError,
)
from evalkit.retry import RetryPolicy, Sleep, call_with_retries
from evalkit.runstore import CaseMeta, Run, RunMeta, TrialRecord


@dataclass(frozen=True)
class RunOptions:
    trials: int | None = None  # overrides suite and per-case values
    concurrency: int | None = None
    case_ids: frozenset[str] = frozenset()
    tags: frozenset[str] = frozenset()
    labels: Mapping[str, str] = field(default_factory=dict)


class SelectionError(ValueError):
    pass


def render_request(loaded: LoadedSuite, case: Case, sample_index: int) -> CompletionRequest:
    prompt = loaded.suite.prompt
    system = template.render(prompt.system, case.inputs) if prompt.system else None
    messages = tuple(
        ChatMessage(m.role, template.render(m.content, case.inputs)) for m in prompt.messages
    )
    return CompletionRequest(system=system, messages=messages, sample_index=sample_index)


def select_cases(loaded: LoadedSuite, options: RunOptions) -> list[Case]:
    cases = loaded.suite.cases
    known = {c.id for c in cases}
    unknown = sorted(options.case_ids - known)
    if unknown:
        raise SelectionError(f"unknown case id(s): {', '.join(unknown)}")
    if options.case_ids:
        cases = [c for c in cases if c.id in options.case_ids]
    if options.tags:
        cases = [c for c in cases if options.tags & set(c.tags)]
    if not cases:
        raise SelectionError("no cases selected")
    return cases


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


class Runner:
    def __init__(
        self,
        loaded: LoadedSuite,
        provider: Provider,
        judge_provider: Provider | None,
        cache: ResponseCache,
        options: RunOptions | None = None,
        *,
        rng: random.Random | None = None,
        sleep: Sleep = asyncio.sleep,
        on_case_done: Callable[[str, list[TrialRecord]], None] | None = None,
    ) -> None:
        self.loaded = loaded
        self.suite = loaded.suite
        self.provider = provider
        self.judge_provider = judge_provider
        self.cache = cache
        self.options = options or RunOptions()
        self.rng = rng or random.Random()
        self.sleep = sleep
        self.on_case_done = on_case_done
        run_cfg = self.suite.run
        self.policy = RetryPolicy(
            max_attempts=run_cfg.max_attempts,
            base_s=run_cfg.backoff_base_s,
            cap_s=run_cfg.backoff_cap_s,
        )
        self._concurrency = self.options.concurrency or run_cfg.concurrency
        self._sem = asyncio.Semaphore(self._concurrency)
        self._gating: dict[str, judge.GatingDecision] = {}
        self._identities: dict[int, dict[str, Any]] = {}
        self._calibration = (
            judge.load_calibration(self.suite.judge, loaded.base_dir) if self.suite.judge else None
        )

    def trials_for(self, case: Case) -> int:
        return self.options.trials or case.trials or self.suite.run.trials

    # ------------------------------------------------------------------ calls

    async def _complete(
        self, provider: Provider, request: CompletionRequest
    ) -> tuple[Completion, int, bool]:
        """(completion, attempts, cached). The slot is held across backoff sleeps:
        a 429 means the endpoint is saturated, and freeing the slot would just
        let another request walk into the same limit."""
        key = None
        if self.cache.mode is not CacheMode.OFF:
            if id(provider) not in self._identities:
                self._identities[id(provider)] = provider.identity()
            key = cache_key(self._identities[id(provider)], request)
            hit = self.cache.get(key)
            if hit is not None:
                return hit, 0, True
        async with self._sem:
            completion, attempts = await call_with_retries(
                lambda: provider.complete(request), self.policy, rng=self.rng, sleep=self.sleep
            )
        if key is not None:
            self.cache.put(key, completion)
        return completion, attempts, False

    def _decision(self, assertion: LlmJudge) -> judge.GatingDecision:
        assert self.suite.judge is not None
        assert self.judge_provider is not None
        fp = judge.fingerprint(self.judge_provider.identity(), assertion)
        if fp not in self._gating:
            self._gating[fp] = judge.decide_gating(
                self.suite.judge, self._calibration, fp, assertion.display_name()
            )
        return self._gating[fp]

    async def _judge(
        self, assertion: LlmJudge, case: Case, request: CompletionRequest, output: str
    ) -> tuple[AssertionResult, float | None, str | None]:
        """(result, judge cost, infrastructure error)."""
        assert self.judge_provider is not None
        decision = self._decision(assertion)
        base = AssertionResult(
            name=assertion.display_name(),
            type=assertion.type,
            passed=False,
            gating=decision.gating,
            fingerprint=decision.fingerprint,
        )
        jreq = judge.build_request(
            task=request.prompt_text(),
            output=output,
            rubric=template.render(assertion.rubric, case.inputs),
            sample_index=request.sample_index,
        )
        try:
            completion, _, _ = await self._complete(self.judge_provider, jreq)
        except ProviderError as exc:
            msg = f"judge call failed: {exc}"
            return replace(base, detail=msg), None, msg
        cost = cost_usd(completion.usage, self.suite.prices)
        try:
            score, reason = judge.parse_verdict(completion.text)
        except judge.JudgeParseError as exc:
            # A judge that cannot follow its output contract is broken tooling, not
            # a regression in the model under test.
            return replace(base, detail=str(exc)), cost, str(exc)
        passed = score >= assertion.threshold
        detail = f"score {score}/5 (threshold {assertion.threshold}): {reason}"
        return replace(base, passed=passed, detail=detail, score=float(score)), cost, None

    # ----------------------------------------------------------------- trials

    async def _trial(self, case: Case, index: int) -> TrialRecord:
        request = render_request(self.loaded, case, index)
        try:
            completion, attempts, cached = await self._complete(self.provider, request)
        except ProviderError as exc:
            return TrialRecord(
                case_id=case.id,
                trial=index,
                passed=False,
                output=None,
                attempts=exc.attempts,
                error=str(exc),
            )
        cost = cost_usd(completion.usage, self.suite.prices)
        ctx = TrialContext(
            output=completion.text,
            latency_ms=completion.latency_ms,
            cost_usd=cost,
            strip_code_fences=self.suite.output.strip_code_fences,
        )
        results: list[AssertionResult] = []
        judge_cost: float | None = 0.0
        errors: list[str] = []
        for assertion in self.suite.assertions_for(case):
            if isinstance(assertion, LlmJudge):
                result, jcost, err = await self._judge(assertion, case, request, completion.text)
                judge_cost = None if (judge_cost is None or jcost is None) else judge_cost + jcost
                if err:
                    errors.append(err)
                results.append(result)
            else:
                results.append(evaluate(assertion, ctx))
        has_judge = any(isinstance(a, LlmJudge) for a in self.suite.assertions_for(case))
        return TrialRecord(
            case_id=case.id,
            trial=index,
            passed=all(r.passed for r in results if r.gating),
            output=completion.text,
            assertions=results,
            latency_ms=completion.latency_ms,
            usage=completion.total_usage(),
            cost_usd=cost,
            judge_cost_usd=judge_cost if has_judge else None,
            cached=cached,
            attempts=attempts,
            served_model=completion.served_model,
            stop_reason=completion.stop_reason,
            error="; ".join(errors) or None,
        )

    async def _case(self, case: Case) -> list[TrialRecord]:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self._trial(case, i)) for i in range(self.trials_for(case))]
        records = [t.result() for t in tasks]
        if self.on_case_done:
            self.on_case_done(case.id, records)
        return records

    async def run(self) -> Run:
        cases = select_cases(self.loaded, self.options)
        started = _now()
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self._case(c)) for c in cases]
        trials = [record for task in tasks for record in task.result()]
        suite = self.suite
        meta = RunMeta(
            run_id=uuid.uuid4().hex[:12],
            suite=suite.name,
            suite_path=self.loaded.path.as_posix(),
            suite_fingerprint=self.loaded.fingerprint,
            evalkit_version=__version__,
            started_at=started,
            finished_at=_now(),
            provider={"type": self.provider.name, "model": self.provider.model},
            judge=(
                {"type": self.judge_provider.name, "model": self.judge_provider.model}
                if self.judge_provider
                else None
            ),
            min_suite_pass_rate=suite.gate.min_suite_pass_rate,
            cases=[
                CaseMeta(
                    id=c.id,
                    trials=self.trials_for(c),
                    min_pass_rate=suite.min_pass_rate_for(c),
                    tags=list(c.tags),
                )
                for c in cases
            ],
            judges=[d.to_dict() for d in self._gating.values()],
            labels=dict(self.options.labels),
            cache={"hits": self.cache.hits, "misses": self.cache.misses},
        )
        return Run(meta=meta, trials=trials)
