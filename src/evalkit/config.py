"""Suite configuration: a strict schema, validated in one pass.

Two kinds of mistakes are collected together and reported at once:

* structural errors (unknown keys, wrong types, bad regexes, unreadable schema or
  script files) come from pydantic, which already aggregates them;
* cross-reference errors (a prompt placeholder a case gives no input for,
  duplicate case ids, an `llm_judge` without a `judge:` section) are checked
  against the raw document, so they are found even when the structure is broken.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from evalkit import template
from evalkit.providers.fake import FakeScript

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


# --------------------------------------------------------------------------- errors


@dataclass(frozen=True)
class ConfigIssue:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}" if self.path else self.message


class ConfigError(Exception):
    def __init__(self, source: str, issues: Sequence[ConfigIssue]) -> None:
        self.source = source
        self.issues = list(issues)
        super().__init__(self.render())

    def render(self) -> str:
        noun = "error" if len(self.issues) == 1 else "errors"
        lines = [f"{self.source}: {len(self.issues)} {noun}"]
        lines += [f"  - {issue}" for issue in self.issues]
        return "\n".join(lines)


# ---------------------------------------------------------------- file resolution


def _base_dir(info: ValidationInfo) -> Path:
    ctx = info.context or {}
    base = ctx.get("base_dir")
    return base if isinstance(base, Path) else Path.cwd()


def _read_relative(value: str, info: ValidationInfo) -> str:
    path = _base_dir(info) / value
    if not path.is_file():
        raise ValueError(f"file not found: {value!r} (paths are relative to the suite file)")
    return path.read_text(encoding="utf-8")


def _load_structured(value: str, info: ValidationInfo) -> Any:
    text = _read_relative(value, info)
    try:
        if value.endswith(".json"):
            return json.loads(text)
        return yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{value}: cannot parse: {exc}") from exc


def fmt_num(x: float) -> str:
    """45500.0 -> '45500' (no exponent for whole amounts), 0.1 -> '0.1'."""
    if x == x and abs(x) < 1e15 and float(x).is_integer():
        return str(int(x))
    return f"{x:.6g}"


def _short(value: Any, limit: int = 32) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------- assertions


class _Assertion(Strict):
    name: str | None = Field(default=None, min_length=1, max_length=80)

    def default_name(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def display_name(self) -> str:
        return self.name or self.default_name()


class Equals(_Assertion):
    type: Literal["equals"]
    value: Any
    path: str | None = None
    ignore_case: bool = False
    strip: bool = True

    def default_name(self) -> str:
        target = f"{self.path}=" if self.path else ""
        return f"equals({target}{_short(self.value)})"


class Contains(_Assertion):
    type: Literal["contains"]
    value: str
    path: str | None = None
    ignore_case: bool = False
    negate: bool = False

    def default_name(self) -> str:
        verb = "not_contains" if self.negate else "contains"
        target = f"{self.path}:" if self.path else ""
        return f"{verb}({target}{_short(self.value)})"


class Regex(_Assertion):
    type: Literal["regex"]
    pattern: str
    path: str | None = None
    ignore_case: bool = False
    fullmatch: bool = False
    negate: bool = False

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return value

    def default_name(self) -> str:
        return f"regex({_short(self.pattern)})"


class JsonSchema(_Assertion):
    type: Literal["json_schema"]
    schema_: dict[str, Any] = Field(alias="schema")

    @field_validator("schema_", mode="before")
    @classmethod
    def _load(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, str):
            value = _load_structured(value, info)
        if isinstance(value, dict):
            try:
                validator_for(value).check_schema(value)
            except SchemaError as exc:
                raise ValueError(f"not a valid JSON Schema: {exc.message}") from exc
        return value

    def default_name(self) -> str:
        return "json_schema"


class NumericClose(_Assertion):
    type: Literal["numeric_close"]
    expected: float
    path: str | None = None
    abs_tol: float = Field(default=0.0, ge=0)
    rel_tol: float = Field(default=0.0, ge=0)

    def default_name(self) -> str:
        target = f"{self.path}~" if self.path else "~"
        return f"numeric_close({target}{fmt_num(self.expected)})"


class MaxLatencyMs(_Assertion):
    type: Literal["max_latency_ms"]
    value: float = Field(gt=0)

    def default_name(self) -> str:
        return f"max_latency_ms({fmt_num(self.value)})"


class MaxCostUsd(_Assertion):
    type: Literal["max_cost_usd"]
    value: float = Field(gt=0)

    def default_name(self) -> str:
        return f"max_cost_usd({fmt_num(self.value)})"


class LlmJudge(_Assertion):
    type: Literal["llm_judge"]
    rubric: str = Field(min_length=1)
    threshold: int = Field(default=4, ge=1, le=5)

    def default_name(self) -> str:
        return "llm_judge"


Assertion = Annotated[
    Equals | Contains | Regex | JsonSchema | NumericClose | MaxLatencyMs | MaxCostUsd | LlmJudge,
    Field(discriminator="type"),
]


# ----------------------------------------------------------------------- providers


class FakeProviderConfig(Strict):
    type: Literal["fake"]
    model: str = Field(min_length=1)
    script: FakeScript

    @field_validator("script", mode="before")
    @classmethod
    def _load(cls, value: Any, info: ValidationInfo) -> Any:
        return _load_structured(value, info) if isinstance(value, str) else value


class AnthropicProviderConfig(Strict):
    """Anthropic Messages API via the official SDK.

    Credentials are resolved by the SDK itself (ANTHROPIC_API_KEY or a configured
    profile); evalkit never reads them. Sampling parameters (temperature/top_p)
    are not offered: current Claude models reject them; `effort` is the knob.
    """

    type: Literal["anthropic"]
    model: str = Field(default=DEFAULT_ANTHROPIC_MODEL, min_length=1)
    max_tokens: int = Field(default=16_000, ge=1, le=128_000)
    effort: Effort | None = None
    # Server-side refusal fallback. None = on for models known to support it.
    fallbacks: Literal["default", "none"] | None = None
    timeout_s: float = Field(default=120.0, gt=0)


class OpenAIProviderConfig(Strict):
    """Any server that speaks the OpenAI chat-completions wire format."""

    type: Literal["openai"]
    model: str = Field(min_length=1)
    base_url: str
    api_key_env: str | None = "OPENAI_API_KEY"
    max_tokens: int = Field(default=1024, ge=1)
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = None
    structured_output: bool = True
    timeout_s: float = Field(default=120.0, gt=0)

    @field_validator("base_url")
    @classmethod
    def _http(cls, value: str) -> str:
        if not re.match(r"^https?://", value):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")


ProviderConfig = Annotated[
    FakeProviderConfig | AnthropicProviderConfig | OpenAIProviderConfig,
    Field(discriminator="type"),
]


# ----------------------------------------------------------------- judge, pricing


class JudgeCalibration(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    name: str
    n: int
    agreement: float
    kappa: float | None


class CalibrationFile(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    schema_version: Literal[1]
    judges: dict[str, JudgeCalibration]


class JudgeConfig(Strict):
    provider: ProviderConfig
    # Path (relative to the suite) written by `evalkit calibrate`. May not exist yet.
    calibration: str | None = None
    # When set, llm_judge assertions only gate CI if the calibrated Cohen's kappa
    # for that exact judge (model + rubric + threshold) is at least this value.
    min_kappa: float | None = Field(default=None, ge=-1, le=1)

    @field_validator("calibration")
    @classmethod
    def _well_formed(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is not None and (_base_dir(info) / value).is_file():
            try:
                CalibrationFile.model_validate(_load_structured(value, info))
            except ValidationError as exc:
                first = exc.errors(include_url=False)[0]
                where = ".".join(str(p) for p in first["loc"]) or "(root)"
                raise ValueError(
                    f"{value} is not a calibration file ({where}: {first['msg']})"
                ) from exc
        return value


class Price(Strict):
    """USD per million tokens. Example values only: verify against your provider."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    cache_read_per_mtok: float | None = Field(default=None, ge=0)
    cache_write_per_mtok: float | None = Field(default=None, ge=0)


# --------------------------------------------------------------------------- suite


class MessageTemplate(Strict):
    role: Literal["user", "assistant"]
    content: str


class PromptConfig(Strict):
    system: str | None = None
    messages: list[MessageTemplate] = Field(min_length=1)

    @model_validator(mode="after")
    def _user_turns(self) -> PromptConfig:
        if self.messages[0].role != "user" or self.messages[-1].role != "user":
            raise ValueError("messages must start and end with a `user` turn")
        return self

    def templates(self) -> list[str]:
        return ([self.system] if self.system else []) + [m.content for m in self.messages]


class RunConfig(Strict):
    trials: int = Field(default=5, ge=1, le=1000)
    concurrency: int = Field(default=4, ge=1, le=256)
    max_attempts: int = Field(default=5, ge=1, le=20)
    backoff_base_s: float = Field(default=0.5, ge=0)
    backoff_cap_s: float = Field(default=30.0, ge=0)


class GateConfig(Strict):
    # A case passes the gate when its observed pass rate is at least this.
    min_pass_rate: float = Field(default=1.0, ge=0, le=1)
    min_suite_pass_rate: float | None = Field(default=None, ge=0, le=1)


class OutputConfig(Strict):
    # Unwrap a single ```json fenced block before JSON-based assertions.
    strip_code_fences: bool = False


class Defaults(Strict):
    assert_: list[Assertion] = Field(default_factory=list, alias="assert")


class Case(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$", max_length=128)
    description: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    assert_: list[Assertion] = Field(default_factory=list, alias="assert")
    trials: int | None = Field(default=None, ge=1, le=1000)
    min_pass_rate: float | None = Field(default=None, ge=0, le=1)
    tags: list[str] = Field(default_factory=list)


class Suite(Strict):
    version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str | None = None
    provider: ProviderConfig
    judge: JudgeConfig | None = None
    run: RunConfig = Field(default_factory=RunConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    prices: dict[str, Price] = Field(default_factory=dict)
    prompt: PromptConfig
    defaults: Defaults = Field(default_factory=Defaults)
    cases: list[Case] = Field(min_length=1)

    @field_validator("prices", mode="before")
    @classmethod
    def _load_prices(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, str):
            data = _load_structured(value, info)
            if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
                raise ValueError(f"{value}: expected a top-level `models:` mapping")
            return data["models"]
        return value

    def assertions_for(self, case: Case) -> list[Assertion]:
        return [*self.defaults.assert_, *case.assert_]

    def min_pass_rate_for(self, case: Case) -> float:
        return case.min_pass_rate if case.min_pass_rate is not None else self.gate.min_pass_rate

    def fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(mode="json", by_alias=True), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


def pricing_issues(suite: Suite, model: str) -> list[ConfigIssue]:
    """`max_cost_usd` is meaningless without a price for the model under test."""
    uses_cost = any(
        isinstance(a, MaxCostUsd) for case in suite.cases for a in suite.assertions_for(case)
    )
    if uses_cost and model not in suite.prices:
        return [
            ConfigIssue(
                "prices",
                f"max_cost_usd is used but there is no price for model {model!r}",
            )
        ]
    return []


# ------------------------------------------------------------------------- loading


@dataclass(frozen=True)
class LoadedSuite:
    suite: Suite
    path: Path
    base_dir: Path

    @property
    def fingerprint(self) -> str:
        return self.suite.fingerprint()


def _format_loc(loc: Iterable[int | str], raw: Any) -> str:
    """('cases', 2, 'assert', 0, 'regex', 'pattern') -> cases[2](my-case).assert[0].pattern"""
    out = ""
    node = raw
    parent_key: int | str | None = None
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
            node = node[part] if isinstance(node, list) and 0 <= part < len(node) else None
            if parent_key == "cases" and isinstance(node, dict) and isinstance(node.get("id"), str):
                out += f"({node['id']})"
        else:
            # Discriminated unions insert the tag into the location; it is not a key.
            if isinstance(node, dict) and part not in node and node.get("type") == part:
                continue
            out += f".{part}" if out else part
            node = node.get(part) if isinstance(node, dict) else None
        parent_key = part
    return out


def _cross_reference_issues(raw: dict[str, Any]) -> list[ConfigIssue]:
    issues: list[ConfigIssue] = []

    def strings(*values: Any) -> list[str]:
        return [v for v in values if isinstance(v, str)]

    def as_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    def as_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    prompt = as_dict(raw.get("prompt"))
    prompt_vars: set[str] = set()
    for text in strings(
        prompt.get("system"), *(as_dict(m).get("content") for m in as_list(prompt.get("messages")))
    ):
        prompt_vars |= template.variables(text)

    defaults = [as_dict(a) for a in as_list(as_dict(raw.get("defaults")).get("assert"))]
    has_judge = raw.get("judge") is not None

    def judge_assertions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [a for a in items if a.get("type") == "llm_judge"]

    if judge_assertions(defaults) and not has_judge:
        issues.append(
            ConfigIssue("defaults.assert", "llm_judge needs a top-level `judge:` section")
        )

    cases = as_list(raw.get("cases"))
    ids = [c.get("id") for c in cases if isinstance(c, dict) and isinstance(c.get("id"), str)]
    for dup, count in Counter(ids).items():
        if count > 1:
            issues.append(ConfigIssue("cases", f"duplicate case id {dup!r} ({count}x)"))

    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        where = f"cases[{i}]" + (f"({case['id']})" if isinstance(case.get("id"), str) else "")
        own = [as_dict(a) for a in as_list(case.get("assert"))]
        if judge_assertions(own) and not has_judge:
            issues.append(
                ConfigIssue(f"{where}.assert", "llm_judge needs a top-level `judge:` section")
            )
        inputs = case.get("inputs", {})
        if not isinstance(inputs, dict):
            continue  # reported structurally
        needed = set(prompt_vars)
        for a in judge_assertions(defaults + own):
            for text in strings(a.get("rubric")):
                needed |= template.variables(text)
        missing = sorted(needed - set(inputs))
        if missing:
            issues.append(
                ConfigIssue(
                    f"{where}.inputs", f"missing value(s) for placeholder(s): {', '.join(missing)}"
                )
            )
    return issues


def load_suite(path: str | Path) -> LoadedSuite:
    """Parse and validate a suite file. Raises ConfigError listing every problem."""
    path = Path(path)
    source = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(source, [ConfigIssue("", f"cannot read file: {exc.strerror}")]) from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f"line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise ConfigError(source, [ConfigIssue(where, f"invalid YAML: {problem}")]) from exc
    if not isinstance(raw, dict):
        raise ConfigError(source, [ConfigIssue("", "a suite must be a YAML mapping")])

    base_dir = path.parent
    issues: list[ConfigIssue] = []
    suite: Suite | None = None
    try:
        suite = Suite.model_validate(raw, context={"base_dir": base_dir})
    except ValidationError as exc:
        for err in exc.errors(include_url=False):
            msg = str(err["msg"]).removeprefix("Value error, ")
            issues.append(ConfigIssue(_format_loc(err["loc"], raw), msg))
    issues += _cross_reference_issues(raw)
    if suite is not None:
        issues += pricing_issues(suite, suite.provider.model)
    if issues:
        raise ConfigError(source, list(dict.fromkeys(issues)))  # dedupe, keep order
    assert suite is not None
    return LoadedSuite(suite=suite, path=path, base_dir=base_dir)
