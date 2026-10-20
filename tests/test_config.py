from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from evalkit.config import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicProviderConfig,
    ConfigError,
    FakeProviderConfig,
    load_suite,
)
from tests.conftest import EXAMPLE_SUITE, MINIMAL_SCRIPT

GOOD = """
version: 1
name: tiny
provider: {type: fake, model: m, script: script.yaml}
prompt:
  system: "be terse"
  messages:
    - {role: user, content: "say {{ word }}"}
cases:
  - id: one
    inputs: {word: hello}
    assert:
      - {type: equals, value: HELLO}
"""


def _issues(exc: pytest.ExceptionInfo[ConfigError]) -> list[str]:
    return [str(i) for i in exc.value.issues]


def test_example_suite_loads() -> None:
    loaded = load_suite(EXAMPLE_SUITE)
    suite = loaded.suite
    assert suite.name == "receipts-extraction"
    assert len(suite.cases) == 8
    assert isinstance(suite.provider, FakeProviderConfig)
    assert suite.judge is not None
    assert suite.judge.min_kappa == 0.6
    assert "fake-extractor-a" in suite.prices
    # The schema file was resolved relative to the suite and inlined.
    first = suite.defaults.assert_[0]
    assert first.type == "json_schema"
    assert first.schema_["title"] == "Extracted receipt"
    assert len(loaded.fingerprint) == 64


def test_minimal_suite_and_defaults(write_suite: Callable[..., Path]) -> None:
    path = write_suite(GOOD, {"script.yaml": MINIMAL_SCRIPT})
    suite = load_suite(path).suite
    assert suite.run.trials == 5
    assert suite.gate.min_pass_rate == 1.0
    assert suite.cases[0].assert_[0].display_name() == "equals(HELLO)"


def test_anthropic_defaults_to_current_model() -> None:
    cfg = AnthropicProviderConfig(type="anthropic")
    assert cfg.model == DEFAULT_ANTHROPIC_MODEL == "claude-opus-5-5"
    assert cfg.max_tokens == 16_000


def test_every_error_is_reported_at_once(write_suite: Callable[..., Path]) -> None:
    path = write_suite(
        """
        version: 1
        name: broken
        provider: {type: fake, model: m, script: script.yaml}
        promt: typo-at-top-level
        prompt:
          messages:
            - {role: user, content: "{{ receipt }} in {{ lang }}"}
        defaults:
          assert:
            - {type: llm_judge, rubric: "mentions {{ topic }}"}
        cases:
          - id: a
            inputs: {receipt: x}
            assert:
              - {type: regex, pattern: "(unclosed"}
              - {type: json_schema, schema: missing.json}
          - id: a
            inputs: {receipt: y, lang: id, topic: t}
            assert:
              - {type: no_such_assertion}
              - {type: numeric_close, expected: "not a number"}
        """,
        {"script.yaml": MINIMAL_SCRIPT},
    )
    with pytest.raises(ConfigError) as info:
        load_suite(path)
    issues = _issues(info)
    text = "\n".join(issues)
    assert len(issues) >= 7, text
    assert "promt: Extra inputs are not permitted" in text
    assert "cases[0](a).assert[0].pattern: invalid regex" in text
    assert "cases[0](a).assert[1].schema: file not found: 'missing.json'" in text
    assert "cases[1](a).assert[0]: Input tag 'no_such_assertion'" in text
    assert "cases[1](a).assert[1].expected" in text
    assert "duplicate case id 'a'" in text
    assert "cases[0](a).inputs: missing value(s) for placeholder(s): lang, topic" in text
    assert "defaults.assert: llm_judge needs a top-level `judge:` section" in text
    assert info.value.render().startswith(f"{path}: {len(issues)} errors")


def test_yaml_syntax_error_has_position(write_suite: Callable[..., Path]) -> None:
    path = write_suite("name: [unclosed\nprovider: x\n")
    with pytest.raises(ConfigError) as info:
        load_suite(path)
    assert info.value.issues[0].path.startswith("line ")
    assert "invalid YAML" in info.value.issues[0].message


def test_prompt_must_end_with_user_turn(write_suite: Callable[..., Path]) -> None:
    body = GOOD.replace(
        '- {role: user, content: "say {{ word }}"}',
        '- {role: user, content: "say {{ word }}"}\n    - {role: assistant, content: "{"}',
    )
    with pytest.raises(ConfigError) as info:
        load_suite(write_suite(body, {"script.yaml": MINIMAL_SCRIPT}))
    assert any("start and end with a `user` turn" in str(i) for i in info.value.issues)


def test_max_cost_requires_a_price(write_suite: Callable[..., Path]) -> None:
    body = GOOD.replace("- {type: equals, value: HELLO}", "- {type: max_cost_usd, value: 0.01}")
    with pytest.raises(ConfigError) as info:
        load_suite(write_suite(body, {"script.yaml": MINIMAL_SCRIPT}))
    assert "no price for model 'm'" in str(info.value)
    priced = body.replace(
        "prompt:", "prices: {m: {input_per_mtok: 1, output_per_mtok: 2}}\nprompt:"
    )
    assert load_suite(write_suite(priced, {"script.yaml": MINIMAL_SCRIPT}, name="p.yaml"))


def test_bad_fake_script_and_bad_schema(write_suite: Callable[..., Path]) -> None:
    body = GOOD.replace(
        "- {type: equals, value: HELLO}", "- {type: json_schema, schema: {type: 12}}"
    )
    with pytest.raises(ConfigError) as info:
        load_suite(
            write_suite(body, {"script.yaml": "rules:\n  - when_regex: '('\n    responses: [x]\n"})
        )
    text = str(info.value)
    assert "provider.script.rules[0].when_regex" in text
    assert "not a valid JSON Schema" in text


def test_openai_provider_validation(write_suite: Callable[..., Path]) -> None:
    body = GOOD.replace(
        "provider: {type: fake, model: m, script: script.yaml}",
        "provider: {type: openai, model: m, base_url: 'ftp://x', temperature: 9}",
    )
    with pytest.raises(ConfigError) as info:
        load_suite(write_suite(body))
    text = str(info.value)
    assert "provider.base_url: base_url must start with http:// or https://" in text
    assert "provider.temperature" in text


def test_malformed_calibration_file_is_an_error(write_suite: Callable[..., Path]) -> None:
    body = GOOD + (
        "judge:\n  provider: {type: fake, model: j, script: script.yaml}\n"
        "  calibration: cal.json\n  min_kappa: 0.5\n"
    )
    with pytest.raises(ConfigError) as info:
        load_suite(write_suite(body, {"script.yaml": MINIMAL_SCRIPT, "cal.json": '{"x": 1}'}))
    assert "judge.calibration: cal.json is not a calibration file" in str(info.value)
    assert "(schema_version: Field required)" in str(info.value)
    # A calibration file that does not exist yet is fine (judge becomes advisory).
    later = body.replace("cal.json", "not-written-yet.json")
    assert load_suite(write_suite(later, {"script.yaml": MINIMAL_SCRIPT}, name="ok.yaml"))


def test_unreadable_suite(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read file"):
        load_suite(tmp_path / "nope.yaml")
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_suite(tmp_path / "list.yaml")
