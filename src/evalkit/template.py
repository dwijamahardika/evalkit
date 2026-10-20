"""Minimal `{{ name }}` prompt templating.

Deliberately not Jinja: prompts are data, not programs. Substitution is a single
regex pass, so a case input that itself contains `{{ x }}` is inserted verbatim and
never re-expanded (no template injection from test data).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_VAR = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class TemplateError(KeyError):
    """A placeholder had no value."""


def variables(template: str) -> set[str]:
    """Names referenced by `template`."""
    return set(_VAR.findall(template))


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render(template: str, values: Mapping[str, Any]) -> str:
    """Substitute every `{{ name }}`; non-string values are inserted as JSON."""

    def sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise TemplateError(name)
        return _stringify(values[name])

    return _VAR.sub(sub, template)
