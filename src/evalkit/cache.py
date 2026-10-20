"""On-disk response cache.

Key = sha256(provider identity, rendered prompt, sample index, response schema).
The sample index is part of the key on purpose: trial 3 of a case is a distinct
draw from the model, so rerunning a suite replays the exact same 10 draws, and
raising `trials` from 10 to 20 only pays for the 10 new ones.

Entries are written atomically (temp file + rename), so concurrent trials or an
interrupted run never leave a half-written entry; an unreadable entry is a miss.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evalkit.providers.base import Completion, CompletionRequest

CACHE_FORMAT = 1


class CacheMode(enum.StrEnum):
    USE = "use"  # read and write
    REFRESH = "refresh"  # ignore existing entries, write fresh ones
    OFF = "off"  # neither


def cache_key(provider_identity: Mapping[str, Any], request: CompletionRequest) -> str:
    payload = {
        "format": CACHE_FORMAT,
        "provider": provider_identity,
        "system": request.system,
        "messages": [[m.role, m.content] for m in request.messages],
        "sample_index": request.sample_index,
        "response_schema": request.response_schema,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, root: Path, mode: CacheMode = CacheMode.USE) -> None:
        self.root = root
        self.mode = mode
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        if self.mode is not CacheMode.USE:
            return None
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
            completion = Completion.from_dict(data["completion"])
        except (OSError, ValueError, KeyError, TypeError):
            self.misses += 1
            return None
        self.hits += 1
        return completion

    def put(self, key: str, completion: Completion) -> None:
        if self.mode is CacheMode.OFF:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"format": CACHE_FORMAT, "completion": completion.to_dict()})
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
