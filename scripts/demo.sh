#!/usr/bin/env bash
# Offline demo on the scripted `fake` provider: baseline vs candidate, the way a CI
# job would run it. Every step asserts its exit code, so this doubles as a smoke test.
set -euo pipefail
cd "$(dirname "$0")/.."

EVALKIT=${EVALKIT:-evalkit}
OUT=${OUT:-runs/demo}
SUITE=examples/receipts/suite.yaml

rm -rf "$OUT"
mkdir -p "$OUT"
export EVALKIT_CACHE_DIR="$OUT/cache"

expect() { # expect <exit code> <command...>
  local want=$1
  shift
  printf '\n$ %s\n' "$*"
  set +e
  "$@"
  local got=$?
  set -e
  if [[ $got -ne $want ]]; then
    echo "demo: expected exit $want, got $got" >&2
    exit 1
  fi
  echo "(exit $got, as expected)"
}

# 1. The suite is valid.
expect 0 "$EVALKIT" validate "$SUITE"

# 2. Baseline passes its gate. Note the scripted 429/503s on apotek-ppn are retried.
expect 0 "$EVALKIT" run "$SUITE" --out "$OUT/baseline.jsonl" --label variant=baseline

# 3. Re-running is free: every response comes from the on-disk cache.
expect 0 "$EVALKIT" run "$SUITE" --out "$OUT/baseline-rerun.jsonl" --quiet

# 4. The candidate model fails the gate (exit 1) and writes PR-ready reports.
expect 1 "$EVALKIT" run "$SUITE" --model fake-extractor-b --out "$OUT/candidate.jsonl" \
  --markdown "$OUT/candidate.md" --junit "$OUT/candidate.junit.xml" --quiet

# 5. The diff flags the real regression and ignores the noise.
expect 1 "$EVALKIT" diff "$OUT/baseline.jsonl" "$OUT/candidate.jsonl" --markdown "$OUT/diff.md"

# 6. The judge agrees with (synthetic) human labels 97.5% of the time, but kappa
#    is below 0.6, so it stays advisory.
expect 1 "$EVALKIT" calibrate "$OUT/baseline.jsonl" examples/receipts/human_labels.csv \
  --out "$OUT/calibration.json" --min-kappa 0.6

printf '\nArtifacts: %s\n' "$(cd "$OUT" && ls -1 ./*.jsonl ./*.md ./*.xml ./*.json | tr '\n' ' ')"
