# evalkit

**LLM regression testing that runs in CI like unit tests.** Teams change a prompt or swap a model and silently break behaviour that used to work. evalkit runs a versioned YAML suite of cases against a model, repeats every case N times (LLMs are not deterministic, so one pass/fail is a coin flip), and answers the question a reviewer actually has: *did this change make things worse, or is that just noise?* It answers with statistics (Wilson intervals, Fisher's exact test with Holm correction, a two-stage bootstrap) and returns a meaningful exit code, a Markdown report for the PR, and JUnit XML for the CI dashboard.

```
$ make demo          # offline: scripted `fake` provider, no API key
...
$ evalkit diff runs/demo/baseline.jsonl runs/demo/candidate.jsonl
diff receipts-extraction: fake/fake-extractor-a (run 18c82f2346dd) -> fake/fake-extractor-b (run 9e0e5edfa438)
  pass rate 96.2% -> 83.8%; mean per-case delta -12.5 pp, 95% CI [-38.8 pp, +5.0 pp] (two-stage bootstrap over 8 cases): no significant change
  UNCHANGED  minimarket-cash         9/10 -> 8/10      -10.0 pp  p=1  p_holm=1
  REGRESSED  toko-bangunan-large    10/10 -> 0/10     -100.0 pp  p=1.1e-05  p_holm=8.7e-05
  UNCHANGED  bengkel-transfer        8/10 -> 9/10      +10.0 pp  p=1  p_holm=1
  (5 case(s) with identical pass counts not shown)
  power: with 10 vs 10 trials per case, a case at 100% must fall to <= 30.0% to be flagged (alpha 0.05, Holm over 8 cases).
result: REGRESSION (1 case(s) regressed)
$ echo $?
1
```

The candidate model reads `Rp 1.250.000,00` as 1250 on every trial: that is flagged. Two cases that moved by one trial in either direction are correctly reported as noise. The suite-level interval still straddles zero because a single bad case out of eight is not a suite-wide trend, which is why both levels exist.

## What it demonstrates

| Concern | Solution | Where |
|---|---|---|
| Suite definition | YAML with a strict pydantic schema (unknown keys rejected). Prompt template (system + messages, `{{ var }}` single-pass substitution, so no template injection from test data). Eight assertions: `equals`, `contains`, `regex`, `json_schema`, `numeric_close`, `max_latency_ms`, `max_cost_usd`, `llm_judge`; the JSON-aware ones take a `path` like `items.0.qty`. Schema, script and price files are resolved relative to the suite. | `config.py`, `assertions.py`, `template.py` |
| Config errors | Structural errors (pydantic) and cross-reference errors (missing placeholder inputs, duplicate ids, `llm_judge` without a judge) are collected in **one pass** and reported together, with paths like `cases[3](toko-bangunan-large).assert[1].pattern`. `evalkit validate` never calls a model. | `config.py` |
| Nondeterminism | Every case runs `n` trials and gets a pass rate with a **Wilson score interval**, which stays inside [0, 1] and is sensible at 0/n and n/n, where LLM pass rates usually sit. | `runner.py`, `stats.py` |
| Regression vs noise | Per case: **Fisher's exact test** on pass counts, computed with exact integer binomials (no epsilon for ties), **Holm-adjusted** across cases. Suite level: mean per-case delta with a **two-stage bootstrap** CI (cases, then trials). A power line says how large a drop the gate can actually see at this `n`. | `diff.py`, `stats.py` |
| LLM-as-judge | A 1-5 rubric score, requested as structured JSON output. Each judge is **fingerprinted** (judge model + rubric + threshold). `evalkit calibrate` compares verdicts with human labels and reports agreement, **Cohen's kappa** and a confusion matrix. With `min_kappa` set, a judge below the threshold (or with a stale calibration) is **advisory**: reported, but not a CI gate. | `judge.py`, `calibrate.py` |
| Providers | One async interface. **Anthropic Messages API** via the official `anthropic` SDK (default `claude-opus-5-5`; `effort` instead of temperature; server-side refusal fallback recorded per trial); a **generic OpenAI-compatible** `/chat/completions` client; a deterministic **`fake`** provider with scripted answers and scripted 429/5xx failures. | `providers/` |
| Resilience | **Capped full-jitter backoff** on 408/409/429/5xx, connection errors and timeouts, honouring `Retry-After`. The SDK's own retries are turned off, so exactly one retry policy is in force and attempts are recorded per trial. A global semaphore caps in-flight calls. | `retry.py`, `runner.py` |
| Cost & reproducibility | **On-disk response cache** keyed by `sha256(provider identity, model params, rendered prompt, sample index)`, written atomically. Reruns are free and identical, and raising `trials` only pays for the new samples. **Token cost** comes from a user-supplied price table, per model actually billed (a fallback hop is priced at its own model's rate). | `cache.py`, `pricing.py` |
| Run files | JSONL: a self-describing header (gate thresholds, cases, judge gating decisions), then one record per trial (output, every assertion result, latency, tokens, cost, attempts, cache hit, served model, stop reason, error). `report` and `diff` need nothing else, e.g. an artifact from another branch. | `runstore.py` |
| CI gate | Exit codes `0` ok, `1` gate failed / significant regression, `2` invalid input, `3` provider or judge errors (an outage must not read as a regression, and vice versa). Markdown for PR comments, JUnit XML, and an example GitHub Actions workflow that diffs the PR head against its base commit. | `cli.py`, `report.py`, `examples/github-actions/` |
| Testing | 121 tests. Statistics are checked against **scipy/numpy as oracles with hypothesis**. Provider adapters are tested through the real SDK/HTTP code path against `httpx2.MockTransport` (no network, no keys). The CLI is tested end to end. `make demo` asserts each command's exit code. CI runs lint, strict mypy, tests and the demo on Python 3.11-3.13. | `tests/`, `.github/workflows/ci.yml` |

Runtime dependencies: `anthropic`, `httpx2`, `pydantic`, `PyYAML`, `jsonschema`, `typer`. The statistics module is dependency-free; scipy is a test-only oracle.

## Run

```bash
make demo                       # the whole flow above, offline
```

or step by step:

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
evalkit validate examples/receipts/suite.yaml
evalkit run  examples/receipts/suite.yaml --out runs/base.jsonl
evalkit run  examples/receipts/suite.yaml --out runs/cand.jsonl --model fake-extractor-b \
             --markdown runs/cand.md --junit runs/cand.xml
evalkit diff runs/base.jsonl runs/cand.jsonl --markdown runs/diff.md
evalkit report runs/cand.jsonl                    # re-apply the gate to a stored run
evalkit calibrate runs/base.jsonl examples/receipts/human_labels.csv --out cal.json --min-kappa 0.6
```

A suite, abridged (the full example is [`examples/receipts/suite.yaml`](examples/receipts/suite.yaml)):

```yaml
name: receipts-extraction
provider: {type: anthropic, model: claude-opus-5-5, effort: low}   # or openai / fake
judge:
  provider: {type: anthropic, model: claude-opus-5-5}
  calibration: calibration.json
  min_kappa: 0.6                  # below this the judge is advisory, not a gate
run: {trials: 10, concurrency: 8}
gate: {min_pass_rate: 0.8, min_suite_pass_rate: 0.9}
prices: prices.yaml               # example values; verify against your provider
prompt:
  system: "Extract the receipt as JSON ..."
  messages: [{role: user, content: "<receipt>{{ receipt }}</receipt>"}]
defaults:
  assert:
    - {type: json_schema, schema: receipt.schema.json}
    - {type: max_latency_ms, value: 3000}
    - {type: llm_judge, rubric: "Every item in `items` appears on the receipt ..."}
cases:
  - id: toko-bangunan-large
    inputs: {receipt: "... GRAND TOTAL  Rp 1.250.000,00 ..."}
    assert:
      - {type: numeric_close, path: total, expected: 1250000, abs_tol: 0.5}
      - {type: equals, path: payment_method, value: transfer}
```

Real providers read credentials from the environment only (`ANTHROPIC_API_KEY`, or the variable named by `api_key_env` for OpenAI-compatible servers); see `.env.example`. The example suite and receipts are synthetic: "Contoh", "Fiktif", "Sampel" and "Uji Coba" mean example, fictional, sample and trial.

**Prices are not authoritative.** `examples/receipts/prices.yaml` contains example values. Prices change and have tiers this file does not model, so check your provider's pricing page before relying on any cost evalkit prints. A model with no price yields cost `n/a`, never `$0`, and `max_cost_usd` refuses to validate without one.

## Design notes

**Why repeat every case?** A single run of a case that passes 70% of the time is a 70/30 coin. Comparing two such coins once tells you nothing, and a prompt change that looked like a regression in a single run gets reverted for no reason. Repeating gives a rate. The Wilson interval is used rather than the textbook normal approximation because that one collapses to `[1, 1]` at 10/10 and claims certainty it does not have.

**Why Fisher plus Holm per case, and why print the power?** Trial counts are small (an LLM call costs money), so normal-approximation tests are wrong exactly where they are used. Fisher's exact test is exact at any `n`. The catch is multiple comparisons: a per-case test at α = 0.05 is allowed a 5% false-alarm rate per case, so a 40-case suite can expect up to two spurious "regressions" per PR, and people learn to ignore the gate. Holm keeps the chance of *any* false alarm across the suite at α. The cost is power, so `diff` prints the smallest drop it could detect: at 10 vs 10 trials over 8 cases, a perfect case has to fall to 30%. If that is not good enough, the fix is more trials, not a looser test. The test is two-sided, which makes it slightly conservative for regressions; for a gate that blocks merges, that is the right side to err on.

**A bug I hit: the suite-level interval was too confident.** The first version of the suite delta resampled *cases* only (a paired bootstrap over per-case differences). A unit test where a one-case suite went 5/5 → 4/5 came back as a *significant* suite regression. With identical per-case deltas, resampling cases always reproduces the same mean, so the interval has zero width; it ignored the fact that each pass rate is itself a noisy 5-trial estimate. The fix is a two-stage bootstrap: resample cases, then resample trials within each chosen case. The same test now gives a CI of [-60, 0] pp. Two-stage bootstraps are known to be slightly conservative, which is again the right direction for a gate. The single-case decision belongs to Fisher anyway.

**A judge is an instrument, so calibrate it.** An `llm_judge` gate is only as good as its agreement with people. The demo shows the trap: the scripted judge agrees with the (synthetic) human labels on **97.5%** of trials, yet **κ = 0.49**. Nearly everything passes, so even "always pass" would agree about 95% of the time, and kappa corrects for that chance agreement. evalkit therefore keeps it advisory under `min_kappa: 0.6`. The calibration is keyed by a fingerprint of judge model, rubric and threshold, so editing the rubric silently invalidates it instead of silently reusing it, and a test in this repo fails if the committed calibration no longer matches the suite. Kappa is unstable when one class is rare: label a sample that deliberately includes failures.

**The cache key includes the sample index.** Trial 3 of a case is a distinct draw, so it gets its own cache entry. That makes reruns bit-for-bit reproducible, lets `trials: 10 → 20` pay for 10 new calls, and means the base-branch run in the example CI workflow is mostly served from cache. The trade-off is that a cached rerun is not a fresh sample. If the provider changes the model behind an id, `--cache refresh` re-draws. Keys use the *rendered* prompt, so editing one case's input only re-runs that case.

**One retry loop, not two.** The Anthropic SDK retries on its own by default. Stacked under evalkit's loop, one 429 could turn into 3 × 5 requests, with attempts invisible to the run. The SDK is constructed with `max_retries=0`, and evalkit's loop does capped full-jitter backoff (`U(0, min(cap, base·2^k))`, raised to `Retry-After`). A request holds its concurrency slot while it backs off. Releasing the slot would let another request walk straight into the same rate limit.

**Errors are not failures.** A trial whose provider call fails after all retries is recorded with its error and counted as not passing, but the exit code is `3`, not `1`, so an outage cannot be mistaken for a regression. A judge reply that is not valid JSON is also infrastructure (the instrument broke), not a verdict on the model under test.

**Anthropic-specific choices.** Current Claude models reject `temperature`/`top_p` (and the 1.x SDK no longer accepts them), so the provider exposes `effort` instead. Thinking blocks are not treated as output. Server-side refusal fallback (`fallbacks: "default"`) is on by default for models that support it, because a declined request otherwise yields no answer at all. For an eval, though, a fallback means *a different model answered*. So every trial records `served_model`, the report counts trials served by a fallback model, and per-attempt usage is priced at each model's own rate. Set `fallbacks: none` to measure refusals as failures instead.

**Absolute gate vs. statistical diff.** `run` gates on the observed pass rate against `min_pass_rate`. That is predictable and unit-test-like, and the Wilson interval is printed next to it so a human can see how much to trust it. The statistical machinery lives in `diff`, where the question is comparative and the noise model matters.

## Benchmarks

Measured with `make bench` on a 4 vCPU cloud VM (Intel Xeon 2.8 GHz), Python 3.13, fake provider. This is evalkit's own overhead; real runs are dominated by model latency.

| 200 cases x 10 trials, 5 assertions per trial | wall time | per trial |
|---|---|---|
| cache off | 0.20 s | ~99 µs |
| cold cache (2000 atomic writes) | 1.23 s | ~0.6 ms |
| warm cache (all hits) | 0.32 s | ~160 µs |
| warm cache + `llm_judge` (2 cached calls per trial) | 0.53 s | ~265 µs |

| operation | time |
|---|---|
| 200 calls at 20 ms simulated latency, concurrency 1 / 8 / 32 | 4.15 s / 0.54 s / 0.17 s |
| `diff`, 8 cases x 10 trials, 10k bootstrap resamples | 72 ms |
| `diff`, 200 cases x 10 trials, 10k bootstrap resamples | 891 ms |
| Fisher exact, 1000 vs 1000 trials (exact integers) | 0.07 ms |

A run file for 2000 trials is about 2.1 MiB of JSONL.

## Layout

```
src/evalkit/
  cli.py              typer CLI: validate, run, report, diff, calibrate; exit codes
  config.py           suite schema, file resolution, one-pass validation
  runner.py           cases x trials, cache -> semaphore -> retries -> assertions -> judge
  assertions.py       deterministic assertions, JSON paths
  judge.py            llm_judge prompt, verdict parsing, fingerprint, gating decision
  stats.py            Wilson, Fisher exact, Holm, two-stage bootstrap, Cohen's kappa
  diff.py             run vs run: per-case tests, suite delta, power, rendering
  calibrate.py        judge vs human labels
  report.py           summaries, gate, terminal / Markdown / JUnit
  runstore.py         JSONL run format
  cache.py            content-addressed response cache
  retry.py            capped full-jitter backoff
  pricing.py          tokens x price table
  template.py         {{ var }} rendering
  providers/          base interface, anthropic (SDK), openai_compat (HTTP), fake
examples/receipts/    synthetic Indonesian receipt suite, scripted models, labels
examples/github-actions/evalkit.yml   PR gate: base vs head, cached, PR comment
bench/                harness benchmarks (make bench)
scripts/demo.sh       the offline demo, asserting exit codes
tests/
```

## Develop

```bash
make test        # 121 tests, ~8 s, offline
make lint        # ruff check + ruff format --check + mypy (strict)
make fmt
make demo        # offline end-to-end, asserts exit codes
make bench
```

## Status / not yet done

- **No live API calls were made while building this.** The Anthropic and OpenAI-compatible adapters are tested against mocked HTTP responses shaped like the documented APIs, through the real SDK and HTTP code paths. They have not been exercised against live endpoints here, and the first real run may surface details the mocks do not cover.
- **The demo's numbers come from scripted data.** The pass rates, the 97.5% agreement and κ = 0.49 are properties of `fake_model.yaml`, `fake_judge.yaml` and hand-written synthetic labels. They show the mechanics; they are not measurements of any real model or judge.
- The list of models that accept server-side refusal fallback is hard-coded and will go stale. Server-side "sticky" fallback routing can serve later trials of the same prompt from the fallback model; this is recorded per trial but not prevented.
- Single-turn completions only: no tool use, no streaming. `max_tokens` defaults to 16k on non-streaming requests; very long outputs would need streaming.
- Calibration is binary (pass/fail). There is no ordinal or weighted kappa on the 1-5 scores, and no multi-judge ensembles.
- No sequential testing: trial counts are fixed up front rather than stopping early once a result is clear.
- The cache has no eviction; delete `.evalkit-cache/` to reclaim space.
- Not published to PyPI.
