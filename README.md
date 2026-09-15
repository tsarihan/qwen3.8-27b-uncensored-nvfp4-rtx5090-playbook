# Qwen3.8-27B-Uncensored-NVFP4 on a single RTX 5090 (SM120, 32 GB)

Serving `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` — an **abliterated** (refusal-removed)
build of Qwen3.8-27B, quantized to mixed NVFP4 + FP8 — on one RTX 5090 under WSL2, with
throughput, latency and long-context retrieval measured the same way as the other
playbooks in this series.

Most of what cost time here was not the model. It was four traps in the serving path that
each fail *late*, only once an agent's context has grown, and two of which silently
discard a quarter of the context window. Those are documented in full below.

---

## Scope and intended use

This checkpoint has had its refusal direction removed. Per the publisher's own card,
harmful-prompt refusal drops from **64–99%** (base Qwen3.8-27B) to **0–6%**, and the card
states plainly that it has **no meaningful built-in guardrails**.

**This repository documents serving and benchmarking only.** It is published to support
authorized security testing, red-team evaluation, and refusal-mechanism research in
isolated environments. It contains no harmful content, no jailbreak material, and no
capability demonstrations — only engine configuration, throughput/latency measurements,
and needle-in-a-haystack retrieval scores.

Do not deploy an abliterated model to end users or to any public endpoint without an
independent safety and moderation layer in front of it. The base
`Qwen/Qwen3.8-27B` is the correct choice for essentially every other purpose, and this
playbook's serving configuration applies to it almost unchanged.

## Status: incomplete on purpose

Two planned measurements are **not in this repository yet** and will be added:

* **SWE-bench Pro, 40-instance enterprise subset.** Runs are in progress, at 1 worker, in
  two arms — thinking on at `xhigh` and thinking off — to isolate what the reasoning trace
  is worth on agentic coding. Design and the harness traps are documented below; only the
  scores are missing.
* **Comparison against the non-abliterated model.** The control is
  `Qwen3.8-27B-NVFP4` on this same card and the same engine flags, so abliteration is the
  only variable. It has not been run yet.

Until that control lands, **nothing here says what abliteration cost in capability.** The
numbers below characterize this lane, not the weight edit. The publisher reports capability
within ±1.3 points of base (MMLU 84.3 → 84.7, GSM8K 90.0 → 88.7) — but for the **FP8**
build, and the card explicitly says quant-specific numbers for *this* NVFP4 checkpoint
have not been measured.

## Headline

| | measured |
|---|---|
| Weights on disk | 24,718,873,492 bytes (42 files) |
| GPU KV cache | **139,810 tokens** (96 blocks @ 1568) |
| `kv_cache_max_concurrency` | **1.07** |
| Max context served | 131,072 (163,840 does not fit — see below) |
| Decode @ c=1 | **61.5 tok/s** |
| Aggregate @ c=16 | **836.9 tok/s** |
| Aggregate @ c=32 and c=64 | 837.6 / 836.4 — flat, no gain |
| TTFT @ c≤16 | ≤ 0.36 s |
| TTFT max @ c=64 | 10.2 s |
| NIAH 5 needles, 4K → 122K | **5/5 at every depth**, no truncation |
| Prefill @ 32K | 7,403 tok/s |
| Prefill @ 122K | 4,161 tok/s |

Two things worth reading off that table.

**Concurrency is capped by `--max-num-seqs`, not by compute or KV.** Aggregate throughput
scales near-linearly to 16 streams and is then dead flat: 836.9 / 837.6 / 836.4 at 16 / 32 /
64. Past 16 you buy no throughput and pay up to 30× TTFT.

**KV is the binding constraint everywhere else.** 139,810 tokens with
`kv_cache_max_concurrency` of 1.07 means the cache holds barely more than *one*
full-length request. That number, not `--max-num-seqs`, is what should size any real
workload — see "Do not size workers from the ladder" below.

## What fits on 32 GB

`--max-model-len 163840` fails at startup:

```
ValueError: ... max seq len (163840), 5.31 GiB KV needed > 4.62 GiB available ...
estimated maximum model length is 141120
```

A 24.7 GB checkpoint at `--gpu-memory-utilization 0.945` on a 32 GB card leaves about
4.6 GiB for KV. 131,072 is the largest round window that fits with headroom; the card's
recommended 262,144 is not reachable on this hardware at any utilization.

`--kv-cache-dtype fp8` is what makes even 131,072 viable, and it is also a deviation from
the card — see below.

## Concurrency ladder

`scripts/sweep.py`, `max_tokens 512`, one distinct nonce per request so prefix caching
cannot let later streams reuse earlier prefill. Direct to vLLM, not through a proxy.

| conc | ok | fail | TTFT mean | TTFT p50 | TTFT max | tok/s per stream | aggregate |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0 | 0.066 | 0.066 | 0.066 | 61.52 | 61.1 |
| 2 | 2 | 0 | 0.135 | 0.135 | 0.136 | 56.82 | 112.1 |
| 4 | 4 | 0 | 0.149 | 0.153 | 0.153 | 55.41 | 218.4 |
| 8 | 8 | 0 | 0.358 | 0.364 | 0.365 | 57.20 | 440.6 |
| 16 | 16 | 0 | 0.291 | 0.298 | 0.302 | 53.84 | **836.9** |
| 32 | 32 | 0 | 3.965 | 0.347 | 10.061 | 53.77 | 837.6 |
| 64 | 64 | 0 | 5.793 | 9.858 | 10.213 | 53.63 | 836.4 |

Zero failures and zero preemptions at every rung.

The **bimodal TTFT at c=32** is the tell that you have hit `--max-num-seqs`: p50 is 0.347 s
but max is 10.061 s. The first 16 requests admit immediately; the rest wait a full decode
cycle. Mean TTFT alone hides this — at c=64 the mean (5.793) is *lower* than p50 (9.858),
which only makes sense once you realize the distribution has two modes.

Per-stream decode barely sags across the whole ladder, 61.5 → 53.6, a 13% loss for 64×
the load. Nothing here is compute-starved.

Operating points:

* latency-sensitive agent work — **≤ 8 streams** (TTFT ≤ 0.36 s, 57 tok/s per stream)
* maximum throughput — **16 streams** (837 tok/s aggregate, TTFT 0.29 s)
* never — **> 16 streams**

### Do not size an agent workload from this ladder

`sweep.py` sends ~133-token prompts. KV never binds, so the ladder reports clean linear
scaling to `--max-num-seqs` that **does not transfer** to multi-turn agent load, where
contexts grow into the tens of thousands of tokens and evict each other.

Measured on SWE-bench Pro with this same lane:

| workers | requests/hour | preemptions |
|---|---|---|
| 3 | ~387 | 47/hr |
| 1 | ~275 | **0** |

Three workers is ~40% more throughput; one worker is the cleaner measurement. Size against
`kv_cache_size_tokens` versus real turn-grown context, and watch
`vllm:num_preemptions_total` — a climbing preemption count is the signal that concurrency
is past what the cache can retain.

A cautionary note on measuring this: a **180-second** sample of that 1-worker run showed
~520 req/hr, implying 1 worker beat 3. Over 30 minutes it fell to 275. Early agent steps
have short contexts and fast commands and are not representative. Measure over ≥ 30
minutes, and prefer `vllm:e2e_request_latency_seconds` and
`vllm:time_to_first_token_seconds` histogram deltas over requests/hour, which charges
tool-execution time to the model.

## Needle in a haystack, 5 needles

`scripts/needles.py`, 5 needles at depths 0.10 / 0.30 / 0.50 / 0.70 / 0.90, distinct nonce
per run, token counts verified against the server's `/tokenize` rather than estimated.

| target | prompt tokens | TTFT | prefill tok/s | decode tok/s | completion | needles |
|---|---|---|---|---|---|---|
| 4,096 | 4,313 | 0.766 s | 5,627.8 | 61.29 | 346 | **5/5** |
| 32,768 | 33,372 | 4.508 s | 7,403.1 | 58.99 | 619 | **5/5** |
| 65,536 | 66,568 | 11.616 s | 5,731.0 | 58.80 | 1,000 | **5/5** |
| 122,880 | 124,688 | 29.969 s | 4,160.5 | 54.97 | 1,172 | **5/5** |

20/20 needles, at every depth and every context. Max context with all needles retrieved:
**124,688**.

**No answer was truncated** — the largest completion was 1,172 against the 2,048 cap, so
every 5/5 is a genuine retrieval rather than a capped or unfinished answer. This check
matters: with a reasoning model the trace can consume the whole output budget and leave
`content` empty, which reads as a missed needle. Treat `completion_tokens == max_tokens`
as a truncation flag and re-run that rung before calling anything missed.

The top rung is 122,880 and not 131,072 on purpose. A 131,072 *target* produces roughly
132,970 actual prompt tokens, which exceeds this lane's window and hard-fails with a 400.

Decode holds 55–61 tok/s independent of context length. Prefill peaks at 32K and decays to
4,161 tok/s at 122K as attention cost grows with sequence length.

## Thinking: `xhigh` is the default and the maximum

The chat template resolves:

```jinja
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ...') }}
```

So:

* the accepted set is **`xhigh` / `medium` / `low`**. There is no `"max"` — passing one
  raises, which at least fails loudly instead of silently degrading.
* `xhigh` is both the **default** and the **maximum**. Serving with no reasoning parameter
  at all already runs at maximum effort.
* thinking is **on** unless you explicitly pass `enable_thinking: false`.

Toggle per request via `chat_template_kwargs`. Measured through a litellm proxy:

| request | reasoning | completion tokens | "which is larger, 9.11 or 9.9?" |
|---|---|---|---|
| no kwargs (default) | 82 chars | 48 | 9.9 ✓ |
| `enable_thinking: true`, `reasoning_effort: xhigh` | 180 chars | 91 | 9.9 ✓ |
| `enable_thinking: false` | **0 chars** | 8 | **9.11** ✗ |

Thinking off gets the classic decimal comparison wrong. Whatever the trace costs in
tokens, it is doing work.

### Thinking does not restore refusals

A common assumption about abliterated models is that reasoning lets the model recover the
refusal behavior that was edited out, so you should serve with thinking off. For this
checkpoint the publisher's own data says the opposite: refusal is **0–6% with thinking
off** and **≤ 1.7% with thinking on**. Thinking on refuses *less*.

Abliteration is a weight edit, not a prompt-level behavior, so it holds across both
template branches. The only `enable_thinking=False` in the card is in the **quantization
calibration** recipe (512 samples at sequence length 2048, used to fit activation scales)
— that is a calibration detail, not serving guidance. The card states: "Thinking is **on by
default**."

Practical consequence: turning thinking off costs capability without buying compliance.

### Tool calls work on both template branches — verify yours

The no-think branch takes a different path through `chat_template.jinja`, so tool-call
parsing needs checking separately on each. Both were verified here
(`scripts/toolcall_nothink.py`):

```
thinking ON (xhigh)   finish=tool_calls   reasoning=  70ch  tool_calls=YES
thinking OFF          finish=tool_calls   reasoning=   0ch  tool_calls=YES
```

This is worth a 30-second check before any long agent run. A model that cannot emit a
parseable tool call does not fail fast — it burns its entire step budget on format-error
retries and submits an empty patch, per instance, for hours.

## Four traps in the serving path

None of these are model bugs. All of them fail late, after an agent's context has grown,
which is the worst time to find them.

### 1. litellm injects `max_tokens` *after* the pre-call hook, silently capping context

The one that actually cost a benchmark run. An instance died with:

```
maximum context length is 131072 tokens. However, you requested 32768 output tokens
and your prompt contains at least 98305 input tokens, for a total of at least 131073
```

One token over. The cause is ordering. mini-swe-agent sends **no** `max_tokens`. A
pre-call hook that clamps output length therefore sees nothing to clamp and returns early
— and litellm *then* injects the lane's configured `max_tokens` from `config.yaml`. The
hook never sees the value it exists to bound.

With `max_tokens: 32768` configured, usable input is capped at `131072 - 32768 = 98304`.
**A quarter of the window is unreachable**, and you only discover it when a context crosses
that line hours into a run.

Fix: make the *configured* value fit, because the hook cannot help here.

```yaml
  - model_name: "qwen3.8:27b-orca"
    litellm_params:
      max_tokens: 8192            # was 32768
      max_input_tokens: 131072
```

Verified against the exact failure (`scripts/verify_longctx.py`) — prompts of 98,559 /
115,059 / 122,059 tokens with no client `max_tokens` all return 200 where 98,305 previously
400'd. Usable input goes 98,304 → ~122,880. 8192 is generous: measured mean generation for
an agent step is ~300 tokens, *including* the `xhigh` reasoning trace.

Generalizes to any litellm lane whose client omits `max_tokens`. The symptom is instances
failing only once context passes `window − max_tokens`.

### 2. Substring matching in a ceiling table hands a model the wrong window

If you keep a per-model output-clamp table keyed by substring with first-hit-wins, ordering
is load-bearing:

```python
_CEILINGS = [
    ("qwen3.8:27b-orca",    131_072),   # MUST precede the bare key below --
    ("qwen3.8:27b",         163_840),   # ...which is a substring of it
]
```

`"qwen3.8:27b"` is a substring of `"qwen3.8:27b-orca"`, so without the explicit row first,
orca inherits the *base* lane's 163,840 ceiling against a real 131,072 window — the clamp
then permits `prompt + output` up to 155,648 and lets vLLM 400 instead of trimming.
Included as `patches/gbnf_pattern_fix.py`.

### 3. A moved directory left a hardcoded path behind

A reorganization moved the harness tree and left the old path as an empty directory. The
runner still had `BASE=~/swe-enterprise40` hardcoded, so `wc -l < "$DS"` produced an empty
`TOTAL` and the script died on `set -u`. Made overridable:

```bash
BASE="${BASE:-$HOME/swepro/enterprise40}"
```

Trivial, but it presents as an unrelated shell error and costs a debugging cycle.

### 4. `pgrep -f` matches the shell that is running it

Every progress watcher written as

```bash
while pgrep -f "sweep.py --base-url http://127.0.0.1:8138" >/dev/null; do sleep 20; done
```

never exits, because the watcher's own command line contains that string. Worse, a `kill`
built the same way sends the signal to your own shell. Use the bracket trick —
`pgrep -f "sweep[.]py --base-url"` — or capture the PID at launch and poll `kill -0 "$PID"`.

This produced a false "still running" reading on an already-finished sweep, and a false
"still alive" reading on an already-stopped harness.

## Deviations from the card

Held constant across every measurement here so they cancel out of any internal comparison,
but they are real and matter if you quote these as the *checkpoint's* numbers:

| the card says | this lane | why |
|---|---|---|
| do **not** pass `--quantization` / `--kv-cache-dtype` | passes both | `config.json` already carries the mixed-precision config *including* an fp8 `kv_cache_scheme` with **calibrated static** scales; `--kv-cache-dtype fp8` may substitute dynamic ones |
| `--tool-call-parser qwen3_coder` | `qwen3_xml` | works, verified on both branches — but it is not what the publisher specifies |
| vLLM **≥ 0.27** (`v0.27.1`) | **v0.26.0** | below the stated minimum; loads and serves correctly |
| `--speculative-config` MTP, `num_speculative_tokens: 2` | omitted | the MTP draft head is preserved in the checkpoint (`model-extra-00001-of-00001.safetensors`, 0.85 GB); left off to keep one fewer variable |
| `--max-model-len 262144` | 131,072 | does not fit on 32 GB (see above) |
| `--max-num-seqs 96` | 16 | ~4.6 GiB of KV |

The KV-scale one is the substantive deviation and should be revisited before treating these
as capability numbers. `KV_DTYPE=auto` in the serve script follows the card instead.

## Provenance: why orcarouter and not a third-party requant

Quantization reprocesses every weight, so it is an opportunity to alter behavior, and
altered weights cannot practically be audited after the fact. Benchmarks do not catch
trigger-conditioned behavior — a checkpoint that scores normally can still carry something
that surfaces only on specific inputs. Provenance is the only real control.

**orcarouter quantized its own modified weights.** Alternative builds of this abliteration
are third-party requants — an additional, independent opportunity to alter weights, by a
party who did not produce the original edit. For an already-modified checkpoint that is the
wrong direction to compound risk in.

Independent benchmarking by Nathan Sapwell (`nathan.sapwell.net`, 9 abliterated variants,
~167 GPU-hours, with tensor forensics checking each card's claims against the actual
weights) ranked this build best of the abliterations: 82% HarmBench compliance at
near-baseline capability, KL drift 0.04–0.08 versus base. Note those are for the **FP8**
build and that Sapwell is an independent reviewer, not the publisher.

Verify before serving, always:

```bash
python3 scripts/verify-nvfp4.py     # sha256 of every LFS file vs the publisher's hashes
# -> RESULT: VERIFY_OK, 7/7
```

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB (SM120, Blackwell) |
| Driver | 610.74 |
| OS | Ubuntu 24.04.4 LTS on WSL2, kernel 6.6.123.2-microsoft-standard-WSL2+ |
| Image | `vllm/vllm-openai:v0.26.0` |
| Model | `orcarouter/Qwen3.8-27B-Uncensored-NVFP4`, 24,718,873,492 bytes, 42 files |
| Arch | `Qwen3_5ForConditionalGeneration`, vision tower preserved (167 `visual.*` tensors in BF16) |
| Quant split | 168 NVFP4 / 233 FP8 / 206 BF16 linears, `format: mixed-precision` |

FP4 layers require Blackwell FP4 tensor cores. The FP8 layers run on Hopper-class and newer.

## Reproducing

```bash
# 1. verify the checkpoint against the publisher's LFS sha256 before serving 24 GB
python3 scripts/verify-nvfp4.py

# 2. serve
./scripts/serve-qwen38-27b-orca-vllm.sh
#    KV_DTYPE=auto            follow the card on KV scales
#    TOOL_PARSER=qwen3_coder  follow the card on the tool parser
#    SPEC='{"method":"mtp","num_speculative_tokens":2}'   enable the MTP draft head

# 3. confirm tool calls parse on BOTH template branches before any agent run
python3 scripts/toolcall_nothink.py

# 4. confirm the thinking toggle survives your proxy
python3 scripts/toggle_verify.py

# 5. if you serve through litellm, confirm long contexts do not 400
python3 scripts/verify_longctx.py

# 6. measure
python3 scripts/sweep.py   --base-url http://127.0.0.1:8138/v1 --model qwen3.8:27b-orca \
                           --concurrency 1,2,4,8,16,32,64 --max-tokens 512 \
                           --tag orca --out results/sweep-qwen38-27b-orca.json
python3 scripts/needles.py --base-url http://127.0.0.1:8138/v1 --model qwen3.8:27b-orca \
                           --contexts 4096,32768,65536,122880 --needles 5 \
                           --tag orca --out results/niah-qwen38-27b-orca.json
```

## Files

```
scripts/serve-qwen38-27b-orca-vllm.sh   single-card launcher, all knobs are env vars
scripts/verify-nvfp4.py                 checkpoint vs publisher LFS sha256
scripts/sweep.py                        concurrency ladder: aggregate, per-stream, TTFT
scripts/needles.py                      multi-needle retrieval across a context sweep
scripts/toolcall_nothink.py             tool calls on BOTH template branches
scripts/toggle_verify.py                enable_thinking survives the proxy hop
scripts/verify_longctx.py               long contexts do not 400 through litellm
patches/gbnf_pattern_fix.py             output clamp; ordered ceiling table
results/                                raw sweep and NIAH JSON
docs/JOURNEY.md                         what failed, in order, and why
```

## License

Apache-2.0. See `NOTICE` for upstream attribution.
