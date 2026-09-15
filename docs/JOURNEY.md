# Journey: what failed, in order

Kept because the failures are more reusable than the final config. Several of these are
measurement mistakes rather than engine problems, and those were the expensive ones.

## 0. Picking which abliteration to run

Several quantizations of this abliteration exist. The one chosen is not the one with the
best-looking card — it is the one where **the publisher of the modified weights did its own
quantization**. Every alternative is a third-party requant: a second, independent
reprocessing of every weight, by a party that did not produce the original edit.

Quantization cannot be audited after the fact, and benchmarks do not catch
trigger-conditioned behavior. For an already-modified checkpoint, minimizing the number of
parties who touched the tensors is the only control that actually does anything.

Independent benchmarking (Sapwell, 9 variants, ~167 GPU-hours, with tensor forensics
checking card claims against actual weights) also ranked this build best of the
abliterations — but that was a confirmation, not the reason.

sha256 of every LFS file was verified against the publisher's hashes before serving:
`VERIFY_OK, 7/7`.

## 1. The download stalled for three hours and looked like progress

`hf download` sat at 21 GB / 17 files for roughly three hours with no error and no
throughput. Killing and resuming it immediately ran at 93 MB/s.

A stalled transfer with a live connection is indistinguishable from a slow one if you only
watch the file count. Watch bytes/second, not completion.

## 2. 163,840 does not fit, and the error tells you what does

```
ValueError: ... max seq len (163840), 5.31 GiB KV needed > 4.62 GiB available ...
estimated maximum model length is 141120
```

A 24.7 GB checkpoint at `--gpu-memory-utilization 0.945` on a 32 GB card leaves ~4.6 GiB
for KV. vLLM's "estimated maximum model length" is the useful part of that message; 131,072
is the largest round window under it. The card's recommended 262,144 is unreachable here at
any utilization.

## 3. A ceiling table keyed by substring gave the model the wrong window

An output-clamp table matched model names by substring, first hit wins:

```python
("qwen3.8:27b", 163_840)     # base lane
```

`"qwen3.8:27b"` is a substring of `"qwen3.8:27b-orca"`. The new lane silently inherited the
base lane's 163,840 ceiling against its real 131,072 window, so the clamp would permit
`prompt + output` up to 155,648 and let vLLM 400 rather than trimming.

Fixed by inserting an explicit row **above** the bare key. Ordering in a substring-matched
table is load-bearing and deserves a comment saying so, because the next person to
alphabetize it will break it.

## 4. `BASE=~/swe-enterprise40` pointed at an empty directory

A reorganization had moved the harness tree and left the old path behind as an empty
directory. The runner still had the old path hardcoded, so `wc -l < "$DS"` gave an empty
`TOTAL` and `set -u` killed the script with an error that pointed nowhere near the cause.

Made overridable: `BASE="${BASE:-$HOME/swepro/enterprise40}"`.

## 5. `pgrep -f` matched the shell running it, twice

```bash
while pgrep -f "sweep.py --base-url http://127.0.0.1:8138" >/dev/null; do sleep 20; done
```

never exits: the watcher's own command line contains the pattern. This produced a "still
running" reading on a sweep that had already finished, and later a "STILL ALIVE" reading on
a harness that had already stopped — which nearly led to killing the wrong thing.

The same mistake in a `kill` pipeline sends the signal to your own shell. It did, once,
which is what an unexplained SSH exit-255 turned out to be.

Use `pgrep -f "sweep[.]py"` (bracket trick) or capture the PID at launch and poll
`kill -0 "$PID"`.

## 6. litellm injected `max_tokens` after the clamp hook and ate a quarter of the window

The expensive one. An agent instance died with `98,305 input + 32,768 output = 131,073`
against a 131,072 window — one token over.

The client sends **no** `max_tokens`. The pre-call hook that exists to clamp output
therefore finds nothing to clamp and returns early. litellm *then* injects the lane's
configured `max_tokens`. The hook never sees the value it is supposed to bound, so fixing
the hook — which had just been fixed, for the unrelated bug in §3 — does nothing at all
here.

Net effect: `max_tokens: 32768` capped usable input at `131072 - 32768 = 98304`. A quarter
of the context window was unreachable, and it only manifests once a context crosses that
line, hours into a run.

Fixed by lowering the *configured* value to 8192 (measured mean generation for an agent
step is ~300 tokens, including the reasoning trace), lifting usable input to ~122,880.
Verified by replaying the exact failure at 98,559 / 115,059 / 122,059 tokens.

The benchmark run was restarted from scratch rather than continued, so both arms of the
comparison share identical config. The partial output was moved aside, not deleted.

## 7. A 180-second measurement produced a confident, wrong conclusion

After dropping from 3 workers to 1, a 180-second sample showed ~520 requests/hour against
the 3-worker run's ~387 — apparently proving that 1 worker was 34% *faster* despite a third
of the concurrency, and there was a tidy story to explain it (prefix retention eliminating
re-prefill).

Over a 30-minute window it was **275 req/hr**. Three workers is ~40% faster, not slower.

The 180-second window covered only the first instance's opening steps: short contexts,
quick shell commands. Agent throughput falls as contexts grow and test commands get slow.
The tidy explanation was wrong too — it was reasoning backwards from a number that was
itself an artifact.

What survived the longer measurement: preemptions really do go to **exactly 0** at 1 worker
versus 47/hr at 3. That is a real effect and a good reason to use 1 worker for a clean A/B.
It is not a throughput win.

`requests/hour` also charges tool-execution time to the model. Prefer
`vllm:e2e_request_latency_seconds` and `vllm:time_to_first_token_seconds` histogram deltas.

## 8. A metric that looked alarming and meant nothing

Two thirds of responses showed `finished_reason="length"` early in an agent run, which is
the exact condition that produces unparseable tool calls.

It was the concurrency sweep. That harness issues exactly 127 requests
(1+2+4+8+16+32+64), every one capped at `max_tokens 512`, so every one finishes as
`length`. vLLM's counters are cumulative since container start and do not separate your
benchmark from your workload.

Separately, `prefix_cache_queries_total` advanced 75 million in 90 seconds on this build —
physically impossible at ~4,000 tok/s prefill. Whatever it counts here, it is not
comparable to the same counter on another build, and the hit-rate derived from it was
discarded. `kv_cache_usage_perc`, `num_preemptions_total` and the ITL histogram were
trustworthy throughout.

## 9. Known gaps

* **SWE-bench Pro scores are not in this repository yet.** Runs are in progress in two
  arms, thinking on at `xhigh` and thinking off, at 1 worker.
* **No non-abliterated control has been run.** Until `Qwen3.8-27B-NVFP4` is measured on
  this same card with the same flags, nothing here says what abliteration cost.
* **KV cache scales are a known deviation.** The card says not to pass `--kv-cache-dtype`,
  because `config.json` carries calibrated static fp8 scales; this lane passes it anyway.
  Held constant across all measurements so it cancels out internally, but it should be
  re-measured with `KV_DTYPE=auto` before these are treated as the checkpoint's numbers.
* **vLLM v0.26.0 is below the card's stated `>= 0.27` minimum.** It loads and serves
  correctly and retrieval is clean, but it is not what the publisher validated.
* **The MTP draft head is preserved in the checkpoint and unused here.**
