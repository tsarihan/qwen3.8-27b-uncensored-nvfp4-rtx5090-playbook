# Harness

Run the load generator from a **separate host** where you can. On this rig the client and
the engine share one machine, which is tolerable for a single 32 GB card but does charge
some client CPU against the same box.

| file | what it does |
|---|---|
| `serve-qwen38-27b-orca-vllm.sh` | starts the single-card lane; every knob is an env var (`MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `GPU_UTIL`, `KV_DTYPE`, `TOOL_PARSER`, `SPEC`, …) with the card-following alternatives documented in the header |
| `verify-nvfp4.py` | sha256 of every LFS file against the publisher's hashes. `HF_REPO` / `LOCAL_DIR` override the defaults. Run before serving |
| `sweep.py` | concurrency ladder; reports aggregate tok/s, per-stream decode, TTFT mean/p50/max. One distinct nonce per request so prefix caching cannot let later streams reuse earlier prefill |
| `needles.py` | multi-needle retrieval across a context sweep; token counts verified against the server's `/tokenize`, distinct nonce per run |
| `toolcall_nothink.py` | structured tool calls on **both** template branches, thinking on and off |
| `toggle_verify.py` | that `enable_thinking` actually survives your proxy hop |
| `verify_longctx.py` | that long prompts with no client `max_tokens` do not 400 through litellm |

## Run the three probes before any long agent run

`toolcall_nothink.py`, `toggle_verify.py` and `verify_longctx.py` take well under a minute
between them and each catches a failure that otherwise surfaces hours in:

* a model that cannot emit a parseable tool call does not fail fast — it burns its whole
  step budget on format-error retries, per instance
* a thinking toggle that is silently dropped by the proxy invalidates the entire arm of a
  comparison without producing a single error
* a proxy-injected `max_tokens` caps usable context at `window − max_tokens`, and only
  bites once a context grows past that line

## Reading the ladder

`sweep.py` uses ~133-token prompts, so KV never binds and it reports clean linear scaling up
to `--max-num-seqs`. **That does not transfer to multi-turn agent load**, where contexts grow
into the tens of thousands of tokens and evict each other from the cache. Size real
workloads against `kv_cache_size_tokens` versus turn-grown context, and watch
`vllm:num_preemptions_total`.

Watch for bimodal TTFT: when p50 is low but max is an order of magnitude higher, you are
past `--max-num-seqs` and the excess is queueing, not running. Mean TTFT hides this.

## Two counter traps

vLLM's Prometheus counters are **cumulative since container start** and do not separate your
benchmark from your workload. `sweep.py` alone contributes 127 requests that all finish with
`finished_reason="length"` because they are capped at `max_tokens 512` — which looks exactly
like an agent run truncating before it can emit a tool call. Take deltas over a window, not
totals.

On `vllm/vllm-openai:v0.26.0`, `prefix_cache_queries_total` advanced 75 million in 90
seconds, which is impossible at the measured prefill rate. Whatever it counts on this build,
hit rates derived from it are not comparable across builds. `kv_cache_usage_perc`,
`num_preemptions_total`, and the inter-token-latency histogram were reliable.

`1 / mean(inter_token_latency)` is the honest per-stream decode rate. Dividing
`generation_tokens_total` by `num_requests_running` is not — it counts prefilling streams as
producing zero tokens, which badly understates decode in a prefill-heavy agent workload.
