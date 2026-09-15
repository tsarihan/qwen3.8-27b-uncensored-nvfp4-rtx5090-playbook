#!/usr/bin/env bash
# Serve orcarouter/Qwen3.8-27B-Uncensored-NVFP4 on a single RTX 5090 (SM120, 32 GB).
#
# Every knob is an env var so a config can be changed without editing the file.
# Defaults are the measured-good configuration described in README.md.
#
# DELIBERATE DEVIATIONS from the publisher's recommended serve block, all of them
# forced by a 32 GB card or by the vLLM build available here. They are listed in
# README.md under "Deviations from the card" and they matter if you are quoting
# these numbers as the checkpoint's numbers rather than this rig's:
#
#   * --quantization compressed-tensors and --kv-cache-dtype fp8 are passed. The card
#     says do NOT pass either, because config.json already carries the mixed-precision
#     quantization_config INCLUDING an fp8 kv_cache_scheme with calibrated static
#     scales. Passing --kv-cache-dtype fp8 may substitute dynamic scales for those
#     calibrated ones. Set KV_DTYPE=auto to follow the card instead.
#   * --tool-call-parser defaults to qwen3_xml here; the card specifies qwen3_coder.
#     qwen3_xml was verified to emit structured tool_calls on BOTH template branches
#     (thinking on and thinking off). Set TOOL_PARSER=qwen3_coder to follow the card.
#   * no --speculative-config. The checkpoint preserves an MTP draft head
#     (model-extra-00001-of-00001.safetensors, 0.85 GB) and the card enables it with
#     num_speculative_tokens 2. It is off here so the capability baseline has one
#     fewer variable. Set SPEC to a JSON string to turn it on.
#   * MAX_MODEL_LEN 131072, not the card's 262144. See README: 163840 does not fit.
#   * MAX_NUM_SEQS 16, not the card's 96. A 32 GB card leaves ~4.6 GiB for KV.
#
# Requires a Blackwell GPU for native FP4 tensor cores. The card asks for vLLM >= 0.27;
# v0.26.0 is what is pinned here and it loads and serves correctly, but it is below the
# stated minimum and that is a known deviation, not a recommendation.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-27B-Uncensored-NVFP4}"
MODELS_HOST="${MODELS_HOST:-$HOME/models}"
NAME="${NAME:-vllm-qwen3.8-27b-orca}"
PORT="${PORT:-8138}"
SERVED="${SERVED:-qwen3.8:27b-orca}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.26.0}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
GPU_UTIL="${GPU_UTIL:-0.945}"
KV_DTYPE="${KV_DTYPE:-fp8}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
CACHE_DIR="${CACHE_DIR:-$HOME/router/q38cache}"

mkdir -p "$CACHE_DIR/flashinfer" "$CACHE_DIR/vllm"

SPEC_ARGS=()
if [ -n "${SPEC:-}" ]; then SPEC_ARGS=(--speculative-config "$SPEC"); fi

# VLLM_ALLOW_LONG_MAX_MODEL_LEN is set because this checkpoint reports a 262144 native
# window while we deliberately serve a shorter one; without it vLLM refuses some
# combinations of max-model-len and rope config.
exec docker run -d --name "$NAME" --gpus all --ipc=host --network host --shm-size=16g \
  -e MAX_JOBS=4 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v "$MODELS_HOST":/models \
  -v "$CACHE_DIR/flashinfer":/root/.cache/flashinfer \
  -v "$CACHE_DIR/vllm":/root/.cache/vllm \
  "$IMAGE" \
  --model "$MODEL_DIR" \
  --host 0.0.0.0 --port "$PORT" --served-model-name "$SERVED" \
  --quantization compressed-tensors \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-dtype "$KV_DTYPE" \
  --enable-prefix-caching \
  --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
  --reasoning-parser qwen3 \
  "${SPEC_ARGS[@]}"
