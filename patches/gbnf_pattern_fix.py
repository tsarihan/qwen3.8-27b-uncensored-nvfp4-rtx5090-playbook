"""litellm pre-call hook: strip JSON-schema regex patterns that llama.cpp cannot
turn into a valid GBNF grammar.

WHY
llama.cpp compiles tool schemas into a GBNF grammar to constrain tool calls. Its
json-schema-to-grammar pass copies regex escapes through verbatim, but GBNF
accepts only a fixed escape set (\\n \\r \\t \\\\ \\" \\xNN \\uNNNN). A pattern
containing \\- or \\/ -- both legal in a regex -- emits an escape GBNF cannot
parse, so a grammar is produced and then dies at parse time:

    parse: error parsing grammar: unknown escape at \\-.~:@+]

Captured live from Claude Code's `Artifact` tool, whose collection pattern is
    ^(?!\\.\\.?(?:\\/|$))[A-Za-z0-9_\\-.~:@+]{1,200}(?:\\/...){0,14}$
It surfaces during CC's automatic compaction, because that is the call which
forces constrained tool output. The same log also shows llama.cpp warning
"JSON schema conversion was incomplete: Unsupported pattern syntax" for the
three lookaheads, which it silently drops.

WHY STRIP RATHER THAN REWRITE
The obvious fix -- rewrite \\- to - and \\/ to / -- is WRONG inside a character
class. `[A-Za-z0-9_\\-.~:@+]` becomes `[A-Za-z0-9_-.~:@+]`, where `_-.` is now
read as a RANGE from `_` (0x5F) to `.` (0x2E). That range is reversed, so the
class is invalid: a GBNF parse error would be traded for a broken regex.
Relocating the hyphen safely per-class is fiddly and easy to get wrong.

`pattern` is only a refinement on a string that is already typed as a string.
Dropping it yields a valid, slightly looser grammar -- the tool call still has
to be well-formed JSON with the right keys and types. That is the conservative
trade, and llama.cpp was already discarding parts of these patterns anyway.

Only patterns that are actually unsafe are removed; well-behaved ones are kept
so constraints are not lost needlessly. vLLM/xgrammar lanes are unaffected in
practice, since their patterns compile either way.
"""
import json

from litellm.integrations.custom_logger import CustomLogger

# Escapes that are valid in a regex but not in GBNF.
_UNSAFE = ("\\-", "\\/")


def _is_unsafe(pat: str) -> bool:
    return any(tok in pat for tok in _UNSAFE)


def sanitize(node, stats):
    """Drop `pattern` keys whose regex would emit invalid GBNF."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "pattern" and isinstance(v, str) and _is_unsafe(v):
                stats["dropped"] += 1
                continue
            out[k] = sanitize(v, stats)
        return out
    if isinstance(node, list):
        return [sanitize(v, stats) for v in node]
    return node


def _has_unsafe(obj) -> bool:
    try:
        blob = json.dumps(obj)
    except Exception:
        return False
    # json.dumps renders a literal backslash as two, so match on that form.
    return any(tok.replace("\\", "\\\\") in blob for tok in _UNSAFE)


# Engine hard ceilings (total window = input + output), by served model.
# litellm's max_input_tokens makes the CLIENT compact in time; this clamp is the
# backstop for when it still arrives one token over, which is not hypothetical:
#   requested 32000 output + 968001 input = 1000001 vs the old 1000000 ceiling -> 400.
# Engine hard ceilings (TOTAL window = input + output), most-specific first --
# "qwen3.8:27b" is a prefix of "qwen3.8:27b-fast1", so order matters.
#
# litellm ENFORCES max_input_tokens and raises ContextWindowExceededError itself,
# so declaring less than the real window creates a rejection point BELOW the
# engine's capability -- and the request it rejects is the compaction call, the
# largest of the session. That is the dead end we are fixing, not avoiding.
# So each lane declares its true window, and this clamp keeps input+output
# inside it by trimming the OUTPUT budget instead of failing the request.
_CEILINGS = [
    ("qwen3.8:27b-fast1", 524_288),     # llama.cpp -c 524288 / P=1
    ("qwen3.8:27b-fast2", 262_144),     # -c 524288 / P=2
    ("qwen3.8:27b-fast4", 122_880),     # -c 491520 / P=4
    ("qwen3.8:27b-fast8",  49_152),     # -c 393216 / P=8
    ("qwen3.8:27b-orca",    131_072),   # vLLM --max-model-len. MUST precede the
                                        # bare "qwen3.8:27b" row below: that key is a
                                        # substring of this model name and would hand
                                        # orca the 163840 base-lane ceiling.
    ("qwen3.8:27b-nightly", 163_840),
    ("qwen3.8:27b",        163_840),    # vLLM --max-model-len
    ("qwen3.8-flash-next", 1_048_576),  # 2^20: YaRN factor 4.0 x 262144
    ("qwen-3.8-flash-next", 1_048_576),
]

# Reserve below the ceiling. Absorbs tokenizer-vs-estimate drift and the few
# tokens a chat template adds around the messages. 920577 + 128000 = 1048577
# failed by ONE token against a 1048576 window; 256 was not enough slack.
_MARGIN = 8192
# Fallback only, when the real tokenizer is unavailable. 3 over-estimates on
# purpose (English is ~4 chars/token, JSON/code nearer 3): trimming a little
# extra output is harmless, under-trimming 400s the request.
_CHARS_PER_TOKEN = 3
# Never clamp below this -- a request with no room left should fail loudly at
# the engine rather than silently return a truncated stub.
_MIN_OUTPUT = 512


def _count_input(data):
    """Real token count where possible, conservative char estimate otherwise."""
    msgs = data.get("messages") or []
    try:
        import litellm
        return int(litellm.token_counter(model=str(data.get("model") or ""), messages=msgs))
    except Exception:
        pass
    try:
        return len(json.dumps(msgs)) // _CHARS_PER_TOKEN
    except Exception:
        return 0


def _clamp_output(data):
    """Trim max_tokens so input+output fits the engine window, instead of 400ing.

    litellm enforces max_input_tokens itself, so each lane declares its TRUE
    window (anything lower rejects the compaction request outright -- the exact
    dead end this is here to prevent). Keeping the total inside that window is
    then this clamp's job, and it gives up OUTPUT budget rather than the call.
    """
    model = str(data.get("model") or "")
    ceiling = next((v for k, v in _CEILINGS if k in model), None)
    if not ceiling:
        return
    want = data.get("max_tokens")
    if not isinstance(want, int) or want <= 0:
        return
    est_in = _count_input(data)
    if not est_in:
        return
    room = ceiling - est_in - _MARGIN
    if room < _MIN_OUTPUT:
        room = _MIN_OUTPUT          # input itself is at/over the window
    if want > room:
        data["max_tokens"] = room
        print(f"[gbnf-pattern-fix] clamped max_tokens {want} -> {room} "
              f"(input {est_in} + margin {_MARGIN} vs ceiling {ceiling})", flush=True)


class GBNFPatternFix(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        _clamp_output(data)
        for key in ("tools", "response_format"):
            val = data.get(key)
            if val and _has_unsafe(val):
                stats = {"dropped": 0}
                data[key] = sanitize(val, stats)
                if stats["dropped"]:
                    print(f"[gbnf-pattern-fix] dropped {stats['dropped']} GBNF-unsafe "
                          f"regex pattern(s) from {key}", flush=True)
        return data


handler = GBNFPatternFix()
