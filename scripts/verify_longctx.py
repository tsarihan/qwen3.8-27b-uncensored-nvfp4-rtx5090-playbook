"""Reproduce the exact failure that killed an instance, and confirm it is fixed.

The failing request had ~98,305 input tokens and NO client-side max_tokens; litellm
injected max_tokens=32768 after the clamp hook, giving 131,073 against a 131,072 window.
This sends prompts at and above that old cliff with no max_tokens, exactly as
mini-swe-agent does. With max_tokens now 8192, usable input should reach 122,880.
"""
import json, urllib.request, urllib.error
import os
MODEL = os.environ.get("MODEL", "qwen3.8:27b-orca")
API_KEY = os.environ.get("OPENAI_API_KEY", "local")

URL = os.environ.get("PROXY_URL", "http://127.0.0.1:4000/v1") + "/chat/completions"
TOK = os.environ.get("VLLM_URL", "http://127.0.0.1:8138") + "/tokenize"


def ntok(text):
    req = urllib.request.Request(
        TOK, data=json.dumps({"model": MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["count"]


UNIT = ("The maintenance log records routine calibration of the sensor array. "
        "Ambient conditions remained nominal throughout the observation window. ")


def filler(target):
    s = UNIT * max(1, target // 12)
    # converge on the target by measuring, not estimating
    for _ in range(8):
        n = ntok(s)
        if abs(n - target) <= target * 0.01:
            break
        s = s[: max(1, int(len(s) * target / max(n, 1)))]
    return s, ntok(s)


for target in (98_500, 115_000, 122_000):
    text, actual = filler(target)
    body = {"model": MODEL,
            "messages": [{"role": "user",
                          "content": text + "\n\nReply with the single word: ok"}]}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
        u = d["usage"]
        print(f"  input~{actual:>7,} tokens -> OK   prompt={u['prompt_tokens']:,} "
              f"completion={u['completion_tokens']}")
    except urllib.error.HTTPError as e:
        msg = e.read()[:220].decode(errors="replace")
        print(f"  input~{actual:>7,} tokens -> HTTP {e.code}: {msg}")
