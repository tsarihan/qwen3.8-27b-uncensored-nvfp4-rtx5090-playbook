"""Verify the enable_thinking toggle survives the litellm hop to vLLM.

A SWE Pro run where the toggle is silently dropped would be worthless, so this
checks the actual reasoning field rather than assuming extra_body is forwarded.
Expect: thinking-on returns a non-empty reasoning trace, thinking-off returns none.
"""
import json, urllib.request, urllib.error
import os
MODEL = os.environ.get("MODEL", "qwen3.8:27b-orca")
API_KEY = os.environ.get("OPENAI_API_KEY", "local")

URL = os.environ.get("PROXY_URL", "http://127.0.0.1:4000/v1") + "/chat/completions"
Q = "Which is larger, 9.11 or 9.9? Answer in one line."


def ask(label, ctk):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": Q}],
        "max_tokens": 1024,
    }
    if ctk is not None:
        body["chat_template_kwargs"] = ctk
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        print(f"{label:34s} HTTP {e.code}: {e.read()[:180].decode(errors='replace')}")
        return
    m = d["choices"][0]["message"]
    rc = m.get("reasoning_content") or m.get("reasoning") or ""
    print(f"{label:34s} reasoning={len(rc):5d} chars  "
          f"completion_tokens={d['usage']['completion_tokens']:5d}  "
          f"content={ (m.get('content') or '')[:48]!r}")


ask("no kwargs (default)", None)
ask("enable_thinking=true + xhigh", {"enable_thinking": True, "reasoning_effort": "xhigh"})
ask("enable_thinking=false", {"enable_thinking": False})
