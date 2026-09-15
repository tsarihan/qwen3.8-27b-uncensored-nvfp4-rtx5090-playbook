"""Verify tool calling still works on the NON-THINKING template branch.

The earlier tool-call check ran with thinking on (the default). The no-think branch
takes a different path through chat_template.jinja (it pre-fills an empty <think></think>
before the model's output), so qwen3_xml parsing has not actually been exercised there.
If structured tool_calls do not come back, the thinking-off SWE Pro run is worthless
before it starts -- every instance would die with RepeatedFormatError.
"""
import json, urllib.request, urllib.error
import os
MODEL = os.environ.get("MODEL", "qwen3.8:27b-orca")
API_KEY = os.environ.get("OPENAI_API_KEY", "local")

URL = os.environ.get("PROXY_URL", "http://127.0.0.1:4000/v1") + "/chat/completions"
TOOLS = [{"type": "function", "function": {
    "name": "bash",
    "description": "Run a bash command",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]


def ask(label, ctk):
    body = {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": "List the files in /testbed. Use the bash tool."}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "drop_params": True,
        "max_tokens": 512,
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
        print(f"{label:26s} HTTP {e.code}: {e.read()[:200].decode(errors='replace')}")
        return
    ch = d["choices"][0]
    m = ch["message"]
    tc = m.get("tool_calls")
    rc = m.get("reasoning_content") or m.get("reasoning") or ""
    print(f"{label:26s} finish={ch.get('finish_reason'):12s} "
          f"reasoning={len(rc):4d}ch  tool_calls={'YES' if tc else 'NO '}")
    if tc:
        print(f"{'':26s}   -> {json.dumps(tc[0]['function'])[:120]}")
    else:
        print(f"{'':26s}   content={(m.get('content') or '')[:160]!r}")
    print(f"{'':26s}   VERDICT: {'OK' if tc else 'BROKEN -- would RepeatedFormatError'}")


ask("thinking ON (xhigh)", {"enable_thinking": True, "reasoning_effort": "xhigh"})
ask("thinking OFF", {"enable_thinking": False})
