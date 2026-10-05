"""Small capability probe for OmniRoute's keyless UncloseAI upstream.

Uses a public identification string, no stored account credentials.
Records actual upstream responses; chat output and echoed prompt scoring are
separate capabilities. Run from the repository root.
"""
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://hermes.ai.unturf.com/v1"
OUT = Path("pbgp-pilot/uncloseai_logprobs_probe.json")
results = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "base_url": BASE, "source_reference": "OmniRoute 23a11484862b3bb589a55e85b00e4ac53ffeb234",
           "tests": []}


def request(path, body=None):
    req = urllib.request.Request(
        BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer pbgp-public-capability-probe",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            return {"status": response.status, "response": json.loads(response.read())}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "error": exc.read().decode(errors="replace")[:3000]}
    except Exception as exc:
        return {"error": str(exc)}


def save(test, body, result):
    results["tests"].append({"test": test, "request": body, **result})
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"test": test, "status": result.get("status"),
                      "error": result.get("error"),
                      "logprobs": [c.get("logprobs") for c in
                                   result.get("response", {}).get("choices", [])]}), flush=True)


catalog = request("/models")
save("models", None, catalog)
for entry in catalog.get("response", {}).get("data", []):
    model = entry["id"]
    chat = {"model": model, "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
            "max_tokens": 8, "temperature": 0, "logprobs": True, "top_logprobs": 5}
    save("chat_top5", chat, request("/chat/completions", chat))
    # Echo with zero generated tokens tests probabilities for the input text.
    echo = {"model": model, "prompt": "The sky is blue.", "max_tokens": 0,
            "temperature": 0, "echo": True, "logprobs": 5}
    save("completion_echo", echo, request("/completions", echo))
