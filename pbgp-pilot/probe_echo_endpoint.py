"""Probe a user-selected public endpoint with synthetic text only."""
import argparse
import json
from pathlib import Path

from score_fixed_text import api_request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit("Preserve previous evidence; choose a fresh output file")
    catalog = api_request("/models", base_url=args.base_url)
    result = {"base_url": args.base_url, "catalog": catalog, "probe_text": "The sky is blue."}
    models = catalog.get("response", {}).get("data", [])
    if catalog.get("status") == 200 and len(models) == 1:
        body = {"model": models[0]["id"], "prompt": result["probe_text"], "echo": True,
                "max_tokens": 0, "logprobs": 5}
        result["request"] = body
        result["echo"] = api_request("/completions", body, base_url=args.base_url)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"catalog_status": catalog.get("status"), "models": [m["id"] for m in models],
                      "echo_status": result.get("echo", {}).get("status"),
                      "error": result.get("echo", {}).get("error", catalog.get("error"))}))


if __name__ == "__main__":
    main()
