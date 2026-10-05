"""Retry only synthetic timeout measurements once; preserve initial evidence."""
import json
from datetime import datetime, timezone
from pathlib import Path

from score_fixed_text import api_request, auc, compare_profiles, extract_continuation


def main():
    out = Path("pbgp-pilot/synthetic_pilot")
    if (out / "profiles_initial.json").exists():
        raise SystemExit("One retry pass already attempted; inspect evidence")
    profile_path = out / "profiles.json"
    summary_path = out / "summary.json"
    results = json.loads(profile_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model = summary["model"]
    catalog = api_request("/models")
    if [x["id"] for x in catalog.get("response", {}).get("data", [])] != [model]:
        raise SystemExit("Model catalog changed; no retries")
    cases = {c["id"]: c for c in json.loads((out / "cases.json").read_text(encoding="utf-8"))}
    (out / "profiles_initial.json").write_text(profile_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out / "summary_initial.json").write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    attempts = 0
    for item in results:
        remaining = []
        case = cases[item["id"]]
        for failure in item["errors"]:
            if "timed out" not in failure.get("error", "") or "variant" not in failure:
                remaining.append(failure)
                continue
            variant = failure["variant"]
            prefix = case["contexts"]["original" if variant == "original_repeat" else variant]
            body = {"model": model, "prompt": prefix + case["continuation"],
                    "echo": True, "max_tokens": 0, "logprobs": 5, "temperature": 0}
            response = api_request("/completions", body)
            attempts += 1
            with (out / "responses.jsonl").open("a", encoding="utf-8") as raw:
                raw.write(json.dumps({"case_id": case["id"], "variant": variant,
                                      "retry": 1, "request": body, **response}) + "\n")
            try:
                if response.get("status") != 200:
                    raise ValueError(str(response.get("error", response.get("status"))))
                item["profiles"][variant] = extract_continuation(response["response"], prefix, case["continuation"])
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                remaining.append({"variant": variant, "error": str(exc)})
            print(json.dumps({"case": case["id"], "variant": variant, "status": response.get("status"), "error": response.get("error")}), flush=True)
        item["errors"] = remaining
        if not remaining:
            try:
                item["comparisons"] = {v: compare_profiles(item["profiles"]["original"], item["profiles"][v])
                                       for v in ("peer_removed", "format_rephrased", "original_repeat")}
            except ValueError as exc:
                item["errors"].append({"error": str(exc)})
        profile_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    valid = [r for r in results if not r["errors"]]
    summary.update({"retry_timestamp_utc": datetime.now(timezone.utc).isoformat(), "retry_attempts": attempts,
                    "valid_cases": len(valid), "rejected_cases": len(results)-len(valid),
                    "auroc_peer_removal": auc([r["label"] for r in valid], [r["comparisons"]["peer_removed"]["wasserstein_logprobs"] for r in valid]),
                    "auroc_original_nll_baseline": auc([r["label"] for r in valid], [r["profiles"]["original"]["mean_nll"] for r in valid]),
                    "max_repeat_wasserstein": max((r["comparisons"]["original_repeat"]["wasserstein_logprobs"] for r in valid), default=None)})
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
