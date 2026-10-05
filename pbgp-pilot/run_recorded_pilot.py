"""Replay-score selected actions without running them or leaking their outcomes.

Counterfactual repairs use exactly the same pre-action reconstructed context.
There is no benign population here: do not report AUROC or calibrated FPR.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from score_fixed_text import api_request, compare_profiles, extract_continuation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-external-recorded-data", action="store_true",
                        help="Confirm explicit authorization to send archived case contexts to UncloseAI")
    args = parser.parse_args()
    if not args.allow_external_recorded_data:
        raise SystemExit("Recorded contexts stay local. Explicit transfer authorization is required before using --allow-external-recorded-data.")
    out = Path("pbgp-pilot/recorded_pilot")
    out.mkdir(exist_ok=True)
    raw_path = out / "responses.jsonl"
    if raw_path.exists():
        raise SystemExit("Existing run evidence; choose another output before rerunning")
    catalog = api_request("/models")
    (out / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    models = catalog.get("response", {}).get("data", [])
    if catalog.get("status") != 200 or len(models) != 1:
        raise SystemExit("Expected one live model")
    model = models[0]["id"]
    cases = json.loads(Path("pbgp-pilot/sample_audit/cases.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        entry = {"id": case["id"], "kind": case["kind"], "candidates": {}}
        for name, candidate in case["candidates"].items():
            continuation = json.dumps(candidate, ensure_ascii=True, sort_keys=True)
            profiles, errors = {}, []
            contexts = dict(case["contexts"], original_repeat=case["contexts"]["original"])
            for variant, prefix in contexts.items():
                body = {"model": model, "prompt": prefix + continuation, "max_tokens": 0,
                        "echo": True, "logprobs": 5, "temperature": 0}
                response = api_request("/completions", body)
                with raw_path.open("a", encoding="utf-8") as raw:
                    raw.write(json.dumps({"case_id": case["id"], "candidate": name,
                                          "variant": variant, "request": body, **response}) + "\n")
                try:
                    if response.get("status") != 200:
                        raise ValueError(str(response.get("error", response.get("status"))))
                    profiles[variant] = extract_continuation(response["response"], prefix, continuation)
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    errors.append({"variant": variant, "error": str(exc)})
            result = {"continuation": continuation, "profiles": profiles, "errors": errors}
            if not errors:
                try:
                    result["comparisons"] = {v: compare_profiles(profiles["original"], profiles[v])
                                             for v in ("latest_feedback_removed", "format_rephrased", "original_repeat")}
                except ValueError as exc:
                    errors.append({"error": str(exc)})
            entry["candidates"][name] = result
            print(json.dumps({"case": case["kind"], "candidate": name, "errors": errors,
                              "comparisons": result.get("comparisons")}), flush=True)
        results.append(entry)
        (out / "profiles.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "model": model,
               "observed_error_cases": len(cases), "case_results": [],
               "interpretation": "Two selected tool errors. Reconstructed context and counterfactual repairs; no original-model or swarm-prevention claim."}
    for entry in results:
        observed = entry["candidates"]["observed_error"]
        repair = entry["candidates"]["counterfactual_repair"]
        item = {"id": entry["id"], "kind": entry["kind"],
                "errors": observed["errors"] + repair["errors"]}
        if not item["errors"]:
            item.update({"observed_mean_nll": observed["profiles"]["original"]["mean_nll"],
                         "repair_mean_nll": repair["profiles"]["original"]["mean_nll"],
                         "observed_feedback_removal_wasserstein": observed["comparisons"]["latest_feedback_removed"]["wasserstein_logprobs"],
                         "repair_feedback_removal_wasserstein": repair["comparisons"]["latest_feedback_removed"]["wasserstein_logprobs"],
                         "observed_repeat_wasserstein": observed["comparisons"]["original_repeat"]["wasserstein_logprobs"]})
        summary["case_results"].append(item)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
