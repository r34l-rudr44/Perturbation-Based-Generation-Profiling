"""Replay-score selected actions without running them or leaking their outcomes.

Counterfactual repairs use exactly the same pre-action reconstructed context.
There is no benign population here: do not report AUROC or calibrated FPR.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from score_fixed_text import BASE_URL, api_request, compare_profiles, extract_continuation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-external-recorded-data", action="store_true",
                        help="Confirm explicit authorization to send archived case contexts to UncloseAI")
    parser.add_argument("--output", default="pbgp-pilot/recorded_pilot")
    parser.add_argument("--max-prior-field-chars", type=int, default=0,
                        help="Bound each prior field with explicit head/tail excerpts; zero preserves full history")
    parser.add_argument("--case-id", help="Run only one of the already-selected case IDs")
    parser.add_argument("--max-prior-turns", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--base-url", default=BASE_URL,
                        help="Approved service endpoint; changing provider requires separate transfer authorization")
    args = parser.parse_args()
    if not args.allow_external_recorded_data:
        raise SystemExit("Recorded contexts stay local. Explicit transfer authorization is required before using --allow-external-recorded-data.")
    if args.max_prior_field_chars and args.max_prior_field_chars < 200:
        raise SystemExit("Excerpt budget must be at least 200 characters")
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    raw_path = out / "responses.jsonl"
    if raw_path.exists():
        raise SystemExit("Existing run evidence; choose another output before rerunning")
    def request(path, body=None):
        return api_request(path, body, base_url=args.base_url)

    catalog = request("/models")
    (out / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    models = catalog.get("response", {}).get("data", [])
    if catalog.get("status") != 200 or len(models) != 1:
        raise SystemExit("Expected one live model")
    model = models[0]["id"]
    health = request("/completions", {"model": model, "prompt": "The sky is blue.",
                                         "max_tokens": 0, "echo": True, "logprobs": 5})
    (out / "health_check.json").write_text(json.dumps(health, indent=2), encoding="utf-8")
    if health.get("status") != 200 or not health.get("response", {}).get("choices"):
        raise SystemExit("Synthetic health check failed; no recorded contexts sent in this run")
    cases = json.loads(Path("pbgp-pilot/sample_audit/cases.json").read_text(encoding="utf-8"))
    if args.case_id:
        cases = [c for c in cases if c["id"] == args.case_id]
        if not cases:
            raise SystemExit("Case ID is not among the approved selected cases")
    if args.max_prior_field_chars or args.max_prior_turns < 3:
        # Work only with the already-approved prior interactions. No current
        # outcomes, following turns, or labels enter these reconstructed prompts.
        for case in cases:
            original = case["contexts"]["original"]
            prior = json.loads(original.split("\n", 1)[1].rsplit("\n\nNext tool call:\n", 1)[0])
            prior = prior[-args.max_prior_turns:]
            for turn in prior:
                for field in ("action", "output", "error"):
                    value = turn[field]
                    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True, sort_keys=True)
                    if args.max_prior_field_chars and len(text) > args.max_prior_field_chars:
                        head = args.max_prior_field_chars * 3 // 4
                        turn[field] = {"excerpted": True, "original_characters": len(text),
                                       "head": text[:head], "tail": text[-(args.max_prior_field_chars-head):]}
            prefix = "Prior tool interactions (reconstructed; long fields explicitly excerpted):\n"
            removed = [dict(p, output=None, error=None) if i == len(prior)-1 else p for i, p in enumerate(prior)]
            case["contexts"] = {
                "original": prefix + json.dumps(prior, ensure_ascii=True, sort_keys=True),
                "latest_feedback_removed": prefix + json.dumps(removed, ensure_ascii=True, sort_keys=True),
                "format_rephrased": "Earlier tool actions and observations (reconstructed; explicit excerpts):\n" + json.dumps(prior, ensure_ascii=True, sort_keys=True, indent=2),
            }
            case["contexts"] = {k: v + "\n\nNext tool call:\n" for k, v in case["contexts"].items()}
    results = []
    circuit_error = None
    for case in cases:
        entry = {"id": case["id"], "kind": case["kind"], "candidates": {}}
        for name, candidate in case["candidates"].items():
            continuation = json.dumps(candidate, ensure_ascii=True, sort_keys=True)
            profiles, errors = {}, []
            contexts = dict(case["contexts"], original_repeat=case["contexts"]["original"])
            for variant, prefix in contexts.items():
                if circuit_error:
                    errors.append({"variant": variant, "error": "not attempted after upstream failure: " + circuit_error})
                    continue
                body = {"model": model, "prompt": prefix + continuation, "max_tokens": 0,
                        "echo": True, "logprobs": 5, "temperature": 0}
                time.sleep(1)  # At most one new request/second; shared free service.
                response = request("/completions", body)
                with raw_path.open("a", encoding="utf-8") as raw:
                    raw.write(json.dumps({"case_id": case["id"], "candidate": name,
                                          "variant": variant, "request": body, **response}) + "\n")
                if response.get("status") in (429, 500, 502, 503, 504):
                    circuit_error = str(response.get("error") or response["status"])
                try:
                    if response.get("status") != 200:
                        raise ValueError(str(response.get("error") or response.get("status")))
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
               "base_url": args.base_url,
               "max_prior_field_chars": args.max_prior_field_chars,
               "max_prior_turns": args.max_prior_turns,
               "upstream_circuit_error": circuit_error,
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
