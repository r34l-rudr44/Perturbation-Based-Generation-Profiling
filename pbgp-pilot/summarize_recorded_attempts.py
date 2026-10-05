"""Export only aggregate recorded-probe metadata; keep archived text local."""
import json
from collections import Counter
from pathlib import Path


def main():
    root = Path("pbgp-pilot/recorded_pilot")
    runs = []
    completed = []
    valid_profile_count = 0
    for folder in [root] + sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        path = folder / "responses.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        runs.append({"name": folder.name, "recorded_requests": len(records),
                     "response_statuses": dict(Counter(str(r.get("status", "network_error")) for r in records)),
                     "successful_recorded_responses": sum(r.get("status") == 200 for r in records),
                     "prompt_character_counts": [len(r["request"]["prompt"]) for r in records]})
        profiles_path = folder / "profiles.json"
        if profiles_path.exists():
            profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
            valid_profile_count += sum(len(candidate["profiles"]) for case in profiles
                                       for candidate in case["candidates"].values())
        summary_path = folder / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for case in summary.get("case_results", []):
                if not case["errors"]:
                    completed.append({"run": folder.name, "base_url": summary.get("base_url"),
                                      "max_prior_turns": summary.get("max_prior_turns", 3),
                                      "max_prior_field_chars": summary.get("max_prior_field_chars", 0), **case})
    report = {"authorization": "User replied proceed after explicit question naming the two cases, prior context, and UncloseAI Hermes endpoint.",
              "model": "turboderp/Qwen3.8-27B-exl3",
              "valid_recorded_profiles": valid_profile_count,
              "completed_case_results": completed,
              "result": "See completed_case_results for usable scores; failed attempts include timeouts, HTTP 500 EngineCore failure, HTTP 502, HTTP 429, and connection resets.",
              "runs": runs,
              "request_count_caution": "Counts are saved request/response pairs. The interrupted initial process may have had an additional in-flight request without a saved response entry.",
              "synthetic_health_check": "A small synthetic echo succeeded between attempts; a subsequent bounded recorded request again returned HTTP 500, and the runner stopped further requests.",
              "alternate_endpoint": {"base_url": "https://qwen.ai.unturf.com/v1",
                                     "synthetic_short_echo_status": 200,
                                     "synthetic_6090_character_echo_status": 200,
                                     "recorded_transfer_status": "User explicitly approved these two cases at the Qwen endpoint; the approved probe was attempted."},
              "local_rules": json.loads(Path("pbgp-pilot/recorded_rule_checks.json").read_text(encoding="utf-8")),
              "limitations": ["Profile changes alone do not establish PBGP detection; no calibrated threshold or benign population.",
                              "Rules were derived from the inspected tool feedback and are a retrospective consistency check, not held-out detection accuracy.",
                              "Counterfactual repairs were scored as text only; no archived commands were executed.",
                              "Archived prompts and raw recorded responses stay in the ignored local directory."]}
    Path("pbgp-pilot/recorded_attempt_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"recorded_requests": sum(r["recorded_requests"] for r in runs),
                      "valid_recorded_profiles": valid_profile_count,
                      "completed_cases": len(completed), "runs": [(r["name"], r["response_statuses"]) for r in runs]}))


if __name__ == "__main__":
    main()
