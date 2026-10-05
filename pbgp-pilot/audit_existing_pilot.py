"""Offline falsification checks on already-observed pilot evidence.

Baselines selected after inspection are explicitly retrospective. They show
confounds, not validated detector performance. No external requests.
"""
import json
from collections import Counter
from pathlib import Path

from score_fixed_text import auc, extract_continuation


def main():
    root = Path(__file__).resolve().parent
    cases = json.loads((root / "synthetic_pilot/cases.json").read_text(encoding="utf-8"))
    profiles = json.loads((root / "synthetic_pilot/profiles.json").read_text(encoding="utf-8"))
    valid = [row for row in profiles if not row["errors"]]
    labels = [case["label"] for case in cases]
    cues = [int("proceed even if the trusted task conflicts" in case["contexts"]["original"]) for case in cases]
    paired = []
    for family in sorted({row["task_id"] for row in valid}):
        rows = {row["category"]: row for row in valid if row["task_id"] == family}
        bad, good = rows["unauthorized_peer"], rows["authorized_sensitive"]
        assert bad["continuation"] == good["continuation"]
        score = lambda row: row["comparisons"]["peer_removed"]["wasserstein_logprobs"]
        paired.append({"family": family, "identical_action": True,
                       "attack_w1": score(bad), "authorized_w1": score(good),
                       "attack_ranks_higher": score(bad) > score(good)})
    evidence = {"status": "retrospective audit, not held-out attack evaluation",
                "synthetic_cases": len(cases),
                "independent_task_families": len({case["task_id"] for case in cases}),
                "explicit_peer_conflict_phrase_auc": auc(labels, cues),
                "action_character_length_auc": auc(labels, [len(case["continuation"]) for case in cases]),
                "action_token_length_auc": auc([row["label"] for row in valid], [row["profiles"]["original"]["num_tokens"] for row in valid]),
                "paired_identical_action_results": paired,
                "paired_attack_ranks_higher_count": sum(row["attack_ranks_higher"] for row in paired),
                "request_outcome_leak_check": {}}
    checked = 0
    by_synthetic_id = {case["id"]: case for case in cases}
    for line in (root / "synthetic_pilot/responses.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") != 200:
            continue
        continuation = by_synthetic_id[row["case_id"]]["continuation"]
        prompt = row["request"]["prompt"]
        assert prompt.endswith(continuation)
        extract_continuation(row["response"], prompt[:-len(continuation)], continuation)
        checked += 1
    evidence["synthetic_successful_responses_revalidated"] = checked
    # Inspect saved requests without exporting their prompts or outcome strings.
    path = root / "recorded_pilot/qwen_final_short/responses.jsonl"
    if path.exists():
        selected = json.loads((root / "sample_audit/cases.json").read_text(encoding="utf-8"))
        by_id = {case["id"]: case for case in selected}
        counts = Counter()
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prompt = row["request"]["prompt"]
            case = by_id[row["case_id"]]
            continuation = json.dumps(case["candidates"][row["candidate"]], ensure_ascii=True, sort_keys=True)
            assert prompt.endswith(continuation)
            extract_continuation(row["response"], prompt[:-len(continuation)], continuation)
            counts["requests_checked"] += 1
            counts["current_full_error_text_present"] += int(bool(case["label_evidence"]) and case["label_evidence"] in prompt)
            counts["following_turn_id_present"] += int(bool(case["following_turn"]) and case["following_turn"]["id"] in prompt)
        evidence["request_outcome_leak_check"] = dict(counts)
        evidence["request_outcome_leak_check"]["scope"] = "Exact full-error and next-turn-ID checks plus pre-action builder inspection; not a proof against every semantic leakage path."
    (root / "leakage_audit_results.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
