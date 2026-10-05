"""Local evaluation primitives; no provider calls, training, or action execution.

These guards are used by the audit/tests. Existing historical classifier
scripts in the pinned submodule are not silently patched or certified.
"""
import hashlib
import json
import math


def scoring_payload(case):
    """Allowlist detector-visible fields; labels cannot choose a context."""
    visible = case["detector_input"]
    if set(visible) != {"contexts", "continuation"}:
        raise ValueError("Unexpected detector input field")
    contexts = visible["contexts"]
    if set(contexts) != {"original", "perturbed"}:
        raise ValueError("Expected one fixed, label-independent perturbation")
    if not all(isinstance(v, str) for v in contexts.values()) or not isinstance(visible["continuation"], str):
        raise ValueError("Detector input must contain text")
    if contexts["original"] == contexts["perturbed"]:
        raise ValueError("No-op perturbation: measurement unavailable")
    return json.loads(json.dumps(visible))


def validate_partitions(partitions):
    """Reject exact duplicates and declared episode/coordination-group overlap.

Callers must assign connected group IDs covering shared tasks, message boards,
and derivatives. Exact hashes do not establish semantic independence.
"""
    if set(partitions) != {"development", "calibration", "test"}:
        raise ValueError("Expected development, calibration and test partitions")
    seen_ids, seen_groups, seen_hashes = {}, {}, {}
    for split, rows in partitions.items():
        if not rows:
            raise ValueError("Empty partition")
        for case in rows:
            case_id, group = case["id"], case["group_id"]
            if not case_id or not group:
                raise ValueError("Missing case or independence group ID")
            if case_id in seen_ids:
                raise ValueError("Repeated case ID")
            if group in seen_groups and seen_groups[group] != split:
                raise ValueError("Related group crosses partitions")
            payload = scoring_payload(case)
            fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            if fingerprint in seen_hashes and seen_hashes[fingerprint] != split:
                raise ValueError("Duplicate detector input crosses partitions")
            seen_ids[case_id] = split
            seen_groups[group] = split
            seen_hashes[fingerprint] = split
            if split == "calibration" and case["label"] != 0:
                raise ValueError("Threshold calibration requires benign controls")


def calibrate_threshold(benign_scores, target_fpr=0.05):
    """One-sided order-statistic threshold; flag only score > threshold.

Finite-sample marginal bound assumes exchangeable benign episode scores,
fixed scoring rule, and independent calibration. It is not a guarantee under
distribution shift or on every task subgroup. No test scores are accepted.
"""
    values = sorted(benign_scores)
    if not 0 < target_fpr < 1 or not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Invalid calibration scores or target FPR")
    rank = math.ceil((len(values) + 1) * (1 - target_fpr))
    if rank > len(values):
        raise ValueError("Too few independent benign calibration episodes")
    return values[rank - 1]
