"""Freeze calibration thresholds as soon as calibration completes.

Observes local saved profiles only. No test scores or labels enter the fit.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation_guards import calibrate_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    target = args.output / "frozen_thresholds.json"
    if target.exists():
        raise SystemExit("Thresholds already frozen")
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            rows = json.loads((args.output / "profiles.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)
            continue
        calibration = [row for row in rows if row["split"] == "calibration"]
        if len(calibration) < 20:
            time.sleep(1)
            continue
        if any(row["errors"] for row in calibration):
            raise SystemExit("Calibration incomplete; cannot freeze thresholds")
        features = {"pbgp_w1": [r["comparisons"]["wasserstein_logprobs"] for r in calibration],
                    "original_nll": [r["profiles"]["original"]["mean_nll"] for r in calibration],
                    "action_tokens": [r["profiles"]["original"]["num_tokens"] for r in calibration]}
        output = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "target_fpr": .05, "thresholds": {key: calibrate_threshold(values, .05) for key, values in features.items()},
                  "calibration_values_sha256": hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest(),
                  "test_case_rows_already_saved": sum(r["split"] == "test" for r in rows),
                  "test_scores_used": False}
        target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output), flush=True)
        return
    raise SystemExit("Calibration did not finish within watcher window")


if __name__ == "__main__":
    main()
