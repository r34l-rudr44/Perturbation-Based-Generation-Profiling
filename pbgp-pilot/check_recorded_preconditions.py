"""Retrospective rule checks for the two inspected operational error cases.

Inputs are the candidate tool call and prior observations only. Constraints
come from the inspected tool feedback; this is a consistency demonstration,
not an independently held-out detector evaluation or a PBGP result.
"""
import json
from pathlib import Path


def violations(call, last_error):
    args = call["arguments"]
    flags = []
    if call["tool"] == "bash" and "must be restarted" in (last_error or "") and args.get("restart") is not True:
        flags.append("restart_required_after_timeout")
    if call["tool"] == "use_computer" and args.get("action") == "scroll":
        amount = args.get("scroll_amount")
        if type(amount) is not int or amount < 0:
            flags.append("scroll_amount_must_be_nonnegative_integer")
    return flags


def main():
    cases = json.loads(Path("pbgp-pilot/sample_audit/cases.json").read_text(encoding="utf-8"))
    report = {"interpretation": "Retrospective constraints derived from inspected tool feedback; no PBGP scores or held-out accuracy.",
              "results": []}
    for case in cases:
        prior = json.loads(case["contexts"]["original"].split("\n", 1)[1].rsplit("\n\nNext tool call:\n", 1)[0])
        last_error = prior[-1].get("error") if prior else None
        report["results"].append({"id": case["id"], "kind": case["kind"],
                                  "observed_flags": violations(case["candidates"]["observed_error"], last_error),
                                  "repair_flags": violations(case["candidates"]["counterfactual_repair"], last_error)})
    Path("pbgp-pilot/recorded_rule_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
