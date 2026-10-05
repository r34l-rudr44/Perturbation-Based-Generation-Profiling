"""Secondary, explicitly localized diagnostic for the known faulty fields.

This is not a new detector. Tokenization may include adjacent JSON whitespace
or punctuation, which is disclosed rather than treated as an exact field-only
event probability.
"""
import json
import re
from pathlib import Path


def main():
    rows = json.loads(Path("pbgp-pilot/recorded_pilot/qwen_final_short/profiles.json").read_text(encoding="utf-8"))
    results = []
    for row in rows:
        key = "restart" if row["kind"] == "restart_after_timeout" else "scroll_amount"
        item = {"id": row["id"], "kind": row["kind"], "field": key, "candidates": {}}
        for name, candidate in row["candidates"].items():
            text = candidate["continuation"]
            match = re.search('"' + key + r'": (true|false|-?\d+)', text)
            if not match:
                raise ValueError("Faulty field value not found")
            profile = candidate["profiles"]["original"]
            selected = [(token, lp) for token, lp, start in zip(profile["tokens"], profile["logprobs"], profile["offsets"])
                        if start < match.end(1) and start + len(token) > match.start(1)]
            covered_text = "".join(token for token, _ in selected)
            item["candidates"][name] = {"value": match.group(1),
                                       "covering_token_nll": -sum(lp for _, lp in selected),
                                       "covering_tokens": [token for token, _ in selected],
                                       "value_only_except_whitespace": covered_text.strip() == match.group(1)}
        results.append(item)
    output = {"interpretation": "Secondary diagnostics on known-error fields, selected using labels; not held-out detector performance.",
              "tokenization_caution": "Compare covering-token spans only: negative scroll includes a leading-space/minus token plus a digit; the repaired digit has separate preceding whitespace. Different token counts/conditioning prevent interpreting the gap as a calibrated field-only probability test.",
              "results": results}
    Path("pbgp-pilot/recorded_field_diagnostics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
