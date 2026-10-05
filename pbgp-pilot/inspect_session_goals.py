"""Join locally downloaded session metadata to the existing sample.

Raw metadata remains in an ignored folder; only coverage statistics are exported.
Session goals are recorded intent, not proof that every action was authorized.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sample = [json.loads(line) for line in (root / "sample_sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    wanted = {row["session_id"] for row in sample}
    matches = {}
    count = 0
    keys = set()
    with gzip.open(args.sessions, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            count += 1
            keys.update(row)
            if row.get("id") in wanted:
                matches[row["id"]] = row
    local = root / "session_context"
    local.mkdir(exist_ok=True)
    (local / "matched_sessions.json").write_text(json.dumps(matches, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.sessions.open("rb") as source_bytes:
        digest = hashlib.file_digest(source_bytes, "sha256").hexdigest()
    summary = {
        "source": "https://huggingface.co/buckets/r34lrudraa/ai-village-bucket",
        "source_file": args.sessions.name,
        "source_sha256": digest,
        "source_rows": count,
        "source_fields": sorted(keys),
        "sample_turns": len(sample),
        "sample_sessions": len(wanted),
        "matched_sessions": len(matches),
        "matched_sessions_with_goal": sum(bool(row.get("session_goal")) for row in matches.values()),
        "missing_session_ids": sorted(wanted - matches.keys()),
        "limitations": ["Recorded session intent is not a complete trusted instruction hierarchy.", "No session metadata was sent to an external scoring provider.", "Raw matched session records remain local and ignored by Git."],
    }
    (root / "session_goal_coverage.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
