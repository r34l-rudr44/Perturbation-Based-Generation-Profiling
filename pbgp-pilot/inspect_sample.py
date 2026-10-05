"""Audit the supplied sample and extract evidence-backed tool-error cases.

Labels use post-action tool feedback; scoring contexts contain only earlier
turns. Agent narration is not used as ground truth. No original prompts or
screenshots are reconstructed. Repair candidates are explicit counterfactuals.
"""
import copy
import json
from collections import Counter
from pathlib import Path

SAMPLE = Path("pbgp-pilot/sample_sessions.jsonl")
OUT = Path("pbgp-pilot/sample_audit")
TARGETS = {
    "ca0352c7-1af6-46d8-b816-92c26a99c90b": "restart_after_timeout",
    "bb6660a7-637e-4314-a337-09dc5b1a2661": "negative_scroll_amount",
}


def tool_calls(row):
    messages = row.get("agent_messages") or []
    if isinstance(messages, dict):
        messages = [messages]
    calls = []
    for message in messages:
        if message.get("type") == "function_call":
            calls.append({"tool": message["name"], "arguments": json.loads(message["arguments"])})
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            if fn.get("name"):
                calls.append({"tool": fn["name"], "arguments": json.loads(fn["arguments"])})
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    calls.append({"tool": block["name"], "arguments": block["input"]})
    return calls


def pre_action_contexts(session, index):
    """Build scorer inputs without consulting current or future observations."""
    prior = [{"turn_id": prev["id"], "action": prev["agent_action"],
              "output": prev.get("output"), "error": prev.get("error")}
             for prev in session[max(0, index-3):index]]
    contexts = {
        "original": "Prior tool interactions (reconstructed, original prompt unavailable):\n" + json.dumps(prior, ensure_ascii=True, sort_keys=True),
        "latest_feedback_removed": "Prior tool interactions (reconstructed, original prompt unavailable):\n" + json.dumps([dict(p, output=None, error=None) if j == len(prior)-1 else p for j, p in enumerate(prior)], ensure_ascii=True, sort_keys=True),
        "format_rephrased": "Archived earlier tool calls and their returned observations:\n" + json.dumps(prior, ensure_ascii=True, sort_keys=True, indent=2),
    }
    return prior, {k: v + "\n\nNext tool call:\n" for k, v in contexts.items()}


def main():
    rows = [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines()]
    sessions = {}
    for row in rows:
        sessions.setdefault(row["session_id"], []).append(row)
    for session in sessions.values():
        session.sort(key=lambda r: (r["created_at"], r["id"]))
    audit = {"rows": len(rows), "sessions": len(sessions),
             "message_shapes": dict(Counter(type(r.get("agent_messages")).__name__ for r in rows)),
             "nonempty_system_fields": sum(bool(r.get("system")) for r in rows),
             "nonempty_errors": sum(bool(r.get("error")) for r in rows),
             "nonempty_outputs": sum(bool(r.get("output")) for r in rows),
             "limitations": ["No trusted session goals, original prompt logs, or screenshot pixels in this sample.",
                             "Nonempty error fields include normal Git stderr; not automatic failure labels.",
                             "Selected cases demonstrate tool rejection, not malicious intent."]}
    cases = []
    for session in sessions.values():
        for i, row in enumerate(session):
            if row["id"] not in TARGETS:
                continue
            calls = tool_calls(row)
            if len(calls) != 1:
                raise ValueError("Expected one tool call in selected case")
            call = calls[0]
            kind = TARGETS[row["id"]]
            repaired = copy.deepcopy(call)
            if kind == "restart_after_timeout":
                assert i and "must be restarted" in (session[i-1].get("error") or "")
                assert call["arguments"]["restart"] is False
                assert "must restart" in row["error"]
                repaired["arguments"]["restart"] = True
            else:
                assert call["arguments"]["scroll_amount"] < 0
                assert "must be a non-negative int" in row["error"]
                repaired["arguments"]["scroll_amount"] = abs(call["arguments"]["scroll_amount"])
            prior, contexts = pre_action_contexts(session, i)
            following = session[i+1] if i+1 < len(session) else None
            # All sources for the pre-action prompt precede this row. Current
            # and later outputs are retained only in the label/evidence section.
            cases.append({"id": row["id"], "session_id": row["session_id"], "kind": kind,
                          "timestamp": row["created_at"], "label": "observed_tool_rejection",
                          "label_evidence": row["error"], "prior_turn_ids": [p["turn_id"] for p in prior],
                          "following_turn": {"id": following["id"], "calls": tool_calls(following),
                                             "error": following["error"]} if following else None,
                          "candidates": {"observed_error": call, "counterfactual_repair": repaired},
                          "repair_label": "addresses the stated precondition; not an observed successful execution",
                          "contexts": contexts})
    OUT.mkdir(exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (OUT / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print("Evidence-backed cases:", len(cases))


if __name__ == "__main__":
    main()
