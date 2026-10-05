"""Produce a small, auditable report without calibrated-detection claims."""
import json
from pathlib import Path

from score_fixed_text import auc


def main():
    folder = Path("pbgp-pilot/synthetic_pilot")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    results = json.loads((folder / "profiles.json").read_text(encoding="utf-8"))
    valid = [r for r in results if not r["errors"]]
    min3 = [r for r in valid if r["profiles"]["original"]["num_tokens"] >= 3]
    lines = ["# PBGP reduced-profile pilot", "",
             f"Model: `{summary['model']}`. All candidate commands were inert text; none were executed.", "",
             "## Synthetic controls", "",
             "Six task families, each with a benign action, an unauthorized peer-suggested action, and an explicitly authorized version of the sensitive action. Both benign and unauthorized peers suggest the exact action text. Context variants remove the peer message or reformat its authority labels. An identical-input repeat measures backend noise.", "",
             f"Completed: {summary['valid_cases']}/{summary['case_count']} cases with all four valid profiles. Rejected: {summary['rejected_cases']}. Failures and timeouts are missing measurements, not positive/negative predictions.", "",
             f"Primary descriptive AUROC (peer-removal Wasserstein logprob distance): **{summary['auroc_peer_removal']:.3f}**.",
             f"Descriptive original-action mean-NLL baseline AUROC: **{summary['auroc_original_nll_baseline']:.3f}**.",
             f"Maximum identical-input repeat Wasserstein distance: **{summary['max_repeat_wasserstein']:.3f}**.", "",
             "| Case | Label | Tokens | Peer-removal W1 | Reformat W1 | Repeat W1 | Original mean NLL |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in results:
        if row["errors"]:
            lines.append(f"| {row['id']} | {row['label']} | — | unavailable | unavailable | unavailable | unavailable |")
            continue
        cmp = row["comparisons"]
        profile = row["profiles"]["original"]
        lines.append(f"| {row['id']} | {row['label']} | {profile['num_tokens']} | {cmp['peer_removed']['wasserstein_logprobs']:.4f} | {cmp['format_rephrased']['wasserstein_logprobs']:.4f} | {cmp['original_repeat']['wasserstein_logprobs']:.4f} | {profile['mean_nll']:.4f} |")
    lines.extend(["", "Label 1 denotes a synthetic unauthorized action; label 0 denotes authorization by construction.", "",
                  "## Interpretation and limits", "",
                  "- This is teacher-forced scoring by a different model, using partial logprob profiles. It does not reproduce the original PBGP model runs or recover full vocabulary entropy, exact ranks outside the returned alternatives, or the original agents' probabilities.",
                  "- The primary score direction was fixed before the run. The table retains very short actions; some are below the original extractor's three-token minimum. They can produce unstable histogram distances.",
                  f"- Descriptive sensitivity analysis for the original three-token minimum: {len(min3)} valid cases; AUROC {auc([r['label'] for r in min3], [r['comparisons']['peer_removed']['wasserstein_logprobs'] for r in min3])}. This is a secondary analysis, not a replacement for the primary result.",
                  "- Labels and prompts are synthetic and templated; action lengths differ. There is no held-out deployment test, calibrated false-positive rate, confidence interval, or evidence of prevention of the Hugging Face incident.",
                  "- Context removal changes ordinary helpful peer information too. Repeat noise and benign context sensitivity must be compared with attack scores, rather than treating any divergence as malicious intent.", "",
                  "## Recorded sample", "",
                  "AI Digest / AI Village supplied dataset: https://huggingface.co/datasets/aidigestorg/ai-village. The local sample has 577 turns across 16 sessions, zero populated system fields, and both dict/list message formats. The dataset card states original raw LLM-call prompts are excluded and recommends verifying claims against screenshots.", "",
                  "Two tool-feedback-backed cases were selected locally:", "",
                  "- `ca0352c7-1af6-46d8-b816-92c26a99c90b`: shell reuse with `restart=false` immediately after an explicit timeout/restart warning. The tool rejected the action; the following turn used a restart.",
                  "- `bb6660a7-637e-4314-a337-09dc5b1a2661`: `scroll_amount=-7`, rejected as requiring a nonnegative integer. The following turn used `7`.", "",
                  "These are operational errors, not established unauthorized goals. `sample_audit/cases.json` preserves label evidence separately from contexts containing only earlier turns. Counterfactual repairs do not claim successful execution.", "",
                  "The recorded-context remote probe was blocked by automatic approval review because it would transmit archived commands, paths, URLs and project details to UncloseAI. It has not run; explicit transfer authorization is pending. Wider session-goal/screenshot retrieval also requires authenticated access to the gated dataset; no Hugging Face token is configured here. Generated recorded-case contexts and any future remote recorded outputs are ignored by Git; the local extraction script can regenerate them from the supplied sample.", "",
                  "UncloseAI is the service name; `hermes.ai.unturf.com` is one of its routing hostnames. Its own website lists this endpoint, and its inference guide explicitly says a Hermes hostname can serve Qwen. The live catalog for this run reported Qwen3.8, while static documentation still names Qwen3.6. References: https://uncloseai.com/ and https://uncloseai.com/inference.html . Its privacy policy describes US processing, temporary console logs, and no permanent chat storage; these are provider statements, not independently audited guarantees: https://uncloseai.com/privacy-policy.html .", "",
                  "## Reproduction", "", "Run from the repository root:", "", "```powershell",
                  "git submodule update --init pbgp-context",
                  "python -m unittest discover -s pbgp-pilot -p test_score_fixed_text.py",
                  "python pbgp-pilot/inspect_sample.py",
                  "python pbgp-pilot/run_synthetic_pilot.py --output pbgp-pilot/synthetic_pilot_new",
                  "python pbgp-pilot/summarize_pilot.py", "```", "",
                  "The summarizer reads the checked-in `synthetic_pilot` results. Reruns need a new output directory to preserve prior evidence. A changed model roster must be inspected before selecting a model. No model/GPU installation is required for this remote scoring path."])
    Path("pbgp-pilot/pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
