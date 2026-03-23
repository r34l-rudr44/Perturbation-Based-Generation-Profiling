# Perturbation-Based Generation Profiling Detects Covert AI Agent Attacks Where Token-Level Statistics Fail

**AI Control Hackathon 2026 — Track 2: Control Protocol Design**

## Research Question

Can we detect misaligned AI agent behavior by analyzing the statistical properties of the model's output token distributions — without reasoning about the semantics of individual actions?

## Core Hypothesis

**Smoothness Hypothesis:** An honest model solving a task produces token-level statistics (logprobs, entropy, rank distributions) with a smoother, more consistent profile than a model covertly pursuing a malicious side objective. The dual-objective nature of attacking — completing the main task while hiding sabotage — creates detectable statistical anomalies.

**Protocol:** When the statistical anomaly detector flags an action, re-run the model on the same task with a perturbed context (rephrased task, modified environment). If the model's generation profile changes significantly (high JSD / Wasserstein / MMD divergence), the original action was context-dependent and likely honest. If the profile remains anomalous regardless of perturbation, the action was driven by a fixed malicious objective.

## What This Measures

| Metric            | What it captures                                        |
| ----------------- | ------------------------------------------------------- |
| Logprob mean/std  | How expected vs. surprising the output is               |
| Entropy curves    | Model uncertainty over time — regime changes            |
| Rank distribution | How often the model picks non-top tokens                |
| Varentropy        | Inconsistency of uncertainty (dual-objective signature) |
| Spectral features | Frequency-domain anomalies in the generation pattern    |
| JSD / Wasserstein | Distributional divergence between original and re-run   |
| MMD               | Kernel-based two-sample test on multivariate features   |

## Hardware Requirements

- GPU with ≥16GB VRAM for 8B model (≥40GB for quantized 70B)
- Tested on: NVIDIA GH200 (96GB HBM3), 64-core ARM Neoverse-V2

## Quick Start

```bash
hf login

hf download huihui-ai/Huihui-Qwen3.5-27B-abliterated

./run_pipeline.sh --model huihui-ai/Huihui-Qwen3.5-27B-abliterated --num-tasks 5

./run_pipeline.sh --model huihui-ai/Huihui-Qwen3.5-27B-abliterated --num-tasks 50
```

## Citation

If you find this work useful in your research, please consider citing it:

```
@misc{
  title={(HckPrj) Perturbation-Based Generation Profiling Detects Covert AI Agent Attacks Where Token-Level Statistics Fail},
  author={Yatharth Maheshwari, Arka Dash, Abhineet Som},
  date={3/22/26},
  organization={Apart Research},
  note={Research submission to the research sprint hosted by Apart.},
  howpublished={https://apartresearch.com}
}
```
