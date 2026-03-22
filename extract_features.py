#!/usr/bin/env python3

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


def safe_div(a, b, default=0.0):
    return a / b if b != 0 else default


def extract_sequence_features(tokens: list[dict]) -> dict:
    """
    Extract ALL features from a single generation's token-level data.
    Returns a flat dict of scalar features.
    """
    if not tokens or len(tokens) < 3:
        return {"valid": False, "num_tokens": len(tokens) if tokens else 0}

    logprobs = np.array(
        [t["chosen_logprob"] for t in tokens if t["chosen_logprob"] is not None]
    )
    entropies = np.array([t["entropy"] for t in tokens if t["entropy"] is not None])
    ranks = np.array([t["chosen_rank"] for t in tokens if t["chosen_rank"] is not None])
    top1_probs = np.array(
        [t["top1_prob"] for t in tokens if t["top1_prob"] is not None]
    )
    top5_cumprobs = np.array(
        [t["top5_cumprob"] for t in tokens if t["top5_cumprob"] is not None]
    )
    ginis = np.array([t["gini"] for t in tokens if t.get("gini") is not None])
    kurtoses = np.array(
        [t["kurtosis"] for t in tokens if t.get("kurtosis") is not None]
    )
    num_above_01 = np.array(
        [
            t["num_tokens_above_01"]
            for t in tokens
            if t.get("num_tokens_above_01") is not None
        ]
    )

    n = len(logprobs)
    if n < 3:
        return {"valid": False, "num_tokens": n}

    features = {"valid": True, "num_tokens": n}

    features["logprob_mean"] = float(np.mean(logprobs))
    features["logprob_std"] = float(np.std(logprobs))
    features["logprob_min"] = float(np.min(logprobs))
    features["logprob_max"] = float(np.max(logprobs))
    features["logprob_median"] = float(np.median(logprobs))
    features["logprob_q10"] = float(np.percentile(logprobs, 10))
    features["logprob_q90"] = float(np.percentile(logprobs, 90))
    features["logprob_iqr"] = features["logprob_q90"] - features["logprob_q10"]
    features["logprob_skew"] = float(sp_stats.skew(logprobs))
    features["logprob_kurtosis"] = float(sp_stats.kurtosis(logprobs, fisher=True))

    features["perplexity"] = float(np.exp(-np.mean(logprobs)))

    features["entropy_mean"] = float(np.mean(entropies))
    features["entropy_std"] = float(np.std(entropies))
    features["entropy_max"] = float(np.max(entropies))
    features["entropy_min"] = float(np.min(entropies))
    features["entropy_median"] = float(np.median(entropies))
    features["entropy_skew"] = float(sp_stats.skew(entropies))
    features["entropy_kurtosis"] = float(sp_stats.kurtosis(entropies, fisher=True))
    features["varentropy"] = float(np.var(entropies))

    features["rank_mean"] = float(np.mean(ranks))
    features["rank_std"] = float(np.std(ranks))
    features["rank_max"] = float(np.max(ranks))
    features["rank_0_fraction"] = float(np.mean(ranks == 0))
    features["rank_gt5_fraction"] = float(np.mean(ranks > 5))

    features["top1_prob_mean"] = (
        float(np.mean(top1_probs)) if len(top1_probs) > 0 else 0.0
    )
    features["top5_cumprob_mean"] = (
        float(np.mean(top5_cumprobs)) if len(top5_cumprobs) > 0 else 0.0
    )
    features["gini_mean"] = float(np.mean(ginis)) if len(ginis) > 0 else 0.0
    features["vocab_spread_mean"] = (
        float(np.mean(num_above_01)) if len(num_above_01) > 0 else 0.0
    )

    d_logprobs = np.diff(logprobs)
    d_entropies = np.diff(entropies)
    d_ranks = np.diff(ranks.astype(float))

    features["d_logprob_mean"] = float(np.mean(np.abs(d_logprobs)))
    features["d_logprob_std"] = float(np.std(d_logprobs))
    features["d_logprob_max"] = float(np.max(np.abs(d_logprobs)))

    features["d_entropy_mean"] = float(np.mean(np.abs(d_entropies)))
    features["d_entropy_std"] = float(np.std(d_entropies))
    features["d_entropy_max"] = float(np.max(np.abs(d_entropies)))

    features["d_rank_mean"] = float(np.mean(np.abs(d_ranks)))
    features["d_rank_max"] = float(np.max(np.abs(d_ranks)))

    if n > 3:
        d2_logprobs = np.diff(d_logprobs)
        d2_entropies = np.diff(d_entropies)
        features["d2_logprob_mean"] = float(np.mean(np.abs(d2_logprobs)))
        features["d2_logprob_std"] = float(np.std(d2_logprobs))
        features["d2_entropy_mean"] = float(np.mean(np.abs(d2_entropies)))
        features["d2_entropy_std"] = float(np.std(d2_entropies))
    else:
        features["d2_logprob_mean"] = 0.0
        features["d2_logprob_std"] = 0.0
        features["d2_entropy_mean"] = 0.0
        features["d2_entropy_std"] = 0.0

    window_size = max(10, n // 10)
    if n > window_size * 2:
        windowed_entropy_means = []
        windowed_logprob_means = []
        for i in range(0, n - window_size, window_size // 2):
            w_ent = entropies[i : i + window_size]
            w_lp = logprobs[i : i + window_size]
            windowed_entropy_means.append(np.mean(w_ent))
            windowed_logprob_means.append(np.mean(w_lp))

        w_ent_arr = np.array(windowed_entropy_means)
        w_lp_arr = np.array(windowed_logprob_means)

        features["windowed_entropy_range"] = float(
            np.max(w_ent_arr) - np.min(w_ent_arr)
        )
        features["windowed_logprob_range"] = float(np.max(w_lp_arr) - np.min(w_lp_arr))
        features["windowed_entropy_std"] = float(np.std(w_ent_arr))
        features["windowed_logprob_std"] = float(np.std(w_lp_arr))

        if len(w_ent_arr) > 1:
            features["max_window_entropy_jump"] = float(
                np.max(np.abs(np.diff(w_ent_arr)))
            )
            features["max_window_logprob_jump"] = float(
                np.max(np.abs(np.diff(w_lp_arr)))
            )
        else:
            features["max_window_entropy_jump"] = 0.0
            features["max_window_logprob_jump"] = 0.0
    else:
        features["windowed_entropy_range"] = float(
            np.max(entropies) - np.min(entropies)
        )
        features["windowed_logprob_range"] = float(np.max(logprobs) - np.min(logprobs))
        features["windowed_entropy_std"] = float(np.std(entropies))
        features["windowed_logprob_std"] = float(np.std(logprobs))
        features["max_window_entropy_jump"] = 0.0
        features["max_window_logprob_jump"] = 0.0

    lp_z = (logprobs - np.mean(logprobs)) / (np.std(logprobs) + 1e-10)
    ent_z = (entropies - np.mean(entropies)) / (np.std(entropies) + 1e-10)

    features["logprob_spikes_2sigma"] = int(np.sum(np.abs(lp_z) > 2))
    features["logprob_spikes_3sigma"] = int(np.sum(np.abs(lp_z) > 3))
    features["entropy_spikes_2sigma"] = int(np.sum(np.abs(ent_z) > 2))
    features["entropy_spikes_3sigma"] = int(np.sum(np.abs(ent_z) > 3))

    features["spike_fraction_2sigma"] = float(np.mean(np.abs(lp_z) > 2))
    features["spike_fraction_3sigma"] = float(np.mean(np.abs(lp_z) > 3))

    for lag in [1, 5, 10]:
        if n > lag + 2:
            corr = np.corrcoef(logprobs[:-lag], logprobs[lag:])[0, 1]
            features[f"logprob_autocorr_lag{lag}"] = (
                float(corr) if not np.isnan(corr) else 0.0
            )
        else:
            features[f"logprob_autocorr_lag{lag}"] = 0.0

    for lag in [1, 5, 10]:
        if n > lag + 2:
            corr = np.corrcoef(entropies[:-lag], entropies[lag:])[0, 1]
            features[f"entropy_autocorr_lag{lag}"] = (
                float(corr) if not np.isnan(corr) else 0.0
            )
        else:
            features[f"entropy_autocorr_lag{lag}"] = 0.0

    mid = n // 2
    first_half_lp = logprobs[:mid]
    second_half_lp = logprobs[mid:]
    first_half_ent = entropies[:mid]
    second_half_ent = entropies[mid:]

    features["logprob_mean_shift"] = float(
        np.mean(second_half_lp) - np.mean(first_half_lp)
    )
    features["entropy_mean_shift"] = float(
        np.mean(second_half_ent) - np.mean(first_half_ent)
    )

    if len(first_half_lp) > 2 and len(second_half_lp) > 2:
        ks_stat_lp, ks_pval_lp = sp_stats.ks_2samp(first_half_lp, second_half_lp)
        ks_stat_ent, ks_pval_ent = sp_stats.ks_2samp(first_half_ent, second_half_ent)
        features["ks_logprob_halves_stat"] = float(ks_stat_lp)
        features["ks_logprob_halves_pval"] = float(ks_pval_lp)
        features["ks_entropy_halves_stat"] = float(ks_stat_ent)
        features["ks_entropy_halves_pval"] = float(ks_pval_ent)
    else:
        features["ks_logprob_halves_stat"] = 0.0
        features["ks_logprob_halves_pval"] = 1.0
        features["ks_entropy_halves_stat"] = 0.0
        features["ks_entropy_halves_pval"] = 1.0

    if n >= 8:
        lp_fft = np.abs(np.fft.rfft(logprobs - np.mean(logprobs)))
        ent_fft = np.abs(np.fft.rfft(entropies - np.mean(entropies)))

        features["logprob_spectral_energy"] = float(np.sum(lp_fft**2))
        features["logprob_dominant_freq"] = float(np.argmax(lp_fft[1:]) + 1) / n
        features["logprob_spectral_centroid"] = float(
            np.sum(np.arange(len(lp_fft)) * lp_fft) / (np.sum(lp_fft) + 1e-10)
        )
        features["entropy_spectral_energy"] = float(np.sum(ent_fft**2))
        features["entropy_dominant_freq"] = float(np.argmax(ent_fft[1:]) + 1) / n

        geometric_mean = np.exp(np.mean(np.log(lp_fft + 1e-15)))
        arithmetic_mean = np.mean(lp_fft) + 1e-15
        features["logprob_spectral_flatness"] = float(geometric_mean / arithmetic_mean)
    else:
        features["logprob_spectral_energy"] = 0.0
        features["logprob_dominant_freq"] = 0.0
        features["logprob_spectral_centroid"] = 0.0
        features["entropy_spectral_energy"] = 0.0
        features["entropy_dominant_freq"] = 0.0
        features["logprob_spectral_flatness"] = 0.0

    return features


def extract_all_features(data: list[dict]) -> pd.DataFrame:
    """Process all tasks and runs into a feature DataFrame."""
    rows = []

    for task_data in tqdm(data, desc="Extracting features"):
        task_id = task_data["task_id"]
        category = task_data.get("task_category", "unknown")

        for run_name, run_data in task_data["runs"].items():
            tokens = run_data.get("tokens", [])
            features = extract_sequence_features(tokens)

            if not features.get("valid", False):
                continue

            features["task_id"] = task_id
            features["task_category"] = category
            features["run_name"] = run_name
            features["mode"] = run_data.get("mode", "unknown")
            features["side_task"] = run_data.get("side_task", "none")
            features["perturbation_type"] = run_data.get("perturbation_type", "none")
            features["completion_length"] = len(run_data.get("completion", ""))

            rows.append(features)

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="~/logprob-data/raw_logprob_data.json"
    )
    parser.add_argument("--output", type=str, default="~/logprob-data/features.csv")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    print(f"Loading raw data from: {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} tasks")

    df = extract_all_features(data)
    df.to_csv(output_path, index=False)

    print(f"\nFeature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Saved to: {output_path}")

    print(f"\nRun types:")
    for mode, count in df["mode"].value_counts().items():
        print(f"  {mode}: {count}")

    feature_cols = [
        c
        for c in df.columns
        if c
        not in [
            "task_id",
            "task_category",
            "run_name",
            "mode",
            "side_task",
            "perturbation_type",
            "completion_length",
            "valid",
        ]
    ]
    print(f"\nExtracted {len(feature_cols)} features:")
    for f in sorted(feature_cols):
        print(f"  {f}")


if __name__ == "__main__":
    main()
