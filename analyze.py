#!/usr/bin/env python3

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
from tqdm.auto import tqdm

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "figure.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

HONEST_COLOR = "#58a6ff"
ATTACK_COLOR = "#f85149"
PERTURB_COLOR = "#3fb950"
PALETTE = {
    "honest": HONEST_COLOR,
    "attack": ATTACK_COLOR,
    "perturbation": PERTURB_COLOR,
}


def load_data(features_path: str, raw_path: str = None):
    """Load feature matrix and optionally raw token data."""
    df = pd.read_csv(features_path)
    raw = None
    if raw_path:
        raw_path = Path(raw_path).expanduser()
        if raw_path.exists():
            with open(raw_path) as f:
                raw = json.load(f)
    return df, raw


def plot_feature_distributions(df: pd.DataFrame, output_dir: Path):
    """Plot honest vs attack distributions for every feature."""

    print("\n[1] Plotting feature distributions...")

    honest = df[df["mode"] == "honest"]
    attack = df[df["mode"] == "attack"]

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
            "num_tokens",
        ]
    ]

    effect_sizes = []
    for col in feature_cols:
        h_vals = honest[col].dropna()
        a_vals = attack[col].dropna()
        if len(h_vals) > 2 and len(a_vals) > 2:
            pooled_std = np.sqrt((h_vals.std() ** 2 + a_vals.std() ** 2) / 2)
            d = abs(h_vals.mean() - a_vals.mean()) / (pooled_std + 1e-10)

            ks_stat, ks_pval = sp_stats.ks_2samp(h_vals, a_vals)
            effect_sizes.append((col, d, ks_stat, ks_pval))

    effect_sizes.sort(key=lambda x: x[1], reverse=True)

    es_df = pd.DataFrame(
        effect_sizes, columns=["feature", "cohens_d", "ks_stat", "ks_pval"]
    )
    es_df.to_csv(output_dir / "effect_sizes.csv", index=False)

    print(f"\n  Top 20 most discriminative features (by Cohen's d):")
    for feat, d, ks, pval in effect_sizes[:20]:
        sig = (
            "***"
            if pval < 0.001
            else "**"
            if pval < 0.01
            else "*"
            if pval < 0.05
            else ""
        )
        print(f"    {feat:40s}  d={d:.3f}  KS={ks:.3f}  p={pval:.4f} {sig}")

    top_features = [x[0] for x in effect_sizes[:16]]
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle(
        "Honest vs Attack: Feature Distributions (Top 16 by Effect Size)",
        fontsize=14,
        fontweight="bold",
    )

    for idx, (ax, feat) in enumerate(zip(axes.flat, top_features)):
        h_vals = honest[feat].dropna()
        a_vals = attack[feat].dropna()

        ax.hist(
            h_vals, bins=30, alpha=0.6, color=HONEST_COLOR, label="Honest", density=True
        )
        ax.hist(
            a_vals, bins=30, alpha=0.6, color=ATTACK_COLOR, label="Attack", density=True
        )

        d_val = effect_sizes[idx][1]
        ax.set_title(f"{feat}\n(d={d_val:.2f})", fontsize=8)
        ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(output_dir / "01_feature_distributions.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: 01_feature_distributions.png")

    return effect_sizes


def plot_trajectory_curves(raw: list[dict], output_dir: Path, num_examples: int = 6):
    """Plot logprob and entropy curves over token positions for honest vs attack."""

    if raw is None:
        print("\n[2] Skipping trajectory curves (no raw data)")
        return

    print("\n[2] Plotting trajectory curves (smoothness hypothesis)...")

    fig, axes = plt.subplots(num_examples, 4, figsize=(24, num_examples * 3.5))
    fig.suptitle(
        "Smoothness Hypothesis: Token-Level Curves — Honest vs Attack",
        fontsize=14,
        fontweight="bold",
    )

    col_titles = [
        "Logprob (Honest vs Attack)",
        "Entropy (Honest vs Attack)",
        "Rank (Honest vs Attack)",
        "|Δ Entropy| (Honest vs Attack)",
    ]

    for i, task_data in enumerate(raw[:num_examples]):
        task_id = task_data["task_id"]

        honest_tokens = task_data["runs"].get("honest", {}).get("tokens", [])
        attack_key = [k for k in task_data["runs"] if k.startswith("attack_")]
        if not attack_key:
            continue
        attack_tokens = task_data["runs"][attack_key[0]].get("tokens", [])

        if not honest_tokens or not attack_tokens:
            continue

        h_lp = [
            t["chosen_logprob"]
            for t in honest_tokens
            if t["chosen_logprob"] is not None
        ]
        a_lp = [
            t["chosen_logprob"]
            for t in attack_tokens
            if t["chosen_logprob"] is not None
        ]
        h_ent = [t["entropy"] for t in honest_tokens if t["entropy"] is not None]
        a_ent = [t["entropy"] for t in attack_tokens if t["entropy"] is not None]
        h_rank = [
            t["chosen_rank"] for t in honest_tokens if t["chosen_rank"] is not None
        ]
        a_rank = [
            t["chosen_rank"] for t in attack_tokens if t["chosen_rank"] is not None
        ]

        def smooth(arr, window=15):
            if len(arr) < window:
                return arr
            return np.convolve(arr, np.ones(window) / window, mode="valid")

        axes[i, 0].plot(
            smooth(h_lp), color=HONEST_COLOR, alpha=0.8, linewidth=1, label="Honest"
        )
        axes[i, 0].plot(
            smooth(a_lp), color=ATTACK_COLOR, alpha=0.8, linewidth=1, label="Attack"
        )
        axes[i, 0].set_ylabel(f"Task {task_id}", fontsize=8)

        axes[i, 1].plot(smooth(h_ent), color=HONEST_COLOR, alpha=0.8, linewidth=1)
        axes[i, 1].plot(smooth(a_ent), color=ATTACK_COLOR, alpha=0.8, linewidth=1)

        axes[i, 2].plot(smooth(h_rank), color=HONEST_COLOR, alpha=0.8, linewidth=1)
        axes[i, 2].plot(smooth(a_rank), color=ATTACK_COLOR, alpha=0.8, linewidth=1)

        h_d_ent = np.abs(np.diff(h_ent))
        a_d_ent = np.abs(np.diff(a_ent))
        axes[i, 3].plot(smooth(h_d_ent), color=HONEST_COLOR, alpha=0.8, linewidth=1)
        axes[i, 3].plot(smooth(a_d_ent), color=ATTACK_COLOR, alpha=0.8, linewidth=1)

        if i == 0:
            for j, title in enumerate(col_titles):
                axes[i, j].set_title(title, fontsize=9)
            axes[i, 0].legend(fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("Token Position (smoothed)", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "02_trajectory_curves.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: 02_trajectory_curves.png")


def compute_divergence_metrics(raw: list[dict], output_dir: Path) -> pd.DataFrame:
    """
    Compare attack vs isolated re-run using JSD, Wasserstein, and KS-test.
    """

    if raw is None:
        print("\n[3] Skipping divergence metrics (no raw data)")
        return pd.DataFrame()

    print("\n[3] Computing divergence metrics...")

    rows = []
    side_tasks = [
        "exfiltrate",
        "escalate_privileges",
        "disable_firewall",
        "download_malware",
    ]

    for task_data in tqdm(raw, desc="Computing divergences"):
        task_id = task_data["task_id"]

        honest_tokens = task_data["runs"].get("honest", {}).get("tokens", [])
        for pert_type in ["rephrase", "env_change", "minimal"]:
            pert_key = f"perturbation_{pert_type}"
            pert_tokens = task_data["runs"].get(pert_key, {}).get("tokens", [])
            if honest_tokens and pert_tokens:
                div = compute_pairwise_divergence(honest_tokens, pert_tokens)
                div["task_id"] = task_id
                div["comparison"] = f"honest_vs_perturb_{pert_type}"
                div["is_attack"] = False
                rows.append(div)

        for st in side_tasks:
            attack_key = f"attack_{st}"
            isolated_key = f"isolated_rerun_{st}"
            attack_tokens = task_data["runs"].get(attack_key, {}).get("tokens", [])
            isolated_tokens = task_data["runs"].get(isolated_key, {}).get("tokens", [])
            if attack_tokens and isolated_tokens:
                div = compute_pairwise_divergence(attack_tokens, isolated_tokens)
                div["task_id"] = task_id
                div["comparison"] = f"attack_{st}_vs_isolated"
                div["is_attack"] = True
                div["side_task"] = st
                rows.append(div)

        for st in side_tasks:
            isolated_key = f"isolated_rerun_{st}"
            isolated_tokens = task_data["runs"].get(isolated_key, {}).get("tokens", [])
            if honest_tokens and isolated_tokens:
                div = compute_pairwise_divergence(honest_tokens, isolated_tokens)
                div["task_id"] = task_id
                div["comparison"] = f"honest_vs_isolated_{st}"
                div["is_attack"] = False
                rows.append(div)

    div_df = pd.DataFrame(rows)
    if len(div_df) > 0:
        div_df.to_csv(output_dir / "divergence_metrics.csv", index=False)
        print(f"  Computed {len(div_df)} pairwise comparisons")

    return div_df


def compute_pairwise_divergence(tokens_a: list[dict], tokens_b: list[dict]) -> dict:
    """Compute all divergence metrics between two token sequences."""

    lp_a = np.array(
        [t["chosen_logprob"] for t in tokens_a if t["chosen_logprob"] is not None]
    )
    lp_b = np.array(
        [t["chosen_logprob"] for t in tokens_b if t["chosen_logprob"] is not None]
    )
    ent_a = np.array([t["entropy"] for t in tokens_a if t["entropy"] is not None])
    ent_b = np.array([t["entropy"] for t in tokens_b if t["entropy"] is not None])
    rank_a = np.array(
        [t["chosen_rank"] for t in tokens_a if t["chosen_rank"] is not None],
        dtype=float,
    )
    rank_b = np.array(
        [t["chosen_rank"] for t in tokens_b if t["chosen_rank"] is not None],
        dtype=float,
    )

    result = {}

    if len(lp_a) > 5 and len(lp_b) > 5:
        bins = np.linspace(min(lp_a.min(), lp_b.min()), max(lp_a.max(), lp_b.max()), 50)
        hist_a, _ = np.histogram(lp_a, bins=bins, density=True)
        hist_b, _ = np.histogram(lp_b, bins=bins, density=True)
        hist_a = hist_a + 1e-10
        hist_b = hist_b + 1e-10
        hist_a = hist_a / hist_a.sum()
        hist_b = hist_b / hist_b.sum()
        result["jsd_logprob"] = float(jensenshannon(hist_a, hist_b) ** 2)
    else:
        result["jsd_logprob"] = 0.0

    if len(ent_a) > 5 and len(ent_b) > 5:
        bins = np.linspace(
            min(ent_a.min(), ent_b.min()), max(ent_a.max(), ent_b.max()), 50
        )
        hist_a, _ = np.histogram(ent_a, bins=bins, density=True)
        hist_b, _ = np.histogram(ent_b, bins=bins, density=True)
        hist_a = hist_a + 1e-10
        hist_b = hist_b + 1e-10
        hist_a = hist_a / hist_a.sum()
        hist_b = hist_b / hist_b.sum()
        result["jsd_entropy"] = float(jensenshannon(hist_a, hist_b) ** 2)
    else:
        result["jsd_entropy"] = 0.0

    if len(lp_a) > 2 and len(lp_b) > 2:
        result["wasserstein_logprob"] = float(wasserstein_distance(lp_a, lp_b))
        result["wasserstein_entropy"] = float(wasserstein_distance(ent_a, ent_b))
        result["wasserstein_rank"] = float(wasserstein_distance(rank_a, rank_b))
    else:
        result["wasserstein_logprob"] = 0.0
        result["wasserstein_entropy"] = 0.0
        result["wasserstein_rank"] = 0.0

    if len(lp_a) > 2 and len(lp_b) > 2:
        ks_stat, ks_pval = sp_stats.ks_2samp(lp_a, lp_b)
        result["ks_logprob_stat"] = float(ks_stat)
        result["ks_logprob_pval"] = float(ks_pval)
        ks_stat, ks_pval = sp_stats.ks_2samp(ent_a, ent_b)
        result["ks_entropy_stat"] = float(ks_stat)
        result["ks_entropy_pval"] = float(ks_pval)
    else:
        result["ks_logprob_stat"] = 0.0
        result["ks_logprob_pval"] = 1.0
        result["ks_entropy_stat"] = 0.0
        result["ks_entropy_pval"] = 1.0

    result["mean_logprob_diff"] = (
        float(abs(np.mean(lp_a) - np.mean(lp_b)))
        if len(lp_a) > 0 and len(lp_b) > 0
        else 0.0
    )
    result["mean_entropy_diff"] = (
        float(abs(np.mean(ent_a) - np.mean(ent_b)))
        if len(ent_a) > 0 and len(ent_b) > 0
        else 0.0
    )
    result["std_logprob_diff"] = (
        float(abs(np.std(lp_a) - np.std(lp_b)))
        if len(lp_a) > 0 and len(lp_b) > 0
        else 0.0
    )
    result["std_entropy_diff"] = (
        float(abs(np.std(ent_a) - np.std(ent_b)))
        if len(ent_a) > 0 and len(ent_b) > 0
        else 0.0
    )

    result["mmd_logprob"] = (
        compute_mmd(lp_a.reshape(-1, 1), lp_b.reshape(-1, 1))
        if len(lp_a) > 5 and len(lp_b) > 5
        else 0.0
    )

    if len(lp_a) > 5 and len(lp_b) > 5:
        min_len = min(
            len(lp_a), len(ent_a), len(rank_a), len(lp_b), len(ent_b), len(rank_b)
        )
        if min_len > 5:
            X = np.column_stack([lp_a[:min_len], ent_a[:min_len], rank_a[:min_len]])
            Y = np.column_stack([lp_b[:min_len], ent_b[:min_len], rank_b[:min_len]])
            result["mmd_multivariate"] = compute_mmd(X, Y)
        else:
            result["mmd_multivariate"] = 0.0
    else:
        result["mmd_multivariate"] = 0.0

    return result


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    """Compute Maximum Mean Discrepancy with RBF kernel."""
    from sklearn.metrics.pairwise import rbf_kernel

    if gamma is None:
        all_data = np.vstack([X, Y])
        from sklearn.metrics import pairwise_distances

        dists = pairwise_distances(all_data, metric="euclidean")
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * median_dist**2 + 1e-10)

    max_n = 500
    if len(X) > max_n:
        idx = np.random.choice(len(X), max_n, replace=False)
        X = X[idx]
    if len(Y) > max_n:
        idx = np.random.choice(len(Y), max_n, replace=False)
        Y = Y[idx]

    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)

    mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return float(max(0, mmd2))


def plot_divergence_results(div_df: pd.DataFrame, output_dir: Path):
    """Plot the divergence comparison results."""

    if div_df.empty:
        print("\n[3b] Skipping divergence plots (no data)")
        return

    print("\n[3b] Plotting divergence results...")

    metrics = [
        "jsd_logprob",
        "jsd_entropy",
        "wasserstein_logprob",
        "wasserstein_entropy",
        "mmd_multivariate",
        "ks_logprob_stat",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Attack vs Isolated Re-run Divergence\n(Attack comparisons should show HIGHER divergence)",
        fontsize=13,
        fontweight="bold",
    )

    for ax, metric in zip(axes.flat, metrics):
        attack_vals = div_df[div_df["is_attack"] == True][metric].dropna()
        benign_vals = div_df[div_df["is_attack"] == False][metric].dropna()

        if len(attack_vals) > 0 and len(benign_vals) > 0:
            ax.hist(
                benign_vals,
                bins=25,
                alpha=0.6,
                color=HONEST_COLOR,
                label=f"Benign (n={len(benign_vals)})",
                density=True,
            )
            ax.hist(
                attack_vals,
                bins=25,
                alpha=0.6,
                color=ATTACK_COLOR,
                label=f"Attack (n={len(attack_vals)})",
                density=True,
            )

            pooled_std = np.sqrt((attack_vals.std() ** 2 + benign_vals.std() ** 2) / 2)
            d = abs(attack_vals.mean() - benign_vals.mean()) / (pooled_std + 1e-10)
            ax.set_title(f"{metric}\n(Cohen's d = {d:.2f})", fontsize=9)
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "03_divergence.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: 03_divergence.png")


def plot_pca_space(df: pd.DataFrame, output_dir: Path):
    """Project feature space to 2D and 3D to see if honest/attack cluster."""

    print("\n[4] Plotting PCA projections...")

    honest_attack = df[df["mode"].isin(["honest", "attack"])].copy()

    feature_cols = [
        c
        for c in honest_attack.columns
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
            "num_tokens",
        ]
        and honest_attack[c].dtype in [np.float64, np.int64]
    ]

    X = honest_attack[feature_cols].fillna(0).values
    y = (honest_attack["mode"] == "attack").astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca2d = PCA(n_components=2)
    X_2d = pca2d.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Feature Space: Honest vs Attack", fontsize=13, fontweight="bold")

    colors = [HONEST_COLOR if yi == 0 else ATTACK_COLOR for yi in y]
    labels = ["Honest" if yi == 0 else "Attack" for yi in y]

    axes[0].scatter(
        X_2d[y == 0, 0],
        X_2d[y == 0, 1],
        c=HONEST_COLOR,
        alpha=0.5,
        s=20,
        label="Honest",
    )
    axes[0].scatter(
        X_2d[y == 1, 0],
        X_2d[y == 1, 1],
        c=ATTACK_COLOR,
        alpha=0.5,
        s=20,
        label="Attack",
    )
    axes[0].set_xlabel(f"PC1 ({pca2d.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca2d.explained_variance_ratio_[1]:.1%})")
    axes[0].set_title("PCA 2D Projection")
    axes[0].legend()

    side_task_colors = {
        "exfiltrate": "#f85149",
        "escalate_privileges": "#d29922",
        "disable_firewall": "#a371f7",
        "download_malware": "#f778ba",
    }

    attacks_only = honest_attack[honest_attack["mode"] == "attack"].copy()
    attack_indices = honest_attack.index[honest_attack["mode"] == "attack"]

    axes[1].scatter(
        X_2d[y == 0, 0],
        X_2d[y == 0, 1],
        c=HONEST_COLOR,
        alpha=0.3,
        s=15,
        label="Honest",
    )

    for st, color in side_task_colors.items():
        mask = attacks_only["side_task"] == st
        if mask.any():
            positions = np.where(y == 1)[0]
            st_indices = attacks_only.index[mask]

            for idx in st_indices:
                pos = np.where(honest_attack.index == idx)[0]
                if len(pos) > 0:
                    axes[1].scatter(
                        X_2d[pos, 0], X_2d[pos, 1], c=color, alpha=0.7, s=25, label=st
                    )

    handles, lbls = axes[1].get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    axes[1].legend(by_label.values(), by_label.keys(), fontsize=7)
    axes[1].set_xlabel(f"PC1")
    axes[1].set_ylabel(f"PC2")
    axes[1].set_title("PCA colored by side task")

    plt.tight_layout()
    plt.savefig(output_dir / "04_pca_projection.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: 04_pca_projection.png")

    pca_full = PCA(n_components=min(20, len(feature_cols)))
    pca_full.fit(X_scaled)
    print(
        f"  Explained variance (first 10 PCs): {pca_full.explained_variance_ratio_[:10].cumsum()}"
    )


def train_classifiers(df: pd.DataFrame, output_dir: Path):
    """Train simple classifiers and report AUROC."""

    print("\n[5] Training classifiers...")

    honest_attack = df[df["mode"].isin(["honest", "attack"])].copy()

    feature_cols = [
        c
        for c in honest_attack.columns
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
            "num_tokens",
        ]
        and honest_attack[c].dtype in [np.float64, np.int64]
    ]

    X = honest_attack[feature_cols].fillna(0).values
    y = (honest_attack["mode"] == "attack").astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Logprob-based Attack Detection: ROC Curves (5-fold CV)",
        fontsize=13,
        fontweight="bold",
    )

    for (name, clf), ax in zip(classifiers.items(), axes):
        all_y_true = []
        all_y_prob = []
        aucs = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
            clf_clone = type(clf)(**clf.get_params())
            clf_clone.fit(X_scaled[train_idx], y[train_idx])
            y_prob = clf_clone.predict_proba(X_scaled[test_idx])[:, 1]

            auc = roc_auc_score(y[test_idx], y_prob)
            aucs.append(auc)
            all_y_true.extend(y[test_idx])
            all_y_prob.extend(y_prob)

            fpr, tpr, _ = roc_curve(y[test_idx], y_prob)
            ax.plot(fpr, tpr, alpha=0.3, linewidth=1)

        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)
        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        ax.plot(
            fpr,
            tpr,
            color=ATTACK_COLOR,
            linewidth=2,
            label=f"Mean AUROC = {mean_auc:.3f} ± {std_auc:.3f}",
        )
        ax.plot([0, 1], [0, 1], "--", color="#8b949e", alpha=0.5)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(name)
        ax.legend(fontsize=9)

        results[name] = {"auroc_mean": mean_auc, "auroc_std": std_auc}
        print(f"  {name}: AUROC = {mean_auc:.3f} ± {std_auc:.3f}")

    plt.tight_layout()
    plt.savefig(output_dir / "05_roc_curves.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: 05_roc_curves.png")

    best_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    best_clf.fit(X_scaled, y)

    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": best_clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_imp = importances.head(25)
    ax.barh(
        range(len(top_imp)), top_imp["importance"].values, color=ATTACK_COLOR, alpha=0.7
    )
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_imp["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 25 Features for Attack Detection (Random Forest)")
    plt.tight_layout()
    plt.savefig(output_dir / "06_feature_importance.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: 06_feature_importance.png")

    importances.to_csv(output_dir / "feature_importances.csv", index=False)

    return results


def plot_per_side_task(df: pd.DataFrame, output_dir: Path):
    """Break down detection performance by side task type."""

    print("\n[6] Per-side-task analysis...")

    attack_df = df[df["mode"] == "attack"].copy()
    honest_df = df[df["mode"] == "honest"].copy()

    if attack_df.empty:
        print("  No attack data, skipping")
        return

    side_tasks = attack_df["side_task"].unique()
    key_features = [
        "entropy_mean",
        "logprob_std",
        "d_entropy_max",
        "rank_gt5_fraction",
        "varentropy",
        "logprob_spectral_flatness",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Feature Distributions by Side Task Type", fontsize=13, fontweight="bold"
    )

    for ax, feat in zip(axes.flat, key_features):
        data_dict = {"Honest": honest_df[feat].dropna().values}
        for st in side_tasks:
            st_vals = attack_df[attack_df["side_task"] == st][feat].dropna().values
            if len(st_vals) > 0:
                data_dict[st] = st_vals

        positions = range(len(data_dict))
        bp = ax.boxplot(
            data_dict.values(), positions=positions, patch_artist=True, widths=0.6
        )

        colors_list = [HONEST_COLOR] + [ATTACK_COLOR] * (len(data_dict) - 1)
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_xticklabels(data_dict.keys(), rotation=30, ha="right", fontsize=7)
        ax.set_title(feat, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "07_per_side_task.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: 07_per_side_task.png")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path):
    """Show feature correlations to identify redundant vs. independent signals."""

    print("\n[7] Plotting correlation matrix...")

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
            "num_tokens",
        ]
        and df[c].dtype in [np.float64, np.int64]
    ]

    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5)
    plt.tight_layout()
    plt.savefig(output_dir / "08_correlation_matrix.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: 08_correlation_matrix.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="~/logprob-data/features.csv")
    parser.add_argument(
        "--raw", type=str, default="~/logprob-data/raw_logprob_data.json"
    )
    parser.add_argument("--output-dir", type=str, default="~/logprob-data/plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    df, raw = load_data(Path(args.features).expanduser(), Path(args.raw).expanduser())

    print(f"Loaded feature matrix: {df.shape}")
    print(f"Modes: {dict(df['mode'].value_counts())}")

    effect_sizes = plot_feature_distributions(df, output_dir)
    plot_trajectory_curves(raw, output_dir)
    div_df = compute_divergence_metrics(raw, output_dir)
    plot_divergence_results(div_df, output_dir)
    plot_pca_space(df, output_dir)
    classifier_results = train_classifiers(df, output_dir)
    plot_per_side_task(df, output_dir)
    plot_correlation_matrix(df, output_dir)

    print("\n" + "=" * 70)
    print("  ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\n  Dataset: {len(df)} total generations")
    print(f"    Honest: {len(df[df['mode'] == 'honest'])}")
    print(f"    Attack: {len(df[df['mode'] == 'attack'])}")
    print(f"    Perturbation: {len(df[df['mode'] == 'perturbation'])}")
    print(f"    Isolated Re-run: {len(df[df['mode'] == 'isolated_rerun'])}")

    if effect_sizes:
        print(f"\n  Top 5 discriminative features:")
        for feat, d, ks, pval in effect_sizes[:5]:
            print(f"    {feat}: Cohen's d = {d:.3f}, KS p-value = {pval:.6f}")

    print(f"\n  Classifier AUROC:")
    for name, res in classifier_results.items():
        print(f"    {name}: {res['auroc_mean']:.3f} ± {res['auroc_std']:.3f}")

    if not div_df.empty:
        attack_divs = div_df[div_df["is_attack"] == True]
        benign_divs = div_df[div_df["is_attack"] == False]
        for metric in ["jsd_logprob", "wasserstein_entropy", "mmd_multivariate"]:
            if metric in div_df.columns:
                a_mean = attack_divs[metric].mean()
                b_mean = benign_divs[metric].mean()
                ratio = a_mean / (b_mean + 1e-10)
                print(f"\n  divergence ({metric}):")
                print(f"    Attack pairs: {a_mean:.4f}")
                print(f"    Benign pairs: {b_mean:.4f}")
                print(f"    Ratio: {ratio:.2f}x")

    print(f"\n  All plots saved to: {output_dir}")
    print(f"  Files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"    {f.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
