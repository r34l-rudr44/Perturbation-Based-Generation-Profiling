#!/usr/bin/env python3

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr, wasserstein_distance
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
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
BLUE = "#58a6ff"
RED = "#f85149"
GREEN = "#3fb950"
PURPLE = "#a371f7"
YELLOW = "#d29922"


def load_run(data_dir):
    """Load features.csv and raw JSON from a run directory."""
    data_dir = Path(data_dir).expanduser()
    df = pd.read_csv(data_dir / "features.csv")
    raw = None
    raw_path = data_dir / "raw_logprob_data.json"
    if raw_path.exists():
        with open(raw_path) as f:
            raw = json.load(f)
    return df, raw


def compute_divergences_from_raw(raw):
    """Recompute pairwise divergences from raw data."""
    rows = []
    for task_data in raw:
        task_id = task_data["task_id"]
        honest_tokens = task_data["runs"].get("honest", {}).get("tokens", [])

        for run_name, run_data in task_data["runs"].items():
            if not run_name.startswith("attack_"):
                continue

            side_task = run_data.get("side_task", run_name.replace("attack_", ""))
            attack_tokens = run_data.get("tokens", [])
            isolated_key = f"isolated_rerun_{side_task}"
            isolated_tokens = task_data["runs"].get(isolated_key, {}).get("tokens", [])

            if attack_tokens and isolated_tokens:
                div = _pairwise_div(attack_tokens, isolated_tokens)
                div["task_id"] = task_id
                div["comparison"] = "attack_vs_isolated"
                div["is_attack"] = True
                div["side_task"] = side_task
                rows.append(div)

            if honest_tokens and isolated_tokens:
                div = _pairwise_div(honest_tokens, isolated_tokens)
                div["task_id"] = task_id
                div["comparison"] = "honest_vs_isolated"
                div["is_attack"] = False
                div["side_task"] = side_task
                rows.append(div)

        for run_name, run_data in task_data["runs"].items():
            if not run_name.startswith("perturbation_"):
                continue
            pert_tokens = run_data.get("tokens", [])
            if honest_tokens and pert_tokens:
                div = _pairwise_div(honest_tokens, pert_tokens)
                div["task_id"] = task_id
                div["comparison"] = "honest_vs_perturbation"
                div["is_attack"] = False
                rows.append(div)

    return pd.DataFrame(rows)


def _pairwise_div(tokens_a, tokens_b):
    """Compute divergence metrics between two token sequences."""
    lp_a = np.array(
        [t["chosen_logprob"] for t in tokens_a if t.get("chosen_logprob") is not None]
    )
    lp_b = np.array(
        [t["chosen_logprob"] for t in tokens_b if t.get("chosen_logprob") is not None]
    )
    ent_a = np.array([t["entropy"] for t in tokens_a if t.get("entropy") is not None])
    ent_b = np.array([t["entropy"] for t in tokens_b if t.get("entropy") is not None])
    rank_a = np.array(
        [t["chosen_rank"] for t in tokens_a if t.get("chosen_rank") is not None],
        dtype=float,
    )
    rank_b = np.array(
        [t["chosen_rank"] for t in tokens_b if t.get("chosen_rank") is not None],
        dtype=float,
    )

    result = {}

    if len(lp_a) > 5 and len(lp_b) > 5:
        lo = min(lp_a.min(), lp_b.min())
        hi = max(lp_a.max(), lp_b.max())
        bins = np.linspace(lo, hi, 50)
        ha, _ = np.histogram(lp_a, bins=bins, density=True)
        hb, _ = np.histogram(lp_b, bins=bins, density=True)
        ha = ha + 1e-10
        hb = hb + 1e-10
        ha /= ha.sum()
        hb /= hb.sum()
        result["jsd_logprob"] = float(jensenshannon(ha, hb) ** 2)
    else:
        result["jsd_logprob"] = 0.0

    if len(ent_a) > 5 and len(ent_b) > 5:
        lo = min(ent_a.min(), ent_b.min())
        hi = max(ent_a.max(), ent_b.max())
        bins = np.linspace(lo, hi, 50)
        ha, _ = np.histogram(ent_a, bins=bins, density=True)
        hb, _ = np.histogram(ent_b, bins=bins, density=True)
        ha = ha + 1e-10
        hb = hb + 1e-10
        ha /= ha.sum()
        hb /= hb.sum()
        result["jsd_entropy"] = float(jensenshannon(ha, hb) ** 2)
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
        ks, kp = ks_2samp(lp_a, lp_b)
        result["ks_logprob_stat"] = float(ks)
        result["ks_logprob_pval"] = float(kp)
    else:
        result["ks_logprob_stat"] = 0.0
        result["ks_logprob_pval"] = 1.0

    if len(lp_a) > 5 and len(lp_b) > 5:
        min_n = min(
            len(lp_a), len(ent_a), len(rank_a), len(lp_b), len(ent_b), len(rank_b)
        )
        if min_n > 5:
            X = np.column_stack([lp_a[:min_n], ent_a[:min_n], rank_a[:min_n]])
            Y = np.column_stack([lp_b[:min_n], ent_b[:min_n], rank_b[:min_n]])
            result["mmd_multivariate"] = _mmd(X, Y)
        else:
            result["mmd_multivariate"] = 0.0
    else:
        result["mmd_multivariate"] = 0.0

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

    return result


def _mmd(X, Y, gamma=None):
    from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel

    if gamma is None:
        all_d = np.vstack([X[:200], Y[:200]])
        dists = pairwise_distances(all_d)
        med = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * med**2 + 1e-10)
    mx = min(300, len(X))
    my = min(300, len(Y))
    X = X[np.random.choice(len(X), mx, replace=False)] if len(X) > mx else X
    Y = Y[np.random.choice(len(Y), my, replace=False)] if len(Y) > my else Y
    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)
    return float(max(0, Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()))


def test_divergence_significance(div_df, out):
    print("\n" + "=" * 70)
    print("  TEST 1: Mann-Whitney U on Divergence Metrics")
    print("=" * 70)

    metrics = [
        "jsd_logprob",
        "jsd_entropy",
        "wasserstein_logprob",
        "wasserstein_entropy",
        "wasserstein_rank",
        "mmd_multivariate",
        "mean_logprob_diff",
        "mean_entropy_diff",
    ]

    attack = div_df[div_df["is_attack"] == True]
    benign = div_df[div_df["is_attack"] == False]

    rows = []
    for m in metrics:
        a_vals = attack[m].dropna()
        b_vals = benign[m].dropna()
        if len(a_vals) < 3 or len(b_vals) < 3:
            continue
        u_stat, p_val = mannwhitneyu(a_vals, b_vals, alternative="greater")
        ratio = a_vals.mean() / (b_vals.mean() + 1e-10)
        sig = (
            "***"
            if p_val < 0.001
            else "**"
            if p_val < 0.01
            else "*"
            if p_val < 0.05
            else ""
        )
        rows.append(
            {
                "metric": m,
                "attack_mean": f"{a_vals.mean():.4f}",
                "benign_mean": f"{b_vals.mean():.4f}",
                "ratio": f"{ratio:.2f}x",
                "U_stat": int(u_stat),
                "p_value": f"{p_val:.6f}",
                "sig": sig,
            }
        )
        print(
            f"  {m:30s}  attack={a_vals.mean():.4f}  benign={b_vals.mean():.4f}  "
            f"ratio={ratio:.2f}x  U={int(u_stat):5d}  p={p_val:.6f} {sig}"
        )

    pd.DataFrame(rows).to_csv(out / "divergence_mannwhitney.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(
        "Divergence Distributions — Attack vs Benign Pairs\n(Mann-Whitney U test)",
        fontsize=13,
        fontweight="bold",
    )
    for ax, m in zip(axes.flat, metrics):
        a_vals = attack[m].dropna()
        b_vals = benign[m].dropna()
        if len(a_vals) < 2:
            continue
        u_stat, p_val = mannwhitneyu(a_vals, b_vals, alternative="greater")
        ax.hist(
            b_vals,
            bins=20,
            alpha=0.6,
            color=BLUE,
            label=f"Benign (n={len(b_vals)})",
            density=True,
        )
        ax.hist(
            a_vals,
            bins=20,
            alpha=0.6,
            color=RED,
            label=f"Attack (n={len(a_vals)})",
            density=True,
        )
        sig = (
            "***"
            if p_val < 0.001
            else "**"
            if p_val < 0.01
            else "*"
            if p_val < 0.05
            else "ns"
        )
        ax.set_title(f"{m}\np={p_val:.4f} {sig}", fontsize=8)
        ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(out / "09_divergence_significance.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: 09_divergence_significance.png")


def divergence_classifier(div_df, out):
    print("\n" + "=" * 70)
    print("  TEST 2: Classifier Using ONLY Divergence Features")
    print("=" * 70)

    feature_cols = [
        "jsd_logprob",
        "jsd_entropy",
        "wasserstein_logprob",
        "wasserstein_entropy",
        "wasserstein_rank",
        "mmd_multivariate",
        "mean_logprob_diff",
        "mean_entropy_diff",
        "std_logprob_diff",
        "std_entropy_diff",
    ]

    available = [c for c in feature_cols if c in div_df.columns]
    sub = div_df[available + ["is_attack"]].dropna()

    X = sub[available].values
    y = sub["is_attack"].astype(int).values

    if len(X) < 20 or y.sum() < 5 or (len(y) - y.sum()) < 5:
        print("  Insufficient data for classifier")
        return {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clfs = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Classifier: ROC Curves (Divergence Features Only)",
        fontsize=13,
        fontweight="bold",
    )

    n_splits = min(5, min(y.sum(), len(y) - y.sum()))
    if n_splits < 2:
        print("  Not enough samples per class for CV")
        return {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for (name, clf), ax in zip(clfs.items(), axes):
        aucs = []
        all_y, all_p = [], []

        for tr, te in skf.split(X_scaled, y):
            c = type(clf)(**clf.get_params())
            c.fit(X_scaled[tr], y[tr])
            p = c.predict_proba(X_scaled[te])[:, 1]
            auc = roc_auc_score(y[te], p)
            aucs.append(auc)
            all_y.extend(y[te])
            all_p.extend(p)
            fpr, tpr, _ = roc_curve(y[te], p)
            ax.plot(fpr, tpr, alpha=0.3, linewidth=1)

        all_y, all_p = np.array(all_y), np.array(all_p)
        fpr, tpr, _ = roc_curve(all_y, all_p)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        ax.plot(
            fpr,
            tpr,
            color=RED,
            linewidth=2,
            label=f"AUROC = {mean_auc:.3f} ± {std_auc:.3f}",
        )
        ax.plot([0, 1], [0, 1], "--", color="#8b949e", alpha=0.5)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(name)
        ax.legend(fontsize=9)
        results[name] = {"auroc": mean_auc, "std": std_auc}
        print(f"  {name}: AUROC = {mean_auc:.3f} ± {std_auc:.3f}")

    plt.tight_layout()
    plt.savefig(out / "10_divergence_classifier.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: 10_divergence_classifier.png")
    return results


def length_confound_check(df, out):
    print("\n" + "=" * 70)
    print("  TEST 3: Length Confound Analysis")
    print("=" * 70)

    ha = df[df["mode"].isin(["honest", "attack"])].copy()
    honest_len = ha[ha["mode"] == "honest"]["num_tokens"]
    attack_len = ha[ha["mode"] == "attack"]["num_tokens"]

    print(
        f"  Honest tokens:  mean={honest_len.mean():.0f}, std={honest_len.std():.0f}, "
        f"range=[{honest_len.min():.0f}, {honest_len.max():.0f}]"
    )
    print(
        f"  Attack tokens:  mean={attack_len.mean():.0f}, std={attack_len.std():.0f}, "
        f"range=[{attack_len.min():.0f}, {attack_len.max():.0f}]"
    )

    u, p = mannwhitneyu(honest_len, attack_len, alternative="two-sided")
    print(
        f"  Length difference: U={int(u)}, p={p:.4f} {'(significant)' if p < 0.05 else '(not significant)'}"
    )

    top_feats = [
        "d_entropy_max",
        "entropy_mean_shift",
        "logprob_mean_shift",
        "d_logprob_max",
        "entropy_max",
        "entropy_std",
        "varentropy",
    ]

    print(f"\n  Feature correlations with num_tokens:")
    for feat in top_feats:
        if feat in ha.columns:
            r, p = spearmanr(ha["num_tokens"], ha[feat], nan_policy="omit")
            flag = " [I]" if abs(r) > 0.4 else ""
            print(f"    {feat:30s}  r={r:.3f}  p={p:.4f}{flag}")

    print(f"\n  Length-controlled AUROC (features residualized on token count):")
    feature_cols = [
        c
        for c in ha.columns
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
        and ha[c].dtype in [np.float64, np.int64]
    ]

    X = ha[feature_cols].fillna(0).values
    y = (ha["mode"] == "attack").astype(int).values
    lengths = ha["num_tokens"].values.reshape(-1, 1)

    from sklearn.linear_model import LinearRegression

    X_resid = np.zeros_like(X)
    for j in range(X.shape[1]):
        lr = LinearRegression().fit(lengths, X[:, j])
        X_resid[:, j] = X[:, j] - lr.predict(lengths)

    scaler = StandardScaler()
    X_resid_scaled = scaler.fit_transform(X_resid)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X_resid_scaled, y):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_resid_scaled[tr], y[tr])
        p = clf.predict_proba(X_resid_scaled[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))

    print(
        f"  Random Forest (length-residualized): AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
    )


def per_side_task_divergence(div_df, out):
    print("\n" + "=" * 70)
    print("  TEST 4: Divergence by Side Task Type")
    print("=" * 70)

    attack = div_df[div_df["is_attack"] == True]
    benign = div_df[div_df["is_attack"] == False]

    if "side_task" not in attack.columns:
        print("  No side_task column, skipping")
        return

    side_tasks = attack["side_task"].dropna().unique()
    benign_mmd = benign["mmd_multivariate"].dropna()

    rows = []
    for st in sorted(side_tasks):
        st_vals = attack[attack["side_task"] == st]["mmd_multivariate"].dropna()
        if len(st_vals) < 3:
            continue
        ratio = st_vals.mean() / (benign_mmd.mean() + 1e-10)
        u, p = mannwhitneyu(st_vals, benign_mmd, alternative="greater")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(
            f"  {st:25s}  MMD={st_vals.mean():.4f}  ratio={ratio:.2f}x  p={p:.4f} {sig}  (n={len(st_vals)})"
        )
        rows.append(
            {
                "side_task": st,
                "mmd_mean": st_vals.mean(),
                "ratio": ratio,
                "p_value": p,
                "n": len(st_vals),
            }
        )

    pd.DataFrame(rows).to_csv(out / "per_side_task_divergence.csv", index=False)


def scaling_comparison(df_8b, raw_8b, df_32b, raw_32b, out):
    print("\n" + "=" * 70)
    print("  TEST 5: Scaling Comparison — 8B vs 32B")
    print("=" * 70)

    rows = []

    for label, df, raw in [("8B", df_8b, raw_8b), ("32B", df_32b, raw_32b)]:
        if df is None:
            continue

        ha = df[df["mode"].isin(["honest", "attack"])]
        honest = ha[ha["mode"] == "honest"]
        attack = ha[ha["mode"] == "attack"]

        n_honest = len(honest)
        n_attack = len(attack)

        feature_cols = [
            c
            for c in ha.columns
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
            and ha[c].dtype in [np.float64, np.int64]
        ]

        best_d = 0
        best_feat = ""
        for col in feature_cols:
            h = honest[col].dropna()
            a = attack[col].dropna()
            if len(h) > 2 and len(a) > 2:
                pooled = np.sqrt((h.std() ** 2 + a.std() ** 2) / 2)
                d = abs(h.mean() - a.mean()) / (pooled + 1e-10)
                if d > best_d:
                    best_d = d
                    best_feat = col

        X = ha[feature_cols].fillna(0).values
        y_arr = (ha["mode"] == "attack").astype(int).values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        n_splits = min(5, min(y_arr.sum(), len(y_arr) - y_arr.sum()))
        if n_splits >= 2:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            aucs = []
            for tr, te in skf.split(X_s, y_arr):
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_s[tr], y_arr[tr])
                aucs.append(roc_auc_score(y_arr[te], clf.predict_proba(X_s[te])[:, 1]))
            rf_auroc = np.mean(aucs)
        else:
            rf_auroc = 0.5

        mmd_ratio = 0.0
        if raw is not None:
            div = compute_divergences_from_raw(raw)
            att_mmd = div[div["is_attack"] == True]["mmd_multivariate"].mean()
            ben_mmd = div[div["is_attack"] == False]["mmd_multivariate"].mean()
            mmd_ratio = att_mmd / (ben_mmd + 1e-10)

        row = {
            "Model": label,
            "n_honest": n_honest,
            "n_attack": n_attack,
            "Best Feature": best_feat,
            "Best Cohen's d": f"{best_d:.3f}",
            "RF AUROC (features)": f"{rf_auroc:.3f}",
            "MMD ratio": f"{mmd_ratio:.2f}x",
        }
        rows.append(row)
        print(f"\n  {label}:")
        for k, v in row.items():
            print(f"    {k}: {v}")

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out / "scaling_comparison.csv", index=False)

    if len(rows) == 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "Scaling Comparison: 8B vs 32B Attacker", fontsize=13, fontweight="bold"
        )

        labels = [r["Model"] for r in rows]
        x = np.arange(len(labels))

        d_vals = [float(r["Best Cohen's d"]) for r in rows]
        axes[0].bar(x, d_vals, color=[BLUE, RED], width=0.5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("Cohen's d")
        axes[0].set_title("Best Feature Effect Size")
        for i, v in enumerate(d_vals):
            axes[0].text(
                i, v + 0.05, f"{v:.2f}", ha="center", fontsize=11, color="#c9d1d9"
            )

        auroc_vals = [float(r["RF AUROC (features)"]) for r in rows]
        axes[1].bar(x, auroc_vals, color=[BLUE, RED], width=0.5)
        axes[1].axhline(
            y=0.5, color="#8b949e", linestyle="--", alpha=0.5, label="Chance"
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("AUROC")
        axes[1].set_title("Logprob Classifier AUROC")
        axes[1].set_ylim(0.4, 1.0)
        for i, v in enumerate(auroc_vals):
            axes[1].text(
                i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11, color="#c9d1d9"
            )

        mmd_vals = [float(r["MMD ratio"].replace("x", "")) for r in rows]
        axes[2].bar(x, mmd_vals, color=[BLUE, RED], width=0.5)
        axes[2].axhline(
            y=1.0, color="#8b949e", linestyle="--", alpha=0.5, label="No signal"
        )
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels)
        axes[2].set_ylabel("MMD Ratio")
        axes[2].set_title(" MMD Divergence Ratio")
        for i, v in enumerate(mmd_vals):
            axes[2].text(
                i, v + 0.1, f"{v:.2f}x", ha="center", fontsize=11, color="#c9d1d9"
            )

        plt.tight_layout()
        plt.savefig(out / "11_scaling_comparison.png", bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: 11_scaling_comparison.png")


def bootstrap_divergence(div_df, out, n_bootstrap=1000):
    print("\n" + "=" * 70)
    print("  TEST 6: Bootstrap MMD Ratio")
    print("=" * 70)

    attack_mmd = div_df[div_df["is_attack"] == True]["mmd_multivariate"].dropna().values
    benign_mmd = (
        div_df[div_df["is_attack"] == False]["mmd_multivariate"].dropna().values
    )

    if len(attack_mmd) < 5 or len(benign_mmd) < 5:
        print("  Insufficient data")
        return

    np.random.seed(42)
    ratios = []
    for _ in range(n_bootstrap):
        a_sample = np.random.choice(attack_mmd, size=len(attack_mmd), replace=True)
        b_sample = np.random.choice(benign_mmd, size=len(benign_mmd), replace=True)
        ratios.append(a_sample.mean() / (b_sample.mean() + 1e-10))

    ratios = np.array(ratios)
    ci_low = np.percentile(ratios, 2.5)
    ci_high = np.percentile(ratios, 97.5)
    median = np.median(ratios)

    print(f"  MMD ratio: {median:.2f}x  (95% CI: [{ci_low:.2f}x, {ci_high:.2f}x])")
    print(f"  CI excludes 1.0: {'YES ' if ci_low > 1.0 else 'NO '}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ratios, bins=50, color=PURPLE, alpha=0.7, edgecolor="none")
    ax.axvline(
        1.0, color="#8b949e", linestyle="--", linewidth=2, label="No signal (ratio=1)"
    )
    ax.axvline(
        ci_low,
        color=YELLOW,
        linestyle="--",
        linewidth=1.5,
        label=f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]",
    )
    ax.axvline(ci_high, color=YELLOW, linestyle="--", linewidth=1.5)
    ax.axvline(median, color=RED, linewidth=2, label=f"Median: {median:.2f}x")
    ax.set_xlabel("MMD Ratio (Attack / Benign)")
    ax.set_ylabel("Count")
    ax.set_title("Bootstrap Distribution of MMD Ratio (n=1000)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "12_bootstrap_mmd_ratio.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: 12_bootstrap_mmd_ratio.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-32b", type=str, default="~/logprob-data-70b")
    parser.add_argument("--data-8b", type=str, default="~/logprob-data")
    parser.add_argument("--output", type=str, default="~/logprob-final")
    args = parser.parse_args()

    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print("Loading 32B data...")
    df_32b, raw_32b = load_run(args.data_32b)
    print(
        f"  32B: {df_32b.shape[0]} rows, modes: {dict(df_32b['mode'].value_counts())}"
    )

    df_8b, raw_8b = None, None
    p8 = Path(args.data_8b).expanduser()
    if (p8 / "features.csv").exists():
        print("Loading 8B data...")
        df_8b, raw_8b = load_run(args.data_8b)
        print(
            f"  8B: {df_8b.shape[0]} rows, modes: {dict(df_8b['mode'].value_counts())}"
        )
    else:
        print("  8B data not found, skipping comparison")

    print("\nComputing divergences...")
    div_32b = compute_divergences_from_raw(raw_32b)
    print(f"  {len(div_32b)} pairwise comparisons")
    div_32b.to_csv(out / "divergences_32b.csv", index=False)

    test_divergence_significance(div_32b, out)
    oc_results = divergence_classifier(div_32b, out)
    length_confound_check(df_32b, out)
    per_side_task_divergence(div_32b, out)
    scaling_comparison(df_8b, raw_8b, df_32b, raw_32b, out)
    bootstrap_divergence(div_32b, out)

    print(f"\n  All outputs saved to: {out}")
    for f in sorted(out.glob("*")):
        print(f"    {f.name}")
    print()


if __name__ == "__main__":
    main()
