"""
Weekly Research Sprint: Causal-Invariant Stable/Spurious Decomposition (CISD)
for multi-site clinical distribution shift.

Runs the full pipeline:
  1. Gradient-correctness check (aborts if it fails).
  2. Load + preprocess the real 4-site UCI Heart Disease dataset.
  3. Leave-One-Site-Out (LOSO) domain-generalization evaluation of
     ERM vs IRM vs CISD (proposed), across 5 random seeds per fold.
  4. Missing-modality robustness stress test on the held-out site.
  5. Ablation study on CISD's three mechanisms (adversarial GRL term,
     orthogonality penalty, auxiliary spurious-environment head).
  6. Save all real predictions/metrics/figures to output/.
"""
import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import data as D
from src.metrics import all_metrics
from src.models import (init_params, erm_grad, irm_grad, cisd_grad,
                         predict_erm, predict_cisd, Adam)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

H = 16          # hidden width
H1 = 10         # z_stable width
H2 = H - H1     # z_spurious width
N_ITERS = 2500
LR = 0.02
SEEDS = [0, 1, 2, 3, 4]
LAMBDA_IRM = 5.0
LAMBDA_ADV = 1.0
LAMBDA_ENV = 0.5
LAMBDA_ORTH = 0.1


def run_grad_check():
    res = subprocess.run([sys.executable, str(ROOT / "tests" / "test_gradients.py")],
                          capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError("Gradient check failed -- aborting experiment.")


def train_erm(Xtr, ytr, seed, n_in):
    p = init_params(n_in, H, n_env=1, seed=seed)
    opt = Adam(p, lr=LR)
    for _ in range(N_ITERS):
        _, _, grads = erm_grad(Xtr, ytr, p)
        opt.step(p, grads)
    return p


def train_irm(Xtr, ytr, env_ids_tr, seed, n_in):
    p = init_params(n_in, H, n_env=1, seed=seed)
    opt = Adam(p, lr=LR)
    env_mask_list = [env_ids_tr == e for e in np.unique(env_ids_tr)]
    for _ in range(N_ITERS):
        _, _, _, grads = irm_grad(Xtr, ytr, p, env_mask_list, lambda_irm=LAMBDA_IRM)
        opt.step(p, grads)
    return p


def train_cisd(Xtr, ytr, env_ids_tr, seed, n_in, lambda_adv=LAMBDA_ADV,
               lambda_env=LAMBDA_ENV, lambda_orth=LAMBDA_ORTH):
    n_env = len(np.unique(env_ids_tr))
    env_onehot = np.eye(n_env)[env_ids_tr]
    p = init_params(n_in, H, n_env=n_env, seed=seed, split=(H1, H2))
    opt = Adam(p, lr=LR)
    for _ in range(N_ITERS):
        _, _, grads, _ = cisd_grad(Xtr, ytr, env_onehot, p, H1,
                                    lambda_adv=lambda_adv, lambda_env=lambda_env,
                                    lambda_orth=lambda_orth)
        opt.step(p, grads)
    return p


def mask_modality(X, cols_to_zero):
    """Simulate a missing modality at test time: replace with the
    (already-standardized) training mean, i.e. 0.0 in standardized space."""
    Xm = X.copy()
    Xm[:, cols_to_zero] = 0.0
    return Xm


def main():
    print("=" * 70)
    print("STEP 0: Verifying hand-derived gradients against numerical gradients")
    print("=" * 70)
    run_grad_check()

    print("\n" + "=" * 70)
    print("STEP 1: Loading real 4-site UCI Heart Disease data")
    print("=" * 70)
    df = D.load_all(RAW_DIR)
    summary = D.site_label_summary(df)
    print(summary)
    summary.to_csv(OUT_DIR / "site_summary.csv")
    mod_idx = D.modality_column_indices()
    n_in = len(D.KEEP)
    sites = D.SITES

    all_rows = []          # LOSO main-method results
    ablation_rows = []     # CISD ablation results
    missing_mod_rows = []  # missing-modality stress test results
    pred_records = []      # raw predictions for output/predictions.csv

    print("\n" + "=" * 70)
    print("STEP 2: Leave-One-Site-Out domain generalization (ERM vs IRM vs CISD)")
    print("=" * 70)
    for test_site in sites:
        train_sites = [s for s in sites if s != test_site]
        train_df = df[df["site"].isin(train_sites)].reset_index(drop=True)
        test_df = df[df["site"] == test_site].reset_index(drop=True)

        prep = D.fit_preprocessor(train_df)
        Xtr = D.apply_preprocessor(train_df, prep)
        ytr = train_df["y"].values.astype(float)
        env_ids_tr = train_df["site"].map({s: i for i, s in enumerate(train_sites)}).values

        Xte = D.apply_preprocessor(test_df, prep)
        yte = test_df["y"].values.astype(float)

        print(f"\n-- Held-out site: {test_site} (n={len(yte)}, "
              f"prevalence={yte.mean():.2f}) | train sites={train_sites} "
              f"(n={len(ytr)}, prevalence={ytr.mean():.2f})")

        for seed in SEEDS:
            # ERM
            p_erm = train_erm(Xtr, ytr, seed, n_in)
            pred_erm = predict_erm(Xte, p_erm)
            m = all_metrics(yte, pred_erm)
            m.update(method="ERM", test_site=test_site, seed=seed)
            all_rows.append(m)

            # IRM
            p_irm = train_irm(Xtr, ytr, env_ids_tr, seed, n_in)
            pred_irm = predict_erm(Xte, p_irm)
            m = all_metrics(yte, pred_irm)
            m.update(method="IRM", test_site=test_site, seed=seed)
            all_rows.append(m)

            # CISD (proposed, full)
            p_cisd = train_cisd(Xtr, ytr, env_ids_tr, seed, n_in)
            pred_cisd = predict_cisd(Xte, p_cisd, H1)
            m = all_metrics(yte, pred_cisd)
            m.update(method="CISD (ours)", test_site=test_site, seed=seed)
            all_rows.append(m)

            if seed == SEEDS[0]:
                for i in range(len(yte)):
                    pred_records.append({
                        "test_site": test_site, "index": i, "y_true": int(yte[i]),
                        "pred_ERM": float(pred_erm[i]), "pred_IRM": float(pred_irm[i]),
                        "pred_CISD": float(pred_cisd[i]),
                    })

            # ---- missing-modality stress test (ERM vs CISD) on this seed ----
            for mod_name, cols in mod_idx.items():
                Xte_missing = mask_modality(Xte, cols)
                pe = predict_erm(Xte_missing, p_erm)
                pc = predict_cisd(Xte_missing, p_cisd, H1)
                me = all_metrics(yte, pe)
                mc = all_metrics(yte, pc)
                missing_mod_rows.append(dict(method="ERM", test_site=test_site,
                                              seed=seed, missing_modality=mod_name,
                                              auroc=me["auroc"], brier=me["brier"]))
                missing_mod_rows.append(dict(method="CISD (ours)", test_site=test_site,
                                              seed=seed, missing_modality=mod_name,
                                              auroc=mc["auroc"], brier=mc["brier"]))

            # ---- ablations of CISD ----
            ablation_specs = {
                "CISD-full": dict(lambda_adv=LAMBDA_ADV, lambda_env=LAMBDA_ENV, lambda_orth=LAMBDA_ORTH),
                "CISD-no_adv": dict(lambda_adv=0.0, lambda_env=LAMBDA_ENV, lambda_orth=LAMBDA_ORTH),
                "CISD-no_orth": dict(lambda_adv=LAMBDA_ADV, lambda_env=LAMBDA_ENV, lambda_orth=0.0),
                "CISD-no_env_aux": dict(lambda_adv=LAMBDA_ADV, lambda_env=0.0, lambda_orth=LAMBDA_ORTH),
            }
            for name, kw in ablation_specs.items():
                p_ab = train_cisd(Xtr, ytr, env_ids_tr, seed, n_in, **kw)
                pred_ab = predict_cisd(Xte, p_ab, H1)
                m = all_metrics(yte, pred_ab)
                m.update(variant=name, test_site=test_site, seed=seed)
                ablation_rows.append(m)

        print(f"   done: {test_site}")

    results_df = pd.DataFrame(all_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    missing_df = pd.DataFrame(missing_mod_rows)
    pred_df = pd.DataFrame(pred_records)

    results_df.to_csv(OUT_DIR / "loso_results_raw.csv", index=False)
    ablation_df.to_csv(OUT_DIR / "ablation_results_raw.csv", index=False)
    missing_df.to_csv(OUT_DIR / "missing_modality_results_raw.csv", index=False)
    pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)

    # ---- aggregate summaries ----
    agg = results_df.groupby(["method", "test_site"])[
        ["auroc", "auprc", "f1", "brier", "ece"]].agg(["mean", "std"])
    agg.to_csv(OUT_DIR / "loso_summary_by_site.csv")

    overall = results_df.groupby("method")[["auroc", "auprc", "f1", "brier", "ece"]].agg(["mean", "std"])
    overall.to_csv(OUT_DIR / "loso_summary_overall.csv")
    print("\nOverall LOSO summary (mean over all sites & seeds):")
    print(overall)

    ablation_overall = ablation_df.groupby("variant")[["auroc", "auprc", "f1", "brier", "ece"]].agg(["mean", "std"])
    ablation_overall.to_csv(OUT_DIR / "ablation_summary.csv")
    print("\nAblation summary:")
    print(ablation_overall)

    missing_overall = missing_df.groupby(["method", "missing_modality"])[["auroc", "brier"]].agg(["mean", "std"])
    missing_overall.to_csv(OUT_DIR / "missing_modality_summary.csv")
    print("\nMissing-modality summary:")
    print(missing_overall)

    # combine into metrics.json
    def df_to_nested(d):
        return json.loads(d.reset_index().to_json(orient="records"))

    metrics_json = {
        "loso_overall": df_to_nested(overall),
        "loso_by_site": df_to_nested(agg),
        "ablation_overall": df_to_nested(ablation_overall),
        "missing_modality_overall": df_to_nested(missing_overall),
        "hyperparameters": {
            "hidden_width": H, "stable_width": H1, "spurious_width": H2,
            "n_iters": N_ITERS, "lr": LR, "seeds": SEEDS,
            "lambda_irm": LAMBDA_IRM, "lambda_adv": LAMBDA_ADV,
            "lambda_env": LAMBDA_ENV, "lambda_orth": LAMBDA_ORTH,
        },
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2, default=str)

    # ---- figures ----
    make_figures(results_df, ablation_df, missing_df)

    print("\nAll real outputs written to:", OUT_DIR)


def make_figures(results_df, ablation_df, missing_df):
    methods = ["ERM", "IRM", "CISD (ours)"]
    colors = {"ERM": "#888888", "IRM": "#4C72B0", "CISD (ours)": "#C44E52"}

    # Fig 1: AUROC per held-out site, per method
    fig, ax = plt.subplots(figsize=(8, 5))
    sites = sorted(results_df["test_site"].unique())
    width = 0.25
    x = np.arange(len(sites))
    for i, m in enumerate(methods):
        means, stds = [], []
        for s in sites:
            sub = results_df[(results_df.method == m) & (results_df.test_site == s)]["auroc"]
            means.append(sub.mean())
            stds.append(sub.std())
        ax.bar(x + i * width, means, width, yerr=stds, label=m, color=colors[m], capsize=3)
    ax.set_xticks(x + width)
    ax.set_xticklabels(sites)
    ax.set_ylabel("AUROC (held-out site)")
    ax.set_title("Leave-One-Site-Out AUROC: ERM vs IRM vs CISD (proposed)")
    ax.legend()
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "loso_auroc_by_site.png", dpi=150)
    plt.close()

    # Fig 2: ECE per method (calibration under shift)
    fig, ax = plt.subplots(figsize=(6, 5))
    means = [results_df[results_df.method == m]["ece"].mean() for m in methods]
    stds = [results_df[results_df.method == m]["ece"].std() for m in methods]
    ax.bar(methods, means, yerr=stds, color=[colors[m] for m in methods], capsize=4)
    ax.set_ylabel("Expected Calibration Error (lower is better)")
    ax.set_title("Calibration under site shift (LOSO, averaged)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ece_comparison.png", dpi=150)
    plt.close()

    # Fig 3: missing-modality degradation
    fig, ax = plt.subplots(figsize=(8, 5))
    mods = sorted(missing_df["missing_modality"].unique())
    x = np.arange(len(mods))
    width = 0.35
    for i, m in enumerate(["ERM", "CISD (ours)"]):
        means, stds = [], []
        for mod in mods:
            sub = missing_df[(missing_df.method == m) & (missing_df.missing_modality == mod)]["auroc"]
            means.append(sub.mean())
            stds.append(sub.std())
        ax.bar(x + i * width, means, width, yerr=stds, label=m, color=colors[m], capsize=3)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(mods)
    ax.set_ylabel("AUROC with modality zeroed at test time")
    ax.set_title("Missing-modality robustness (held-out sites, all seeds)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "missing_modality_robustness.png", dpi=150)
    plt.close()

    # Fig 4: ablation
    fig, ax = plt.subplots(figsize=(8, 5))
    variants = ["CISD-full", "CISD-no_adv", "CISD-no_orth", "CISD-no_env_aux"]
    means = [ablation_df[ablation_df.variant == v]["auroc"].mean() for v in variants]
    stds = [ablation_df[ablation_df.variant == v]["auroc"].std() for v in variants]
    bar_colors = ["#C44E52", "#D98F91", "#D98F91", "#D98F91"]
    ax.bar(variants, means, yerr=stds, color=bar_colors, capsize=4)
    ax.set_ylabel("AUROC (LOSO, averaged over all sites/seeds)")
    ax.set_title("Ablation of CISD mechanisms")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ablation_study.png", dpi=150)
    plt.close()

    print("Figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
