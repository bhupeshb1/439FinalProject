"""
End-to-end experiment runner.

Produces every table and figure referenced in the paper, saving CSVs to
results/tables/ and PDFs to results/figures/.

Usage:
    python -m src.run_experiments \
        --data-dir data/lobster \
        --ticker AAPL --date 2012-06-21 \
        --horizons 10 50 100 \
        --k-values 2 3 4 5 6 8 10 \
        --k-star 4

If --use-synthetic is passed, real LOBSTER files are not needed: a synthetic
dataset is generated for pipeline-validation only.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.cluster import (
    ClusteringArtifacts,
    assign_regimes,
    fit_clustering,
    k_sweep,
    regime_centroids_in_original_units,
    residence_time,
    transition_matrix,
)
from src.evaluate import (
    compute_metrics,
    confusion_df,
    per_regime_metrics,
)
from src.features import FEATURE_COLS, compute_features
from src.label import compute_labels, label_distribution
from src.load import filter_session, load_lobster_day, synthesize_lobster_day
from src.models import (
    train_gbm_blind,
    train_gbm_cond,
    train_gbm_feat,
    train_lr_blind,
)
from src import plots

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/lobster")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--date", default="2012-06-21")
    p.add_argument("--horizons", nargs="+", type=int, default=[10, 50, 100])
    p.add_argument("--horizon-main", type=int, default=50,
                   help="The horizon used for the main results table.")
    p.add_argument("--k-values", nargs="+", type=int,
                   default=[2, 3, 4, 5, 6, 8, 10])
    p.add_argument("--k-star", type=int, default=4)
    p.add_argument("--theta", type=float, default=0.005,
                   help="Label threshold (USD). Half a tick = 0.005.")
    p.add_argument("--window", type=int, default=100,
                   help="Rolling window for dynamic features.")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-synthetic", action="store_true",
                   help="Use synthetic data instead of real LOBSTER files.")
    p.add_argument("--n-synthetic", type=int, default=200_000)
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"[1/8] Loading data ...")
    if args.use_synthetic:
        df = synthesize_lobster_day(n_events=args.n_synthetic, seed=args.seed)
    else:
        df = load_lobster_day(args.data_dir, args.ticker, args.date)
    df = filter_session(df, trim_minutes=5)
    print(f"      {len(df):,} events after session filtering")

    # ------------------------------------------------------------------
    # 2. Engineer features and labels
    # ------------------------------------------------------------------
    print(f"[2/8] Computing features (window={args.window}) ...")
    X_full = compute_features(df, window=args.window)

    label_dict = {}
    for h in args.horizons:
        label_dict[h] = compute_labels(df, horizon=h, theta=args.theta)

    # Drop rows with NaN features OR NaN labels at the main horizon
    main_y = label_dict[args.horizon_main]
    valid = ~(X_full.isna().any(axis=1) | main_y.isna())
    for y_h in label_dict.values():
        valid = valid & ~y_h.isna()
    X = X_full[valid]
    y_dict = {h: label_dict[h][valid].values.astype(int) for h in args.horizons}
    print(f"      {len(X):,} valid snapshots after dropping NaNs")

    # Save label distribution per horizon
    label_dist_rows = []
    for h, y_h in y_dict.items():
        s = pd.Series(y_h)
        label_dist_rows.append({
            "horizon": h,
            "down": float((s == -1).mean()),
            "stationary": float((s == 0).mean()),
            "up": float((s == 1).mean()),
        })
    pd.DataFrame(label_dist_rows).to_csv(tables_dir / "label_distribution.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Train/test split (temporal, no shuffle)
    # ------------------------------------------------------------------
    print(f"[3/8] Splitting data ...")
    n = len(X)
    split = int(n * args.train_frac)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    print(f"      Train: {len(X_tr):,}   Test: {len(X_te):,}")

    # ------------------------------------------------------------------
    # 4. K-Means: K-sweep + final fit at K*
    # ------------------------------------------------------------------
    print(f"[4/8] K-sweep over {args.k_values} ...")
    sweep = k_sweep(X_tr, k_values=args.k_values, seed=args.seed)
    sweep.to_csv(tables_dir / "k_sweep.csv", index=False)
    print(f"      SSE/silhouette saved to k_sweep.csv")

    plots.plot_k_sweep(sweep, k_star=args.k_star,
                       save_path=figures_dir / "fig_k_sweep.pdf")

    print(f"[5/8] Fitting K-Means with K*={args.k_star} ...")
    artifacts = fit_clustering(X_tr, k=args.k_star, seed=args.seed)
    regimes_tr = assign_regimes(artifacts, X_tr)
    regimes_te = assign_regimes(artifacts, X_te)

    # Regime characterization tables
    centroids = regime_centroids_in_original_units(artifacts)
    centroids.to_csv(tables_dir / "regime_centroids.csv")
    residence_time(regimes_tr).to_csv(tables_dir / "residence_time_train.csv", index=False)
    transition_matrix(regimes_tr, args.k_star).to_csv(tables_dir / "transition_train.csv")

    plots.plot_pca_regimes(X_tr, regimes_tr, figures_dir / "fig_pca_regimes.pdf")
    plots.plot_transition_heatmap(
        transition_matrix(regimes_tr, args.k_star),
        figures_dir / "fig_transition.pdf",
    )

    # ------------------------------------------------------------------
    # 6. Train all four models at the main horizon
    # ------------------------------------------------------------------
    print(f"[6/8] Training models at horizon={args.horizon_main} ...")
    y_tr_main = y_dict[args.horizon_main][:split]
    y_te_main = y_dict[args.horizon_main][split:]

    m_lr = train_lr_blind(X_tr, y_tr_main)
    m_gbm = train_gbm_blind(X_tr, y_tr_main)
    m_feat = train_gbm_feat(X_tr, y_tr_main, regimes_tr, k=args.k_star)
    m_cond = train_gbm_cond(X_tr, y_tr_main, regimes_tr, k=args.k_star)

    # Predictions at main horizon
    preds = {
        "LR-blind": (m_lr.predict(X_te), m_lr.predict_proba(X_te)),
        "GBM-blind": (m_gbm.predict(X_te), m_gbm.predict_proba(X_te)),
        "GBM-feat": (
            m_feat.predict_regimes(X_te, regimes_te),
            m_feat.predict_proba_regimes(X_te, regimes_te),
        ),
        "GBM-cond": (
            m_cond.predict(X_te, regimes_te),
            m_cond.predict_proba(X_te, regimes_te),
        ),
    }

    # ------------------------------------------------------------------
    # 7. Evaluate at main horizon + at all horizons
    # ------------------------------------------------------------------
    print(f"[7/8] Evaluating ...")
    main_rows = []
    confusion_mats = {}
    for name, (yp, yp_proba) in preds.items():
        m = compute_metrics(y_te_main, yp, yp_proba)
        m["model"] = name
        main_rows.append(m)
        confusion_mats[name] = confusion_df(y_te_main, yp)
    main_results = pd.DataFrame(main_rows)[
        ["model", "n", "macro_f1", "balanced_acc", "auc_up", "auc_down",
         "f1_down", "f1_stat", "f1_up"]
    ]
    main_results.to_csv(tables_dir / "main_results.csv", index=False)
    print(f"\n      Main results (horizon={args.horizon_main}):")
    print(main_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save confusion matrices
    for name, cm in confusion_mats.items():
        cm.to_csv(tables_dir / f"confusion_{name.replace('-', '_').lower()}.csv")
    plots.plot_confusion_matrices(confusion_mats,
                                  figures_dir / "fig_confusion_matrices.pdf")

    # Feature importances
    importances = {
        "GBM-blind": m_gbm.feature_importance,
        "GBM-feat": m_feat.feature_importance,
        "GBM-cond": m_cond.aggregate_feature_importance(),
    }
    for name, imp in importances.items():
        imp.to_csv(tables_dir / f"feat_importance_{name.replace('-', '_').lower()}.csv")
    plots.plot_feature_importance(importances,
                                  figures_dir / "fig_feature_importance.pdf")

    # Per-regime breakdown of GBM-blind vs GBM-cond
    per_blind = per_regime_metrics(y_te_main, preds["GBM-blind"][0], regimes_te)
    per_cond = per_regime_metrics(y_te_main, preds["GBM-cond"][0], regimes_te)
    per_combined = per_blind[["regime", "n_test"]].merge(
        per_blind[["regime", "macro_f1"]].rename(columns={"macro_f1": "macro_f1_blind"}),
        on="regime"
    ).merge(
        per_cond[["regime", "macro_f1"]].rename(columns={"macro_f1": "macro_f1_cond"}),
        on="regime"
    )
    per_combined["delta"] = per_combined["macro_f1_cond"] - per_combined["macro_f1_blind"]
    per_combined.to_csv(tables_dir / "per_regime_results.csv", index=False)
    plots.plot_per_regime_f1(per_combined, figures_dir / "fig_per_regime_f1.pdf")

    # ------------------------------------------------------------------
    # 8. Horizon sweep
    # ------------------------------------------------------------------
    print(f"[8/8] Horizon sweep ...")
    horizon_rows = []
    for h in args.horizons:
        y_tr_h = y_dict[h][:split]
        y_te_h = y_dict[h][split:]

        m_lr_h = train_lr_blind(X_tr, y_tr_h)
        m_gbm_h = train_gbm_blind(X_tr, y_tr_h)
        m_feat_h = train_gbm_feat(X_tr, y_tr_h, regimes_tr, k=args.k_star)
        m_cond_h = train_gbm_cond(X_tr, y_tr_h, regimes_tr, k=args.k_star)

        for name, (yp, _) in {
            "LR-blind": (m_lr_h.predict(X_te), None),
            "GBM-blind": (m_gbm_h.predict(X_te), None),
            "GBM-feat": (m_feat_h.predict_regimes(X_te, regimes_te), None),
            "GBM-cond": (m_cond_h.predict(X_te, regimes_te), None),
        }.items():
            metrics = compute_metrics(y_te_h, yp)
            horizon_rows.append({"horizon": h, "model": name, **metrics})

    horizon_results = pd.DataFrame(horizon_rows)[
        ["horizon", "model", "macro_f1", "balanced_acc", "f1_down", "f1_stat", "f1_up"]
    ]
    horizon_results.to_csv(tables_dir / "horizon_sweep.csv", index=False)
    plots.plot_horizon_comparison(horizon_results, figures_dir / "fig_horizon_sweep.pdf")

    # ------------------------------------------------------------------
    # K-ablation: rerun GBM-cond at varying K
    # ------------------------------------------------------------------
    print(f"      K-ablation at horizon={args.horizon_main} ...")
    k_ablation_rows = []
    for k in args.k_values:
        art_k = fit_clustering(X_tr, k=k, seed=args.seed)
        rt = assign_regimes(art_k, X_tr)
        re = assign_regimes(art_k, X_te)
        m_k = train_gbm_cond(X_tr, y_tr_main, rt, k=k)
        yp = m_k.predict(X_te, re)
        mres = compute_metrics(y_te_main, yp)
        rt_summary = residence_time(rt)
        mean_residence = float(rt_summary["mean"].mean()) if len(rt_summary) else float("nan")
        k_ablation_rows.append({
            "k": k, "macro_f1": mres["macro_f1"],
            "balanced_acc": mres["balanced_acc"],
            "mean_residence_time": mean_residence,
        })
    pd.DataFrame(k_ablation_rows).to_csv(tables_dir / "k_ablation.csv", index=False)

    # ------------------------------------------------------------------
    # Feature subset ablation
    # ------------------------------------------------------------------
    print(f"      Feature subset ablation ...")
    feat_subsets = {
        "price_only": ["spread", "micro_disp", "ret_w"],
        "price_plus_l1": ["spread", "micro_disp", "ret_w", "imb_l1"],
        "price_plus_l1_l5": ["spread", "micro_disp", "ret_w", "imb_l1", "imb_l5"],
        "all": FEATURE_COLS,
    }
    feat_ablation_rows = []
    for subset_name, cols in feat_subsets.items():
        cols_keep = [c for c in cols if c in X_tr.columns]
        Xtr_s, Xte_s = X_tr[cols_keep], X_te[cols_keep]
        art_s = fit_clustering(Xtr_s, k=args.k_star, seed=args.seed)
        rt_s, re_s = assign_regimes(art_s, Xtr_s), assign_regimes(art_s, Xte_s)

        m_blind_s = train_gbm_blind(Xtr_s, y_tr_main)
        m_cond_s = train_gbm_cond(Xtr_s, y_tr_main, rt_s, k=args.k_star)

        f1_blind = compute_metrics(y_te_main, m_blind_s.predict(Xte_s))["macro_f1"]
        f1_cond = compute_metrics(y_te_main, m_cond_s.predict(Xte_s, re_s))["macro_f1"]
        feat_ablation_rows.append({
            "subset": subset_name,
            "n_features": len(cols_keep),
            "gbm_blind_f1": f1_blind,
            "gbm_cond_f1": f1_cond,
        })
    pd.DataFrame(feat_ablation_rows).to_csv(tables_dir / "feature_ablation.csv", index=False)

    # ------------------------------------------------------------------
    # Save run config
    # ------------------------------------------------------------------
    config = vars(args).copy()
    config["n_train"] = len(X_tr)
    config["n_test"] = len(X_te)
    config["n_features"] = X.shape[1]
    with open(results_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Tables in {tables_dir}/, figures in {figures_dir}/")


if __name__ == "__main__":
    main()
