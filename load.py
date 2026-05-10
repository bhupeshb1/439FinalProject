"""
K-Means clustering for microstructure regime discovery.

Fits K-Means on standardized features. Provides:
- K-sweep with SSE (Elbow) and Silhouette for model selection
- Frozen clustering model that can be applied to test data
- Regime characterization (centroid stats, residence time, transition matrix)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringArtifacts:
    """Holds everything needed to apply a fitted clustering pipeline to new data."""
    scaler: StandardScaler
    kmeans: KMeans
    feature_names: list[str]
    k: int


def fit_clustering(
    X_train: pd.DataFrame,
    k: int,
    seed: int = 42,
    n_init: int = 10,
) -> ClusteringArtifacts:
    """Fit StandardScaler + KMeans on the training feature matrix."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    kmeans.fit(X_scaled)
    return ClusteringArtifacts(
        scaler=scaler, kmeans=kmeans, feature_names=list(X_train.columns), k=k
    )


def assign_regimes(artifacts: ClusteringArtifacts, X: pd.DataFrame) -> np.ndarray:
    """Apply the fitted scaler + kmeans to a new feature matrix and return cluster ids."""
    X_aligned = X[artifacts.feature_names]
    X_scaled = artifacts.scaler.transform(X_aligned.values)
    return artifacts.kmeans.predict(X_scaled)


def k_sweep(
    X_train: pd.DataFrame,
    k_values: list[int],
    seed: int = 42,
    silhouette_subsample: int = 50_000,
) -> pd.DataFrame:
    """Sweep K and compute SSE + silhouette for each. Returns a tidy dataframe."""
    rng = np.random.default_rng(seed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)

    # Subsample for silhouette (it is O(n^2) and prohibitive on full data)
    if len(X_scaled) > silhouette_subsample:
        idx = rng.choice(len(X_scaled), silhouette_subsample, replace=False)
        X_sil = X_scaled[idx]
    else:
        X_sil = X_scaled

    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        sse = float(km.inertia_)
        # Silhouette needs at least 2 clusters with >1 member each
        if k >= 2:
            sil_labels = km.predict(X_sil)
            sil = float(silhouette_score(X_sil, sil_labels)) if len(np.unique(sil_labels)) > 1 else np.nan
        else:
            sil = np.nan
        rows.append({"k": k, "sse": sse, "silhouette": sil})
    return pd.DataFrame(rows)


def regime_centroids_in_original_units(
    artifacts: ClusteringArtifacts,
) -> pd.DataFrame:
    """Inverse-transform K-Means centroids back to original feature units, for
    interpretation. Returns a (k, n_features) dataframe."""
    centroids_scaled = artifacts.kmeans.cluster_centers_
    centroids = artifacts.scaler.inverse_transform(centroids_scaled)
    return pd.DataFrame(
        centroids, columns=artifacts.feature_names,
        index=[f"regime_{i}" for i in range(artifacts.k)]
    )


def residence_time(regimes: np.ndarray) -> pd.DataFrame:
    """For each regime, compute the empirical distribution of how many consecutive
    snapshots the market stays in that regime before transitioning out.

    Returns a dataframe with columns [regime, mean, median, std, n_visits].
    """
    if len(regimes) == 0:
        return pd.DataFrame(columns=["regime", "mean", "median", "std", "n_visits"])

    runs = []  # list of (regime_id, run_length)
    cur_regime = regimes[0]
    cur_len = 1
    for r in regimes[1:]:
        if r == cur_regime:
            cur_len += 1
        else:
            runs.append((cur_regime, cur_len))
            cur_regime, cur_len = r, 1
    runs.append((cur_regime, cur_len))

    runs_df = pd.DataFrame(runs, columns=["regime", "length"])
    summary = runs_df.groupby("regime")["length"].agg(["mean", "median", "std", "count"])
    summary = summary.rename(columns={"count": "n_visits"}).reset_index()
    return summary


def transition_matrix(regimes: np.ndarray, k: int) -> pd.DataFrame:
    """Empirical transition probability matrix P(r' | r). Rows = from, cols = to."""
    counts = np.zeros((k, k), dtype=float)
    for a, b in zip(regimes[:-1], regimes[1:]):
        counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid div by zero
    probs = counts / row_sums
    return pd.DataFrame(
        probs,
        index=[f"from_r{i}" for i in range(k)],
        columns=[f"to_r{i}" for i in range(k)],
    )


if __name__ == "__main__":
    from load import synthesize_lobster_day, filter_session
    from features import compute_features

    df = synthesize_lobster_day(n_events=30_000)
    df = filter_session(df)
    X = compute_features(df, window=100).dropna()

    print("K-sweep:")
    sweep = k_sweep(X, k_values=[2, 3, 4, 5, 6])
    print(sweep)

    print("\nFitting K=4 ...")
    art = fit_clustering(X, k=4)
    regimes = assign_regimes(art, X)
    print(f"Regime distribution: {np.bincount(regimes)}")
    print(f"\nCentroids (original units):")
    print(regime_centroids_in_original_units(art).round(4))
    print(f"\nResidence times:")
    print(residence_time(regimes))
