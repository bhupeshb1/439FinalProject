# Microstructure Regimes and Regime-Conditioned Mid-Price Prediction

CS 439 Final Project. A hybrid pipeline that pairs unsupervised microstructure-regime
discovery (K-Means on engineered LOB features) with regime-conditioned predictive
models for short-horizon mid-price direction in a limit order book.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Get the data

The LOBSTER free sample provides one trading day (2012-06-21) for AAPL, AMZN, GOOG,
INTC, MSFT at level-10 depth. Download:

```bash
bash download_data.sh AAPL          # primary ticker
bash download_data.sh AAPL MSFT INTC # plus held-out tickers
```

If the script fails (URLs change occasionally), download manually from
<https://lobsterdata.com/info/DataSamples.php> and place the unzipped CSVs in
`data/lobster/`.

## Run the experiments

```bash
# All experiments at default settings (AAPL, horizons 10/50/100, K* = 4)
python -m src.run_experiments \
    --data-dir data/lobster \
    --ticker AAPL --date 2012-06-21 \
    --horizons 10 50 100 \
    --k-values 2 3 4 5 6 8 10 \
    --k-star 4

# Smoke test on synthetic data (no LOBSTER download needed)
python -m src.run_experiments --use-synthetic --n-synthetic 50000

# Generalization check on a different ticker
python -m src.run_experiments --ticker MSFT --results-dir results_msft
```

Outputs land in `results/`:
- `results/tables/` — every table cited in the paper as a CSV
- `results/figures/` — every figure as a PDF
- `results/run_config.json` — exact configuration that produced the results

## Project structure

```
src/
├── load.py              # parse LOBSTER message + orderbook files
├── features.py          # static + rolling features
├── label.py             # k-step-ahead labeling
├── cluster.py           # K-Means + Elbow/Silhouette + regime stats
├── models.py            # LR-blind, GBM-blind, GBM-feat, GBM-cond
├── evaluate.py          # macro-F1, balanced-acc, AUC, per-regime
├── plots.py             # all paper figures
└── run_experiments.py   # end-to-end runner
paper/
└── main.tex             # NeurIPS-formatted report (uses results/ as inputs)
data/lobster/            # LOBSTER CSVs (gitignored)
results/                 # outputs (gitignored)
```

## Reproducibility

All randomness is seeded (`--seed 42` by default). The temporal train/test split
is deterministic given the data. K-Means uses `n_init=10` with a fixed seed.
Re-running the script reproduces the same tables and figures byte-for-byte.

## Notes on the methodology

- The train/test split is **temporal** — the first 80% of the trading day is
  training, the last 20% is test. We never shuffle, since shuffling rows of a
  time series leaks future information into training.
- Feature scaling and K-Means are **fit on training data only**. The scaler and
  cluster centroids are frozen before any test-time prediction.
- Labels are constructed as a 3-class problem (down / stationary / up) using a
  threshold of half a tick at horizons k ∈ {10, 50, 100} events.
- All classifiers use class-balanced sample weighting since the stationary class
  dominates at short horizons.

## License

Project code: MIT. LOBSTER data is subject to LOBSTER's terms; see
<https://lobsterdata.com/>.
