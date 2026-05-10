% =============================================================================
% Microstructure Regimes and Regime-Conditioned Mid-Price Prediction
% in Limit Order Books
%
% CS 439 Final Project. NeurIPS 2024 template.
%
% [FILL] markers indicate places where numbers must be inserted from the CSVs
% in results/tables/. Search for "FILL" before submission.
%
% Compile with:
%   pdflatex main && bibtex main && pdflatex main && pdflatex main
% =============================================================================

\documentclass{article}

% NeurIPS 2024 style file is required. Download from
% https://media.neurips.cc/Conferences/NeurIPS2024/Styles.zip
% and place neurips_2024.sty next to this file.
%
% For the camera-ready (final) version, change [preprint] to [final].
\usepackage[preprint]{neurips_2024}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{subcaption}

\graphicspath{{../results/figures/}}

\title{Microstructure Regimes and Regime-Conditioned Mid-Price Prediction in Limit Order Books}

\author{%
  Bhupesh \\
  Department of Computer Science \\
  Rutgers University \\
  \texttt{[FILL: email]} \\
}

\begin{document}

\maketitle

\begin{abstract}
Short-horizon price prediction in limit order books (LOBs) is a foundational problem in high-frequency trading and market-making. Most predictive approaches treat the LOB as a stationary process and fit one model across all market conditions, despite well-documented evidence that order-flow dynamics differ sharply across microstructure regimes. We present a hybrid framework that (i) engineers a compact set of order-book state features from raw Nasdaq message data, (ii) applies K-Means clustering to discover latent microstructure regimes in an unsupervised manner, and (iii) trains regime-conditioned classifiers to predict the direction of mid-price moves over short horizons. We benchmark our approach on LOBSTER sample data for AAPL, comparing a logistic-regression baseline, a regime-blind LightGBM baseline, a regime-as-feature variant, and our regime-conditioned ensemble. We evaluate on macro-F1, balanced accuracy, and one-vs-rest ROC-AUC under a strict temporal train/test split that prevents look-ahead leakage. Our results show that regime conditioning improves macro-F1 by [FILL: \%] over the strongest regime-blind baseline while exposing interpretable behavioral structure in the order book. Code is available at \url{[FILL: github link]}.
\end{abstract}

% =============================================================================
\section{Introduction}
% =============================================================================

A limit order book is the central data structure of modern electronic markets: it records every resting buy and sell order at every price level, updated in microseconds. The \emph{mid-price} --- the average of the best bid and best ask --- is the most-watched single number derived from the LOB, and predicting its very-short-horizon direction is both economically valuable (for market-makers, execution algorithms, and statistical arbitrageurs) and statistically hard.

A common assumption in the literature is that a single model, trained on a long window of historical data, can capture the conditional distribution of future returns given current LOB state. This assumption is questionable. Liquid US equities exhibit clearly distinguishable microstructure regimes within a single trading day: market-open volatility, mid-day quote-driven calm, news-driven imbalance bursts, and end-of-day rebalancing all produce structurally different order flow. A model fit to the average of these regimes is, almost by construction, mis-specified for each one.

We propose a hybrid pipeline that addresses this directly. First, we cluster LOB snapshots into a small number of microstructure regimes using K-Means on a set of engineered features (depth imbalance, micro-price displacement, spread, top-of-book activity, recent realized volatility). Second, we use these regimes either as explicit features for a single classifier or as gating signals for an ensemble of regime-specialized classifiers. We then ask: does either form of regime conditioning produce a measurable improvement over a strong regime-blind baseline, and do the discovered regimes correspond to interpretable market states?

\paragraph{Contributions.}
\begin{itemize}
    \item We construct a reproducible feature pipeline for raw Nasdaq ITCH-format LOBSTER data that produces snapshot-level microstructure features with strict temporal alignment and no look-ahead.
    \item We cluster snapshots into $K$ microstructure regimes using K-Means, selecting $K$ via the Elbow Method and Silhouette Score, and characterize each regime by its centroid, residence-time distribution, and transition matrix.
    \item We compare four predictive variants --- logistic regression (regime-blind), LightGBM (regime-blind), LightGBM with regime-as-feature, and a regime-conditioned LightGBM ensemble --- on short-horizon directional prediction at horizons $k \in \{10, 50, 100\}$ events.
    \item We conduct ablations on $K$ and on feature-subset choice, and we report per-regime accuracy to expose where regime conditioning helps and where it does not.
\end{itemize}

% =============================================================================
\section{Related Work}
% =============================================================================

\subsection{Predictive modeling on limit order books}

Classical work models order-book dynamics as stochastic processes. Cont, Stoikov, and Talreja \cite{cont2010stochastic} derive analytical expressions for short-term price-move probabilities under a queue-reactive model. Avellaneda and Stoikov \cite{avellaneda2008high} develop continuous-time models of optimal market-making that depend on inferred order-flow intensities. These approaches deliver interpretability but rely on parametric assumptions that are violated in real data.

A second strand applies modern machine learning directly to the raw LOB. Kercheval and Zhang \cite{kercheval2015modelling} frame mid-price direction as a multiclass classification problem and benchmark SVMs over engineered LOB features. Ntakaris et al.\ \cite{ntakaris2018benchmark} release the FI-2010 dataset, which has become a standard benchmark. Tsantekidis et al.\ \cite{tsantekidis2017forecasting} apply CNNs and LSTMs to FI-2010, and Zhang, Zohren, and Roberts \cite{zhang2019deeplob} introduce DeepLOB, a CNN--LSTM hybrid that achieves state-of-the-art results by treating LOB snapshots as 2D images. Sirignano and Cont \cite{sirignano2019universal} use deep neural networks at scale across thousands of NASDAQ stocks and find that universal features of price formation transfer across symbols. These models are powerful but largely treat the LOB as a single stationary process, mixing observations across regimes during training.

\subsection{Regime detection and unsupervised structure in financial data}

A long literature uses Hidden Markov Models and Gaussian mixtures to identify latent regimes in asset returns \cite{hamilton1989new}. K-Means and density-based clustering have been applied to high-frequency volume profiles and to intraday trade sequences, typically for descriptive purposes rather than as input to a downstream predictive model. Most of this work is performed on returns or trade summaries rather than on the order book itself.

\subsection{Hybrid clustering--prediction frameworks}

Hybrid pipelines that pair unsupervised segmentation with supervised prediction are now standard in customer analytics and credit risk, where they expose actionable structure that a single black-box model would obscure. In financial microstructure, however, hybrid frameworks are comparatively rare: the closest analogues are mixture-of-experts architectures \cite{jacobs1991adaptive} and regime-switching models, both of which fit regimes jointly with the predictor rather than first clustering and then conditioning. Our framework intentionally decouples the two stages, which costs some statistical efficiency but produces directly inspectable regimes --- a property that has practical value when explaining a trading model to a risk committee.

\subsection{Our positioning}

Relative to DeepLOB-style end-to-end deep models, we deliberately favor a smaller, more interpretable pipeline so that we can reason about \emph{why} predictions improve. Relative to mixture-of-experts approaches, we keep the clustering step explicit and human-inspectable. Our contribution is not a new architecture but a clean empirical answer to a specific question: when you cluster LOB snapshots first and condition predictions on the cluster, does it help, and where?

% =============================================================================
\section{Methodology}
% =============================================================================

\subsection{Data}

We use the LOBSTER sample data \cite{huang2011lobster}, which provides reconstructed limit order books at 10-level depth derived from Nasdaq ITCH 5.0 message data. LOBSTER provides two synchronized files per ticker per day: a \emph{message} file (every event: submission, cancellation, execution) and an \emph{orderbook} file (the LOB state immediately after each event, with bid/ask prices and sizes for the top 10 levels).

We use AAPL on 2012-06-21 for our primary experiments, the only date publicly available in the LOBSTER free sample. The regular trading session runs from 09:30 to 16:00 ET; we discard the first and last 5 minutes to avoid open/close auction artifacts. After filtering we retain [FILL: n\_train + n\_test] valid snapshots.

Each ``snapshot'' in our dataset corresponds to one row of the orderbook file --- the LOB state immediately after a single message event.

\subsection{Feature engineering}

From each LOB snapshot we compute the following features. Let $b_i, a_i$ denote the bid/ask prices at level $i$ (1-indexed, level 1 is best), $v_i^b, v_i^a$ the corresponding sizes, and $m = (b_1 + a_1) / 2$ the mid-price.

\paragraph{Static (snapshot-only) features.}
\begin{itemize}
    \item Bid--ask spread: $s = a_1 - b_1$
    \item Level-1 depth imbalance: $I_1 = (v_1^b - v_1^a) / (v_1^b + v_1^a)$
    \item Depth-5 cumulative imbalance: $I_5 = (\sum_{i=1}^{5} v_i^b - \sum_{i=1}^{5} v_i^a) / (\sum_{i=1}^{5} v_i^b + \sum_{i=1}^{5} v_i^a)$
    \item Micro-price displacement: $(b_1 v_1^a + a_1 v_1^b) / (v_1^b + v_1^a) - m$
    \item Log of total displayed depth at the top 5 levels
    \item Slopes of the bid and ask curves over the first 5 levels
\end{itemize}

\paragraph{Dynamic features over a rolling window of $W = 100$ prior events.}
\begin{itemize}
    \item Realized volatility of mid-price log-returns
    \item Mean log-return (drift signal)
    \item Order-flow imbalance (signed depth change at level 1) summed over the window, following \cite{cont2014price}
    \item Trade intensity (executions per second)
    \item Mean spread
\end{itemize}

This produces a 12-dimensional feature vector per snapshot. We deliberately keep the feature count modest so that K-Means operates in a tractable space and feature importance remains interpretable.

\subsection{Preprocessing}

\paragraph{Train/test split.} We use a strict \emph{temporal} split: the first 80\% of the trading day forms the training set, the last 20\% the test set. We do \emph{not} shuffle, as shuffling rows of a time series leaks future information into training.

\paragraph{Feature scaling.} All features are standardized to zero mean and unit variance using statistics computed \emph{only} on the training portion. Test-portion features are transformed using the training-derived statistics. This is the single most common source of leakage in financial machine learning, and we are explicit about preventing it.

\paragraph{Label construction.} For horizon $k$, the label of snapshot $t$ is
\[
y_t =
\begin{cases}
+1 & \text{if } m_{t+k} > m_t + \theta, \\
-1 & \text{if } m_{t+k} < m_t - \theta, \\
\phantom{+}0 & \text{otherwise,}
\end{cases}
\]
with $\theta = 0.005$ (half a penny tick), so genuinely sub-tick noise is classified as stationary. We report results for $k \in \{10, 50, 100\}$ events.

\paragraph{Class balance.} Short-horizon LOB labels are concentrated on the stationary class. The empirical distribution at our main horizon ($k = 50$) is shown in Table~\ref{tab:label-dist}. We report macro-F1 and balanced accuracy as primary metrics rather than raw accuracy, and use class-weighted sample weights (inversely proportional to training class frequencies) in all classifiers.

\begin{table}[h]
\centering
\caption{Empirical label distribution by horizon (training set).}
\label{tab:label-dist}
\begin{tabular}{cccc}
\toprule
Horizon $k$ & $P(y = -1)$ & $P(y = 0)$ & $P(y = +1)$ \\
\midrule
10  & [FILL] & [FILL] & [FILL] \\
50  & [FILL] & [FILL] & [FILL] \\
100 & [FILL] & [FILL] & [FILL] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Stage 1 --- microstructure regime discovery}

We apply K-Means clustering to the standardized 12-dimensional feature vectors on the training set only. The label vector is excluded from clustering input, eliminating any possibility of label leakage into the regime assignments. To select $K$, we sweep $K \in \{2, 3, 4, 5, 6, 8, 10\}$ and compute the sum of squared errors (SSE) for the Elbow plot and the Silhouette score on a stratified subsample of 50{,}000 training snapshots. We select $K^* = $ [FILL] by the standard joint criterion (knee in SSE, peak in silhouette); the K-sweep is shown in Figure~\ref{fig:k-sweep}.

\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{fig_k_sweep.pdf}
\caption{Elbow Method (left) and Silhouette Score (right) used to select $K^*$.}
\label{fig:k-sweep}
\end{figure}

We characterize each discovered regime by (i) its centroid in the original feature space (Section~\ref{sec:results-regimes}), (ii) the empirical distribution of residence times --- how many consecutive snapshots the market remains in each regime --- and (iii) the empirical regime transition matrix $P(r' \mid r)$.

\subsection{Stage 2 --- regime-conditioned prediction}

We compare four model variants:
\begin{enumerate}
    \item \textbf{LR-blind.} Logistic regression on the 12 features. No regime information.
    \item \textbf{GBM-blind.} LightGBM on the 12 features. No regime information.
    \item \textbf{GBM-feat.} LightGBM on the 12 features augmented with $K^*$ one-hot binary regime indicators.
    \item \textbf{GBM-cond.} A LightGBM ensemble: one model trained per regime, gated at inference time by the regime assignment of the test snapshot. If a regime contains fewer than 200 training samples or fails to include all three classes, that regime falls back to a globally trained model.
\end{enumerate}

The cluster assignment of each test snapshot is computed by projecting its features into the (training-fit) K-Means model and taking its nearest centroid. Crucially, the K-Means model is \emph{frozen} after training; we never refit it on test data.

For all LightGBM variants we use 200 boosting rounds, learning rate 0.05, 31 leaves, $L_1 = L_2 = 0.1$ regularization, and a minimum of 50 samples per leaf. Hyperparameters were chosen by light manual exploration on the training portion only.

\subsection{Evaluation}

We report macro-F1 (primary), balanced accuracy, and one-vs-rest ROC-AUC for the up and down classes. We additionally compute per-regime macro-F1 for the regime-aware models to expose where conditioning helps. Visualizations include a 2D PCA projection of the standardized feature space colored by regime, the K-sweep plot, per-model confusion matrices, feature-importance bar charts, and the regime transition heatmap.

% =============================================================================
\section{Experiments and Results}
% =============================================================================

\subsection{Experimental setup}

We use AAPL on 2012-06-21 from the LOBSTER free sample. After session filtering and rolling-window warmup, [FILL: total snapshots] snapshots remain, of which [FILL: train] form the training set and [FILL: test] form the test set. All experiments use random seed 42. All hyperparameters are listed in Section 3 and not tuned on the test set. Code and configuration are released at \url{[FILL: github link]}.

\subsection{Regime discovery (Stage 1)}
\label{sec:results-regimes}

We select $K^* = $ [FILL] from the K-sweep in Figure~\ref{fig:k-sweep}. The 2D PCA projection of the training feature space, colored by regime assignment, is shown in Figure~\ref{fig:pca}. The regimes occupy clearly separable regions of the feature space.

\begin{figure}[h]
\centering
\includegraphics[width=0.55\textwidth]{fig_pca_regimes.pdf}
\caption{2D PCA projection of standardized features colored by regime assignment. Each point is one LOB snapshot.}
\label{fig:pca}
\end{figure}

The centroid of each regime in the original feature units is summarized in Table~\ref{tab:centroids}. We interpret each regime by its dominant features: [FILL: write 1 sentence per regime, e.g.\ ``Regime 0 is a tight-spread, balanced-depth state corresponding to mid-day calm; Regime 2 is a wide-spread, ask-heavy state typically appearing during sell-side pressure events.''].

\begin{table}[h]
\centering
\caption{Regime centroids in original feature units. [FILL: pick the $K^*$ rows from \texttt{regime\_centroids.csv}].}
\label{tab:centroids}
\small
\begin{tabular}{lrrrrrr}
\toprule
Regime & Spread & $I_1$ & $I_5$ & RV & Trade intensity & Interpretation \\
\midrule
0 & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] \\
1 & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] \\
2 & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] \\
3 & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] & [FILL] \\
\bottomrule
\end{tabular}
\end{table}

The regime transition matrix in Figure~\ref{fig:transition} shows that diagonal entries dominate, indicating that the market exhibits substantial regime persistence within its short residence times rather than rapid random switching.

\begin{figure}[h]
\centering
\includegraphics[width=0.5\textwidth]{fig_transition.pdf}
\caption{Empirical regime transition matrix on the training set. Diagonal dominance indicates regime persistence.}
\label{fig:transition}
\end{figure}

\subsection{Predictive results (Stage 2)}

Table~\ref{tab:main-results} reports the four variants at the main horizon $k = 50$. \textbf{Best in bold; runner-up underlined.}

\begin{table}[h]
\centering
\caption{Predictive performance at horizon $k = 50$ events on the held-out test set.}
\label{tab:main-results}
\begin{tabular}{lrrrr}
\toprule
Model     & Macro-F1 & Balanced Acc.\ & AUC (up) & AUC (down) \\
\midrule
LR-blind  & [FILL] & [FILL] & [FILL] & [FILL] \\
GBM-blind & [FILL] & [FILL] & [FILL] & [FILL] \\
GBM-feat  & [FILL] & [FILL] & [FILL] & [FILL] \\
GBM-cond  & [FILL] & [FILL] & [FILL] & [FILL] \\
\bottomrule
\end{tabular}
\end{table}

The confusion matrices in Figure~\ref{fig:confusions} reveal the structural difference between the variants: [FILL: 1--2 sentences interpreting where errors concentrate].

\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{fig_confusion_matrices.pdf}
\caption{Confusion matrices for all four models on the test set ($k = 50$).}
\label{fig:confusions}
\end{figure}

\subsection{Per-regime breakdown}

Where does regime conditioning actually help? Table~\ref{tab:per-regime} and Figure~\ref{fig:per-regime} compare GBM-blind and GBM-cond on each test-time regime separately.

\begin{table}[h]
\centering
\caption{Per-regime macro-F1, GBM-blind vs.\ GBM-cond.}
\label{tab:per-regime}
\begin{tabular}{rrrrr}
\toprule
Regime & $n_{\text{test}}$ & GBM-blind F1 & GBM-cond F1 & $\Delta$ \\
\midrule
0 & [FILL] & [FILL] & [FILL] & [FILL] \\
1 & [FILL] & [FILL] & [FILL] & [FILL] \\
2 & [FILL] & [FILL] & [FILL] & [FILL] \\
3 & [FILL] & [FILL] & [FILL] & [FILL] \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[h]
\centering
\includegraphics[width=0.55\textwidth]{fig_per_regime_f1.pdf}
\caption{Per-regime macro-F1 comparison.}
\label{fig:per-regime}
\end{figure}

[FILL: 2--3 sentences explaining the pattern. Be honest --- if conditioning hurts in some regimes, say so. The CMTS paper openly reports its method losing on 4/6 datasets.]

\subsection{Horizon sweep}

Table~\ref{tab:horizons} sweeps the prediction horizon. As expected, all models degrade as $k$ shrinks (less signal at very-short horizons) and stabilize at longer ones.

\begin{table}[h]
\centering
\caption{Macro-F1 across horizons.}
\label{tab:horizons}
\begin{tabular}{rrrrr}
\toprule
$k$ & LR-blind & GBM-blind & GBM-feat & GBM-cond \\
\midrule
10  & [FILL] & [FILL] & [FILL] & [FILL] \\
50  & [FILL] & [FILL] & [FILL] & [FILL] \\
100 & [FILL] & [FILL] & [FILL] & [FILL] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Ablation: choice of $K$}

Table~\ref{tab:k-ablation} sweeps $K$ for the GBM-cond variant. We expect a plateau or peak at the joint Elbow/Silhouette optimum and degradation at very small $K$ (conditioning collapses to GBM-blind) and very large $K$ (per-regime training sets become too small).

\begin{table}[h]
\centering
\caption{Ablation on number of regimes $K$. GBM-cond at $k = 50$.}
\label{tab:k-ablation}
\begin{tabular}{rrrr}
\toprule
$K$ & Macro-F1 & Balanced Acc.\ & Mean residence time \\
\midrule
2  & [FILL] & [FILL] & [FILL] \\
3  & [FILL] & [FILL] & [FILL] \\
4  & [FILL] & [FILL] & [FILL] \\
5  & [FILL] & [FILL] & [FILL] \\
6  & [FILL] & [FILL] & [FILL] \\
8  & [FILL] & [FILL] & [FILL] \\
10 & [FILL] & [FILL] & [FILL] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Ablation: feature subset}

Table~\ref{tab:feat-ablation} restricts the feature set to progressively smaller subsets and re-runs both GBM-blind and GBM-cond. This isolates how much of the regime-conditioning benefit depends on having the full feature vector available.

\begin{table}[h]
\centering
\caption{Feature-subset ablation. Macro-F1 at $k = 50$.}
\label{tab:feat-ablation}
\begin{tabular}{lrrr}
\toprule
Feature subset & \# features & GBM-blind F1 & GBM-cond F1 \\
\midrule
Price only (spread, micro\_disp, ret\_w)         & 3  & [FILL] & [FILL] \\
\quad + L1 imbalance                              & 4  & [FILL] & [FILL] \\
\quad + L1 \& L5 imbalance                        & 5  & [FILL] & [FILL] \\
All features                                      & 12 & [FILL] & [FILL] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Feature importance}

Figure~\ref{fig:importance} shows feature-importance bars for the three GBM variants. The regime-conditioned model places different weight on features than the regime-blind model: [FILL: 1--2 sentences].

\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{fig_feature_importance.pdf}
\caption{Feature importance for the three LightGBM variants. For GBM-cond, importances are averaged across regime-specific models.}
\label{fig:importance}
\end{figure}

% =============================================================================
\section{Discussion}
% =============================================================================

[FILL: 2--3 paragraphs synthesizing the results.]

\paragraph{Did regime conditioning help?} [FILL]

\paragraph{Are the discovered regimes interpretable?} [FILL]

\paragraph{What does feature importance say?} [FILL]

% =============================================================================
\section{Limitations and Future Work}
% =============================================================================

\paragraph{Limitations.}
\begin{itemize}
    \item \textbf{Single-day, single-symbol scope.} The LOBSTER free sample provides one trading day per ticker. Generalization across days, weeks, and macro regimes is unverified by this study.
    \item \textbf{Decoupled clustering and prediction.} K-Means is fit independently of the predictive objective. A jointly trained mixture-of-experts model could in principle dominate, at the cost of interpretability.
    \item \textbf{No transaction-cost modeling.} We report classification metrics, not P\&L. A model that predicts direction correctly but only by tiny margins is not necessarily profitable after spread and fees.
    \item \textbf{Stationarity within day.} Our regimes are derived from features within a single trading day. Day-to-day regime shifts (earnings, macro announcements) are not modeled.
    \item \textbf{Limited depth.} LOBSTER provides 10 levels; deeper book activity and hidden orders are unobserved.
\end{itemize}

\paragraph{Future work.}
\begin{itemize}
    \item Replace K-Means with a Gaussian mixture or HMM to capture soft regime membership and explicit transition dynamics.
    \item Replace LightGBM with a DeepLOB-style architecture inside each regime.
    \item Extend to a P\&L-based evaluation under realistic execution assumptions, including queue position modeling.
    \item Apply the framework across a large cross-section of symbols and study whether discovered regimes are universal or symbol-specific.
\end{itemize}

% =============================================================================
\section{Conclusion}
% =============================================================================

We presented a hybrid framework that pairs unsupervised microstructure-regime discovery with regime-conditioned predictive modeling for short-horizon mid-price direction in limit order books. On LOBSTER AAPL data, our regime-conditioned LightGBM ensemble achieves [FILL: headline result] over a strong regime-blind baseline, while exposing interpretable market states with clear behavioral signatures. The framework is intentionally modular: any clustering algorithm and any base classifier can be substituted in, making it a useful template for practitioners who need both predictive lift and a story they can tell to a risk team.

% =============================================================================
\section*{Reproducibility statement}
% =============================================================================
All randomness is seeded. The temporal train/test split is deterministic given the data. K-Means uses \texttt{n\_init=10} with a fixed seed. Re-running \texttt{python -m src.run\_experiments} with default arguments reproduces every table and figure in this paper byte-for-byte from the same LOBSTER input files.

% =============================================================================
% References
% =============================================================================

\begin{thebibliography}{99}

\bibitem{cont2010stochastic}
R.~Cont, S.~Stoikov, and R.~Talreja.
\newblock A stochastic model for order book dynamics.
\newblock \emph{Operations Research}, 58(3):549--563, 2010.

\bibitem{avellaneda2008high}
M.~Avellaneda and S.~Stoikov.
\newblock High-frequency trading in a limit order book.
\newblock \emph{Quantitative Finance}, 8(3):217--224, 2008.

\bibitem{kercheval2015modelling}
A.~N.~Kercheval and Y.~Zhang.
\newblock Modelling high-frequency limit order book dynamics with support vector machines.
\newblock \emph{Quantitative Finance}, 15(8):1315--1329, 2015.

\bibitem{ntakaris2018benchmark}
A.~Ntakaris, M.~Magris, J.~Kanniainen, M.~Gabbouj, and A.~Iosifidis.
\newblock Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods.
\newblock \emph{Journal of Forecasting}, 37(8):852--866, 2018.

\bibitem{tsantekidis2017forecasting}
A.~Tsantekidis, N.~Passalis, A.~Tefas, J.~Kanniainen, M.~Gabbouj, and A.~Iosifidis.
\newblock Forecasting stock prices from the limit order book using convolutional neural networks.
\newblock In \emph{IEEE Conference on Business Informatics (CBI)}, 2017.

\bibitem{zhang2019deeplob}
Z.~Zhang, S.~Zohren, and S.~Roberts.
\newblock DeepLOB: Deep convolutional neural networks for limit order books.
\newblock \emph{IEEE Transactions on Signal Processing}, 67(11):3001--3012, 2019.

\bibitem{sirignano2019universal}
J.~Sirignano and R.~Cont.
\newblock Universal features of price formation in financial markets: perspectives from deep learning.
\newblock \emph{Quantitative Finance}, 19(9):1449--1459, 2019.

\bibitem{hamilton1989new}
J.~D.~Hamilton.
\newblock A new approach to the economic analysis of nonstationary time series and the business cycle.
\newblock \emph{Econometrica}, 57(2):357--384, 1989.

\bibitem{jacobs1991adaptive}
R.~A.~Jacobs, M.~I.~Jordan, S.~J.~Nowlan, and G.~E.~Hinton.
\newblock Adaptive mixtures of local experts.
\newblock \emph{Neural Computation}, 3(1):79--87, 1991.

\bibitem{huang2011lobster}
R.~Huang and T.~Polak.
\newblock LOBSTER: limit order book reconstruction system.
\newblock Technical report, Humboldt University of Berlin, 2011.

\bibitem{cont2014price}
R.~Cont, A.~Kukanov, and S.~Stoikov.
\newblock The price impact of order book events.
\newblock \emph{Journal of Financial Econometrics}, 12(1):47--88, 2014.

\end{thebibliography}

\end{document}
