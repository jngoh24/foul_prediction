# Foul-Won Modelling & Analytics — Premier League 2015/16

Predicting the probability that a player **wins a foul on his possession** after he receives or recovers the ball — built on detailed match-event data — plus an interactive Streamlit dashboard to explore who draws fouls, where they happen, and what drives them.

[![Live dashboard](https://img.shields.io/badge/Streamlit-Live%20Demo-1a6b3c?logo=streamlit&logoColor=white)](https://foulprediction-cjzv4kdpsc5mwmq2dtnfzf.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Model](https://img.shields.io/badge/ROC--AUC-0.81-1a6b3c)

---

## Overview

Drawing fouls is a quietly valuable skill — it halts opposition transitions, earns set-piece territory, and draws cards, and some players do it far more than others. This project models, for **every** ball reception or recovery in the 2015/16 Premier League (323,322 of them), the calibrated probability that the player wins a foul before giving the ball up, and ships an analytics dashboard on top of the same data.

## Live dashboard

**[Open the dashboard →](https://foulprediction-cjzv4kdpsc5mwmq2dtnfzf.streamlit.app/)**

Four tabs:

| Tab | What it shows |
|---|---|
| **Players** | Most-fouled players — totals, per-game rates, by position |
| **Teams** | League table of fouls drawn; click a club for its top 5 foul-winners |
| **Pitch** | Foul-won rate by pitch location, play pattern, and receipt vs recovery |
| **Model** | Set a situation and get the calibrated foul-win probability live |

![Foul-Won dashboard — Players tab](assets/dashboard.png)

![Foul-Won dashboard — Model tab](assets/model_tab.png)

## Key results

| Metric | Value |
|---|---|
| Touches modelled | 323,322 |
| Base rate (foul won / possession) | 1.83% |
| **ROC-AUC** | **0.81** (0.800 ± 0.003 over 5 CV splits) |
| **PR-AUC** | **0.099** (≈ 5.5× base rate) |
| Top-decile capture | ~47% of all fouls in 10% of touches |
| Calibration | reliable overall **and** within subgroups |

**What drives it:** being *under pressure* when you get the ball, *who the player is* (a learned foul-drawing tendency), and *play pattern* (counter-attacks draw fouls ~5× the average).

## Repository structure

```
.
├── app.py                  # Streamlit dashboard (single file)
├── requirements.txt
├── .streamlit/
│   └── config.toml         # theme
├── data/
│   ├── gains.parquet       # one row per ball reception/recovery (features + label)
│   ├── player_stats.csv
│   └── team_stats.csv
└── modeling/
    ├── foul_won_model.ipynb   # full analysis: EDA → features → model → evaluation
    └── foul_won_model.html    # rendered, no setup required to read
```

## The model

- **Type:** histogram gradient boosting (scikit-learn `HistGradientBoostingClassifier`), benchmarked against a base-rate baseline, logistic regression, and random forest.
- **Target:** a "touch" (successful ball receipt or recovery) is labelled `1` if the same player wins a foul in that possession before giving the ball up.
- **Features:** spatial (location + distances), pressure, play pattern, position, possession context (transition vs settled), incoming-pass attributes, and a **leakage-safe, out-of-fold player foul-drawing rate** — the single most important engineered feature.
- **Evaluation:** ROC/PR curves, calibration (overall + subgroup), precision/recall by threshold, top-K targeting, segment performance, error analysis, partial dependence, and CV stability — all in the notebook.

## Run locally

**Dashboard**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Notebook** (re-runs the full analysis from the raw event CSVs)
```bash
cd modeling
jupyter notebook foul_won_model.ipynb
```
> The raw event file (~486 MB) is not committed (exceeds GitHub limits). Point the notebook's two path variables at your local copy of the data to re-execute. The committed `.html` is fully rendered and needs no setup.

## Data

Event-level match data for the 2015/16 Premier League season — 380 matches, ~1.3M events.

## Tech stack

`Python` · `pandas` · `scikit-learn` · `Plotly` · `Streamlit`

## Author

**Julian Ngoh** — [GitHub](https://github.com/jngoh24)
