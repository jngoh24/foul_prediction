# Foul-Won Analytics Dashboard — EPL 2015/16

Interactive Streamlit dashboard exploring **who draws fouls, where fouls are won, and
the probability a foul is won on a given possession** after a player receives or
recovers the ball (StatsBomb open event data). Styled to match the HSR dashboard
design system (Athletic-inspired light theme, Plotly charts, Source Serif 4 / Inter).

## Tabs
- **Players** — most-fouled players (totals, per game, by position) with leaderboard + charts.
- **Teams** — league table of fouls drawn; click a row (or use the dropdown) to drill
  into a club and see its top 3 foul-winners with positions.
- **Pitch** — foul-won rate by pitch location, by play pattern, and receipt vs recovery,
  filterable by how the ball was gained.
- **Model** — set a situation (location, pressure, play pattern, player profile, possession
  context) and get the calibrated probability of winning a foul on that play.

The model (histogram gradient boosting, ROC-AUC ≈ 0.81, well calibrated) is trained
once in-app from `data/gains.parquet` and cached, so there are no serialized-model
version dependencies.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Layout (deploy with app.py + data/ at the repo root)
```
app.py
requirements.txt
.streamlit/config.toml
data/
├── gains.parquet      # one row per ball reception/recovery (323,322) — pitch tab + model
├── player_stats.csv   # per-player fouls drawn, games, position, foul-drawing tendency
└── team_stats.csv     # per-team fouls drawn, games, touches
```
Data paths are resolved relative to `app.py`, so `data/` can sit next to it at any nesting level.
