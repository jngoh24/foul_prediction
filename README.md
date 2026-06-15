# Foul-Won Analytics Dashboard — EPL 2015/16

Interactive Streamlit dashboard exploring **who draws fouls, where fouls are won, and
the probability a foul is won on a given possession** after a player receives or
recovers the ball (StatsBomb open event data).

## Tabs
- **Players** — most-fouled players (totals, per game, by position) with leaderboard + charts.
- **Teams** — league table of fouls drawn (total & per game); click a row (or use the
  dropdown) to drill into a club and see its top 3 foul-winners with positions.
- **Pitch** — foul-won rate by pitch location, by play pattern, and receipt vs recovery,
  filterable by how the ball was gained.
- **Model** — set a situation (location, pressure, play pattern, player profile, possession
  context) and get the calibrated probability of winning a foul on that play.

The model (histogram gradient boosting, ROC-AUC ≈ 0.81, well calibrated) is trained
once in-app from `data/gains.csv` and cached, so there are no serialized-model
version dependencies.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data
- `data/player_stats.csv` — per-player fouls drawn, games, position, foul-drawing tendency
- `data/team_stats.csv` — per-team fouls drawn, games, touches
- `data/gains.csv` — one row per ball reception/recovery (323,322) with engineered
  features and the foul-won label; powers the pitch tab and the model.
