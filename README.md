# Baller-Finder.AI

> A football player similarity engine — search any player, find their statistical twins across 28,000+ careers.
---

## What it does

Baller-Finder takes any player name as input and finds the most statistically similar players in the dataset using a K-Nearest Neighbors model trained on per-90 performance metrics. Results are displayed as ranked cards with a radar chart comparison and an AI-generated scout report written by Gemini 2.0 Flash.

---

## Features

- **Player search** across 28,665 indexed careers
- **Bayesian-shrunk per-90 stats** — no small sample distortion
- **KNN similarity engine** using cosine distance on scaled features
- **Radar chart** comparing query player vs top 3 similar players
- **AI scout report** streamed in real time via Gemini 2.0 Flash
- **Low confidence badges** flagging players with limited minutes
- **Same position filter** toggle
- Clean dark UI built with Streamlit

---

## Project structure

```
baller-finder/
├── app.py                  ← main Streamlit application
├── .gitignore
├── .streamlit/
│   └── secrets.toml        ← API keys (never committed)
├── notebooks/
│   └── data_prep.ipynb     ← EDA and feature engineering
└── data/                   ← CSV files from Kaggle (not committed)
    ├── appearances.csv
    ├── players.csv
    ├── player_valuations.csv
    ├── clubs.csv
    ├── transfers.csv
    └── ...
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/baller-finder.git
cd baller-finder
```

### 2. Download the dataset

Get the CSVs from [Kaggle — Player Scores by davidcariboo](https://www.kaggle.com/datasets/davidcariboo/player-scores/data) and place them in a `data/` folder.

### 3. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn plotly google-generativeai
```

### 4. Add your Gemini API key

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "AIza..."
```

Get a free key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

### 5. Run the app

```bash
streamlit run app.py
```

---

## How the similarity engine works

### Feature matrix

Each player is represented by a 6-dimensional vector:

| Feature | Description |
|---|---|
| `goals_p90` | Goals per 90 minutes (Bayesian shrunk) |
| `assists_p90` | Assists per 90 minutes (Bayesian shrunk) |
| `total_cards` | Disciplinary record |
| `mins_per_game` | Average minutes per appearance |
| `age` | Player age at time of analysis |
| `position_encoded` | Label-encoded sub-position |

### Bayesian shrinkage

Raw per-90 stats are unreliable for players with few appearances. Instead of dropping them with a hard cutoff, we blend each player's rate toward the league average weighted by sample size:

```
shrunk_rate = (goals + league_avg × (500/90)) / ((minutes + 500) / 90)
```

Players with 3,000+ minutes keep their true rate. Players with 50 minutes get pulled toward the mean. No player is dropped from the index.

### KNN with cosine distance

Features are standardized with `StandardScaler` and fed into a `NearestNeighbors` model using cosine distance. Cosine distance measures the angle between feature vectors rather than magnitude, making it robust to players with different career lengths.

---

## Future scope

### Additional CSVs (already in the dataset)

| File | What it unlocks |
|---|---|
| `player_valuations.csv` | Market value timeline per player — add a value trajectory chart and a "hidden gems" filter (high performance, low value) |
| `transfers.csv` | Career path visualization — show every club a player has moved through as a timeline |
| `game_events.csv` | Granular event data — break down goals into headers, penalties, open play; weight goal quality not just quantity |
| `game_lineups.csv` | Starting XI vs substitute analysis — separate starter profiles from impact sub profiles |
| `club_games.csv` | Club-level context — normalize stats by team strength, flag players performing above their club's level |
| `competitions.csv` | Competition difficulty weighting — a goal in the Champions League weighted higher than a lower-league goal |

### Additional APIs and integrations

**Performance & enrichment**
- **Football-Data.org API** — free tier gives live fixtures, standings, and squad data; could keep player clubs current
- **API-Football (RapidAPI)** — richer stats including xG, progressive passes, pressures; would replace or supplement the per-90 features
- **Transfermarkt scraper** — scrape current market valuations to keep the valuation data live rather than relying on the static CSV

**AI enhancements**
- **Anthropic Claude API** — swap or A/B test against Gemini for scout report generation; Claude's longer context window handles full career histories
- **OpenAI Embeddings** — embed player career narratives as text and use vector similarity alongside statistical similarity for a hybrid recommender
- **Perplexity API** — ground scout reports in recent news ("Player X was injured in March 2025") by adding a web search step before generation

**Deployment & data**
- **Streamlit Cloud** — one-click deploy directly from this GitHub repo; add `requirements.txt` and it handles the rest
- **Supabase** — move the aggregated player DataFrame into a Postgres table; enables filtering by league, nationality, age range server-side instead of in-memory
- **Kaggle API** — automate weekly dataset refresh with `kaggle datasets download` in a cron job so the index stays current

---

## Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
google-generativeai
```

---

## Data source

[Transfermarkt Player Scores — davidcariboo on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores/data)

Data is not included in this repository. Download and place in `data/` before running.
##Also I have run out of Free Uses. :( 
---

## License

MIT
