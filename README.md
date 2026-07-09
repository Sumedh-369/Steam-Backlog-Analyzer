# Steam-Backlog-Analyzer
Steam games backlog analyzer with data fetching , ML clustering and Streamlit Dashboard

# 🎮 Steam Backlog Analyzer

A project that analyse your Steam library.It determine patterns and predicts probability of you finishing the game.

## Features
- Steam API se automatic data fetching
- Playtime analysis aur backlog calculation
- KMeans Clustering (similar games grouping)
- ML-based Completion Prediction ("Ye game khatam kar paaoge?")
- Beautiful Streamlit Dashboard
- Visualization charts (Donut, Pareto, etc.)

## Tech Stack
- **Python** + Pandas + Scikit-learn
- **Streamlit** (Dashboard)
- **GitHub Actions** (future: daily refresh)

## How to Run

1. Clone the repo
2. `python -m venv venv` To activate the Environment
3. `pip install -r requirements.txt`
4. `.env` create this file and add Steam API key 
5. `python src/fetch_games.py`
6. `streamlit run src/app.py`

## Project Structure
- `src/fetch_games.py` → Fetch data from Steam 
- `src/visualize_games.py` → Charts and classification
- `src/ml_predictions.py` → Advanced ML models
- `src/app.py` → Main Dashboard

## Team
- Sumedh Rokade
- Rishabh Gadpal

Made with ❤️ for learning AI + DevOps + Cloud