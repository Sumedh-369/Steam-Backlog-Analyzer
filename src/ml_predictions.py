"""
ml_predictions.py

Advanced Machine Learning module for Steam Backlog Analyzer.

Yeh file ye kaam karti hai:
- Data loading + feature engineering
- Completion Probability predict karna (Random Forest)
- Estimated time to finish game
- Smart recommendations

Prediction Module with multiple models.
User dashboard se different algorithms choose kar sakta hai.
Main Goal: "aap is game ko khatam kar paaoge ya nahi" predict karna.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_csv('data/steam_games.csv')
    df['playtime_hours'] = df['playtime_forever'] / 60.0
    df['playtime_2weeks'] = df.get('playtime_2weeks', 0).fillna(0)
    df['has_played'] = (df['playtime_hours'] > 0).astype(int)
    df['is_recently_played'] = (df['playtime_2weeks'] > 0).astype(int)
    return df

def train_and_predict(df, model_name="RandomForest"):
    """Multiple models support"""
    features = ['playtime_hours', 'playtime_2weeks', 'has_played', 'is_recently_played']
    X = df[features].fillna(0)
    y = (df['playtime_hours'] > 5).astype(int)   # Target: likely to complete

    # Model selection
    if model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X, y)
    df['completion_probability'] = model.predict_proba(X)[:, 1] * 100
    
    return df, model

def predict_time_to_finish(df):
    df = df.copy()
    conditions = [df['playtime_hours'] < 5, df['playtime_hours'] >= 5]
    choices = [15 - df['playtime_hours'], 40 - df['playtime_hours']]
    df['estimated_remaining_hours'] = np.select(conditions, choices, default=0)
    df['estimated_remaining_hours'] = df['estimated_remaining_hours'].clip(lower=0)
    return df

def get_insights(df):
    return df.nlargest(10, 'completion_probability')[
        ['name', 'playtime_hours', 'completion_probability', 'estimated_remaining_hours']
    ]