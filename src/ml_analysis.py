import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

def load_data():
    df = pd.read_csv('data/steam_games.csv')
    df['playtime_hours'] = df['playtime_forever'] / 60.0
    df['playtime_2weeks'] = df.get('playtime_2weeks', 0).fillna(0)
    return df

def add_ml_features(df):
    """Add clustering and simple prediction features"""
    df = df.copy()
    
    # Clustering
    features = ['playtime_hours', 'playtime_2weeks']
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Simple "Abandon Risk" score (higher = more likely to abandon)
    df['abandon_risk'] = 100 - (df['playtime_hours'] * 5).clip(upper=90)
    df.loc[df['playtime_2weeks'] > 0, 'abandon_risk'] = 10  # Active games low risk
    
    return df

def get_recommendations(df, n=8):
    """Recommend games you might enjoy finishing"""
    # Games that are started but not heavily played
    candidates = df[(df['playtime_hours'] > 1) & (df['playtime_hours'] < 20)]
    return candidates.nlargest(n, 'playtime_hours')[['name', 'playtime_hours', 'abandon_risk']]

if __name__ == "__main__":
    df = load_data()
    df = add_ml_features(df)
    print("ML Analysis Done!")
    print(get_recommendations(df))