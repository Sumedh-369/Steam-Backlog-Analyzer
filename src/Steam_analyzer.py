import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

load_dotenv()

class SteamBacklogAnalyzer:
    def __init__(self):
        self.df = None
        self.data_path = 'data/steam_games.csv'
    
    def load_data(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            self.df['playtime_hours'] = self.df['playtime_forever'] / 60
            print(f"✅ Loaded {len(self.df)} games")
            return True
        else:
            print("Run fetch_games.py first!")
            return False

    def analyze_playtime(self):
        print("\n=== Playtime Analysis ===")
        print(f"Total Playtime: {self.df['playtime_hours'].sum():.1f} hours")
        print(f"Games with 0 playtime: {(self.df['playtime_hours'] == 0).sum()}")
        
        # Backlog estimation
        backlog = self.df[self.df['playtime_hours'] < 2]  # less than 2 hours
        print(f"Potential Backlog (played < 2 hrs): {len(backlog)} games")

    def cluster_games(self, n_clusters=6):
        features = ['playtime_hours', 'playtime_2weeks']
        X = self.df[features].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(X_scaled)
        
        print("\nGame Clusters:")
        print(self.df['cluster'].value_counts())

    def run_full_analysis(self):
        if self.load_data():
            self.analyze_playtime()
            self.cluster_games()
            print("\nAnalysis complete!")

if __name__ == "__main__":
    analyzer = SteamBacklogAnalyzer()
    analyzer.run_full_analysis()