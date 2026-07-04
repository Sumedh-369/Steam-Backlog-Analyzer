import os
import requests # real time data access from internet (For api call)
import pandas as pd
from dotenv import load_dotenv # to read the api key from .env

#Load Environment Variables
load_dotenv() 
STEAM_API_KEY = os.getenv('STEAM_API_KEY')


if not STEAM_API_KEY:
    print("Error: .env file doesn't contain STEAM_API_KEY ")
    exit(1)

# Your Steam ID (change this to your own)
STEAM_ID = "76561198350163815"

def fetch_owned_games(steam_id):
    #Fetch owned games from steam Api
    url="https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
    
    params={
        'key': STEAM_API_KEY,
        'steamid':steam_id,
        'format': 'json',
        'include_appinfo':True, #game details like icon , name etc 
        'include_played_free_games': True
    }
    
    try:
        response= requests.get(url,params=params)
        response.raise_for_status() #if error occurs throw exeception
        
        data=response.json()
        games= data.get('response',{}).get('games',[])
        print(f"Successfully Fetched {len(games)} games!")
        return games
    except requests.exceptions.RequestException as e:
        print(f"api Error: {e}")
        return None

if __name__=="__main__":
        games=fetch_owned_games(STEAM_ID)
        
        if games:
            df=pd.DataFrame(games)
            print("\n first 5 games")
            print(df[['name','playtime_forever','appid']].head())
            
            os.makedirs('data',exist_ok=True)
            df.to_csv('data/steam_games.csv',index=False)
            print("Data saved to data/steam_games.csv")