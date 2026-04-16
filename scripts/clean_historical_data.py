import pandas as pd
import os

"""
This script only needs to run once. It transforms the data into a neater form that only has the years we want.
"""
path = './NHL_Playoff_Data_1986_2025/games_boxscores_playoffs.csv'
print(os.path.exists(path))
print(os.listdir('./NHL_Playoff_Data_1986_2025'))

games_bs = pd.read_csv('NHL_Playoff_Data_1986_2025/game_boxscores_playoffs.csv')
player_bs = pd.read_csv('NHL_Playoff_Data_1986_2025/player_boxscores_playoffs.csv')

# clean any data before 2020, as this is when the playoff format changed to its current state
games_bs['gameId'] = games_bs['gameId'].astype(str)
games_bs['game_year'] = games_bs['gameId'].str[:4].astype(int)

player_bs['gameId'] = player_bs['gameId'].astype(str)
player_bs['game_year'] = player_bs['gameId'].str[:4].astype(int)

filtered_games_bs = games_bs[games_bs['game_year'] >= 2020]
filtered_player_bs = player_bs[player_bs['game_year'] >= 2020]

filtered_games_bs.to_csv('./cleaned_data/cleaned_game_boxscores_playoffs.csv', index=False)
filtered_player_bs.to_csv('./cleaned_data/cleaned_player_boxscores_playoffs.csv', index=False)

print(filtered_games_bs.head())
