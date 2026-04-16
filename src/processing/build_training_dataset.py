from src.features.build_features import build_features
import pandas as pd
from src.processing.history import add_series_id, add_winner

def build_training_dataset(full_df, playoff_games_df, seasons, weights):
    rows = []

    # group playoff games into series
    playoff_games_df = add_series_id(playoff_games_df)
    playoff_games_df = add_winner(playoff_games_df)

    for series_id, group in playoff_games_df.groupby("series_id"):

        teams = list(set(group["homeAbbrev"]).union(set(group["awayAbbrev"])))
        if len(teams) != 2:
            continue

        team1, team2 = sorted(teams)  # keep consistent ordering

        features = build_features(
            team1,
            team2,
            full_df,
            playoff_games_df,
            seasons,
            weights
        )

        wins = group["winner"].value_counts()

        team1_wins = wins.get(team1, 0)
        team2_wins = wins.get(team2, 0)

        features["team1_wins"] = 1 if team1_wins > team2_wins else 0

        rows.append(features)

    return pd.DataFrame(rows)
