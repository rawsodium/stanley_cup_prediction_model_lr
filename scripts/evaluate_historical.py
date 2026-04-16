import pandas as pd
from nhlpy import NHLClient

from src.processing.build_full_dataset import build_full_dataset
from src.processing.build_training_dataset import build_training_dataset
from src.processing.matchups import extract_first_round_matchups
from src.models.logistic_regression_model import (
    train_logistic_regression,
    simulate_playoffs_dynamic
)
from src.processing.team_stats import SEASONS, SEASON_WEIGHTS
import matplotlib.pyplot as plt
from src.processing.team_stats import TEAM_NAME_TO_ABBR

teams = list(TEAM_NAME_TO_ABBR.values())

years = {
    "2021": "20202021",
    "2022": "20212022",
    "2023": "20222023",
    "2024": "20232024",
    "2025": "20242025"
}

actual_winners = {
    "2021": "TBL",
    "2022": "COL",
    "2023": "VGK",
    "2024": "FLA",
    "2025": "FLA"
}

client = NHLClient()

results = []

for year, season_cutoff in years.items():

    print(f"\nEvaluating {year}...")

    # restrict seasons to avoid leakage
    valid_seasons = [s for s in SEASONS if s <= season_cutoff]

    # build dataset
    full_df = build_full_dataset(
        teams=teams,
        seasons=valid_seasons,
        weights=SEASON_WEIGHTS
    )

    playoff_games_df = pd.read_csv(
        "cleaned_data/cleaned_game_boxscores_playoffs.csv"
    )

    training_df = build_training_dataset(
        full_df,
        playoff_games_df,
        valid_seasons,
        SEASON_WEIGHTS
    )

    model = train_logistic_regression(training_df)

    bracket = client.schedule.playoff_bracket(year)
    matchups = extract_first_round_matchups(bracket)

    if not matchups:
        print("No matchups found")
        continue

    _, predicted = simulate_playoffs_dynamic(
        model,
        matchups,
        full_df,
        playoff_games_df,
        valid_seasons,
        SEASON_WEIGHTS
    )

    actual = actual_winners.get(year)

    results.append({
        "year": year,
        "predicted": predicted,
        "actual": actual,
        "correct": predicted == actual
    })

df = pd.DataFrame(results)

print(df)
print("Accuracy:", df["correct"].mean())

plt.bar(df["year"], df["correct"].astype(int))
plt.title("Stanley Cup Prediction Accuracy (Backtest)")
plt.ylabel("Correct (1 = yes)")
plt.show()
