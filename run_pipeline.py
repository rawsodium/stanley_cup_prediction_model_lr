from datetime import datetime
import pandas as pd
from nhlpy import NHLClient
from src.models.logistic_regression_model import make_prob_function, monte_carlo_full_bracket, monte_carlo_progression, train_logistic_regression, simulate_playoffs_dynamic
from src.processing.build_training_dataset import build_training_dataset
from src.processing.standings import get_standings
from src.processing.matchups import extract_first_round_matchups
from src.processing.build_full_dataset import build_full_dataset
from src.processing.team_stats import (
    SEASONS,
    SEASON_WEIGHTS
)
from src.features.build_features import build_features
import matplotlib.pyplot as plt
import numpy as np

# instantiate NHL client
client = NHLClient()

# get today's date for timestamping outputs
today = datetime.today().strftime("%Y-%m-%d")

# load playoff history data
playoff_games_df = pd.read_csv(
    "cleaned_data/cleaned_game_boxscores_playoffs.csv"
)

# 1. get current standings
standings_df = get_standings(today)
standings_df["season"] = standings_df["season"].astype(str)
standings_df.to_csv(f"outputs/current_standings_{today}.csv", index=False)

teams = standings_df["team"].unique()

# 2. playoff bracket/matchups
bracket = client.schedule.playoff_bracket("2026")
matchups = extract_first_round_matchups(bracket)

if not matchups:
    raise ValueError("No playoff matchups found — bracket likely empty or wrong season")

# 3. build multi-season datasets
full_df = build_full_dataset(teams, SEASONS, SEASON_WEIGHTS)

required = ["wins", "games_played", "goals_for", "goals_against", "season"]

for col in required:
    if col not in full_df.columns:
        print(f"Missing column: {col}")

full_df.to_csv("outputs/full_dataset.csv", index=False)

# 4. build features for model training
features = []

for team1, team2 in matchups:
    row = build_features(
        team1,
        team2,
        full_df,
        playoff_games_df,
        SEASONS,
        SEASON_WEIGHTS
    )
    features.append(row)

features_df = pd.DataFrame(features)
features_df.to_csv(f"outputs/model_input_{today}.csv", index=False)

# 5. build training dataset and train model
training_df = build_training_dataset(full_df, playoff_games_df, SEASONS, SEASON_WEIGHTS)
model = train_logistic_regression(training_df)

east_teams = set(standings_df[standings_df["conference"] == "Eastern"]["team"])
west_teams = set(standings_df[standings_df["conference"] == "Western"]["team"])

east_matchups = []
west_matchups = []

for t1, t2 in matchups:
    if t1 in east_teams:
        east_matchups.append((t1, t2))
    else:
        west_matchups.append((t1, t2))

# 6. simulate playoffs and predict champion
prob_fn = make_prob_function(
    model,
    full_df,
    playoff_games_df,
    SEASONS,
    SEASON_WEIGHTS
)

# east and west results
east_results, east_champ = simulate_playoffs_dynamic(
    model, east_matchups, full_df, playoff_games_df, SEASONS, SEASON_WEIGHTS
)

west_results, west_champ = simulate_playoffs_dynamic(
    model, west_matchups, full_df, playoff_games_df, SEASONS, SEASON_WEIGHTS
)

final_features = build_features(
    east_champ,
    west_champ,
    full_df,
    playoff_games_df,
    SEASONS,
    SEASON_WEIGHTS
)

row = pd.DataFrame([final_features])
X = row.drop(columns=["team1", "team2"])

prob = model.predict_proba(X)[0][1]

cup_winner = east_champ if np.random.rand() < prob else west_champ

print("EAST CHAMP:", east_champ)
print("WEST CHAMP:", west_champ)
print("STANLEY CUP WINNER:", cup_winner)

# 7. Evaluation.

checkpoints = [100, 500, 1000, 5000, 10000]
progression = monte_carlo_progression(matchups, prob_fn, checkpoints)

plt.figure(figsize=(10,6))

final_probs = progression[checkpoints[-1]]
top_teams = sorted(final_probs, key=final_probs.get, reverse=True)[:5]

for team in top_teams:
    y = [progression[c].get(team, 0) for c in checkpoints]
    plt.plot(checkpoints, y, marker='o', label=team)

plt.xlabel("Number of Simulations")
plt.ylabel("Estimated Probability")
plt.title("Monte Carlo Convergence (Top 5 Teams)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


east_counts, west_counts, cup_counts = monte_carlo_full_bracket(
    east_matchups,
    west_matchups,
    prob_fn,
    n_sims=10000
)

def plot_combined(east_counts, west_counts):
    combined = {}

    combined.update({f"{k} (E)": v for k, v in east_counts.items()})
    combined.update({f"{k} (W)": v for k, v in west_counts.items()})

    total = sum(combined.values())

    sorted_items = sorted(combined.items(), key=lambda x: x[1])

    teams = [t for t, _ in sorted_items]
    probs = [c / total for _, c in sorted_items]

    plt.figure(figsize=(10,6))
    plt.barh(teams, probs)

    plt.title("Conference Winner Probabilities (East vs West)")
    plt.xlabel("Probability")

    for i, p in enumerate(probs):
        plt.text(p, i, f"{p:.2f}", va='center')

    plt.tight_layout()
    plt.show()

plot_combined(east_counts, west_counts)

plt.bar(cup_counts.keys(), cup_counts.values(), color="green")
plt.xticks(rotation=45)
plt.title("Stanley Cup Win Probabilities")
plt.show()

