from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from src.features.build_features import build_features
import numpy as np
from collections import Counter
from functools import lru_cache

def train_logistic_regression(df):
    X = df.drop(columns=["team1", "team2", "team1_wins"])
    y = df["team1_wins"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Accuracy:", model.score(X_test, y_test))

    return model

def make_prob_function(model, full_df, playoff_games_df, seasons, weights):

    @lru_cache(maxsize=None)
    def get_matchup_prob(team1, team2):
        if team1 > team2:
            team1, team2 = team2, team1
            flip = True
        else:
            flip = False

        features = build_features(
            team1,
            team2,
            full_df,
            playoff_games_df,
            seasons,
            weights
        )

        row = pd.DataFrame([features])
        X = row.drop(columns=["team1", "team2"])

        p = model.predict_proba(X)[0][1]

        return 1 - p if flip else p

    return get_matchup_prob

def simulate_playoffs_dynamic(model, initial_matchups, full_df, playoff_games_df, seasons, weights):
    current_matchups = initial_matchups
    round_num = 1
    results = []

    while len(current_matchups) > 1:
        winners = []

        for team1, team2 in current_matchups:

            features = build_features(
                team1, team2, full_df, playoff_games_df, seasons, weights
            )

            row = pd.DataFrame([features])
            X = row.drop(columns=["team1", "team2"])

            prob = model.predict_proba(X)[0][1]

            winner = team1 if np.random.rand() < prob else team2

            results.append({
                "round": round_num,
                "team1": team1,
                "team2": team2,
                "prob_team1_win": prob,
                "winner": winner
            })

            winners.append(winner)

        current_matchups = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]
        round_num += 1

    champion = current_matchups[0][0]

    return results, champion

"""
Monte Carlo simulation to get probabilities of winning the Stanley Cup, with checkpoints along the way for graphing.
"""
def monte_carlo_progression(initial_matchups, prob_fn, checkpoints):
    champions = []
    results = {}

    max_sims = max(checkpoints)

    for i in range(1, max_sims + 1):
        current = initial_matchups

        while len(current) > 1:
            winners = []
            for team1, team2 in current:
                p = prob_fn(team1, team2)
                winner = team1 if np.random.rand() < p else team2
                winners.append(winner)

            current = [(winners[j], winners[j+1]) for j in range(0, len(winners), 2)]

        champions.append(current[0][0])

        if i in checkpoints:
            counts = Counter(champions)
            total = sum(counts.values())
            probs = {team: counts[team] / total for team in counts}
            results[i] = probs

    return results

"""
Monte Carlo simulation of the full playoff bracket, returning counts of how many times each team won in each round and overall.
"""
def monte_carlo_full_bracket(east_matchups, west_matchups, prob_fn, n_sims=5000):
    import numpy as np
    from collections import Counter

    east_winners = []
    west_winners = []
    champions = []

    for _ in range(n_sims):

        def run_bracket(matchups):
            current = matchups
            while len(current) > 1:
                winners = []
                for t1, t2 in current:
                    p = prob_fn(t1, t2)
                    winner = t1 if np.random.rand() < p else t2
                    winners.append(winner)
                current = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]
            return current[0][0]

        east = run_bracket(east_matchups)
        west = run_bracket(west_matchups)

        p_final = prob_fn(east, west)
        champ = east if np.random.rand() < p_final else west

        east_winners.append(east)
        west_winners.append(west)
        champions.append(champ)

    return Counter(east_winners), Counter(west_winners), Counter(champions)