import pandas as pd

# may also want to weight seasons (more recent seasons weighted higher, less recent lower. etc.)
seasons = ["20202021", "20212022", "20222023", "20232024", "20242025"]

TEAM_COLS = [
    "win_pct",
    "goal_diff_per_game",
    "shots_per_game",
    "shots_against_per_game",
    "pp_pct",
    "pk_pct",
]

ADV_COLS = [
    "corsi_pct",
    "xgf_per60",
    "xga_per60",
    "xgf_pct",
    "hdcf_per60",
    "hdca_per60",
    "goalie_sv_pct",
    "goalie_gsax",
]

ALL_TEAMS = ["ANA", "BOS", "BUF", "CAR", "CGY", "COL", "CBJ", "CHI", 
             "DAL", "DET", "EDM", "FLA", "LAK", "MTL", "MIN", "NYR", 
             "NYI", "NJD", "NSH", "OTT", "PHI", "PIT", "SEA", "SJS", 
             "STL", "TOR", "TBL", "UTA", "VAN", "VGK", "WSH", "WPG"]

"""
Weights features based on the given weights for each season.
"""
def weighted_team_features(full_df, team, seasons, weights, cols):
    df = full_df[full_df["team"] == team]

    result = {}

    for col in cols:
        total = 0
        wsum = 0

        for season in seasons:
            row = df[df["season"] == season]

            if row.empty or col not in row.columns:
                continue

            val = row.iloc[0].get(col, 0)

            if pd.isna(val):
                continue

            w = weights.get(season, 0)

            total += val * w
            wsum += w

        result[col] = total / wsum if wsum > 0 else 0

    return result

"""
Builds features for a given playoff matchup between team1 and team2, based on the full dataset of team/goalie stats, 
the dataset of playoff games, and the head-to-head history between the two teams.
"""
def build_features(team1, team2, full_df, playoff_df, seasons, weights):

    features = {
        "team1": team1,
        "team2": team2
    }

    """
    Add features regarding team strength based on regular season performance, weighted across multiple seasons. 
    For example, we can take a weighted average of each team's win percentage over the past 5 seasons, 
    with more recent seasons weighted more heavily. This is also done for goal differential per game, shots per game, 
    shots against per game, power play percentage and penalty kill percentage.
    """
    t1 = weighted_team_features(full_df, team1, seasons, weights, TEAM_COLS)
    t2 = weighted_team_features(full_df, team2, seasons, weights, TEAM_COLS)

    features.update({
        "win_pct_diff": t1["win_pct"] - t2["win_pct"],
        "goal_diff_per_game_diff": t1["goal_diff_per_game"] - t2["goal_diff_per_game"],
        "shots_diff": t1["shots_per_game"] - t2["shots_per_game"],
        "shots_against_diff": t1["shots_against_per_game"] - t2["shots_against_per_game"],
        "pp_diff": t1["pp_pct"] - t2["pp_pct"],
        "pk_diff": t1["pk_pct"] - t2["pk_pct"],
    })

    """
    Incorporate "advanced stats" that attempt to capture team quality beyond traditional box score stats.
    These include Corsi percentage (shot attempt differential), expected goals for and against per 60, 
    high danger chances for and against per 60, goalie save percentage, and goalie goals saved above expected.
    """
    t1_adv = weighted_team_features(full_df, team1, seasons, weights, ADV_COLS)
    t2_adv = weighted_team_features(full_df, team2, seasons, weights, ADV_COLS)

    features.update({
        "xgf_diff": t1_adv["xgf_per60"] - t2_adv["xgf_per60"],
        "xga_diff": t1_adv["xga_per60"] - t2_adv["xga_per60"],
        "xgf_pct_diff": t1_adv["xgf_pct"] - t2_adv["xgf_pct"],
        "corsi_diff": t1_adv["corsi_pct"] - t2_adv["corsi_pct"],
        "hd_diff": t1_adv["hdcf_per60"] - t2_adv["hdcf_per60"],
        "goalie_sv_pct_diff": t1_adv["goalie_sv_pct"] - t2_adv["goalie_sv_pct"],
        "goalie_gsax_diff": t1_adv["goalie_gsax"] - t2_adv["goalie_gsax"],
    })

    """
    Add features based on the playoff history between the two teams.
    This includes total playoff series played, playoff series wins for team1, and average games per series.
    """

    """
    This had to be amended due to many many issues with non-existent playoff data I couldn't fix
    """

    h2h_features = {
        "h2h_games": 0,
        "team1_h2h_wins": 0,
        "goal_diff": 0,
        "home_win_rate": 0.5,
        "away_win_rate": 0.5,
    }

    features.update(h2h_features)

    return features