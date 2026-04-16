import pandas as pd
from src.processing.team_stats import build_team_stats_dataset
from src.processing.advanced_stats import build_moneypuck_dataset

TEAM_COLS = [
    "win_pct",
    "goal_diff_per_game",
    "pp_pct",
    "pk_pct",
]

MP_COLS = [
    "corsi_pct",
    "xgf",
    "xga",
    "gf",
    "ga",
    "shots_for",
    "shots_against",
    "goalie_sv_pct",
    "goalie_gsax",
    "xgf_pct",
]

COLUMN_MAP = {
    # shots
    "shotsOnGoalFor": "shots_for",
    "shotsOnGoalAgainst": "shots_against",

    # goals
    "goalsFor": "gf",
    "goalsAgainst": "ga",

    # expected goals
    "xGoalsFor": "xgf",
    "xGoalsAgainst": "xga",

    # corsi
    "corsiPercentage": "corsi_pct",

    # high danger
    "highDangerShotsFor": "hdcf",
    "highDangerShotsAgainst": "hdca",
}

def normalize_season(df):
    df = df.copy()
    df["season"] = df["season"].astype(str)

    # force consistent format if needed
    df["season"] = df["season"].str.replace("-", "")

    return df

def add_derived_features(df):
    df = df.copy()

    # games
    df["games"] = df["games_played"].replace(0, 1)

    df["shots_per_game"] = df["shots_for"] / df["games"]
    df["shots_against_per_game"] = df["shots_against"] / df["games"]

    # advanced stats
    if "iceTime" in df.columns:
        df["xgf_per60"] = df.get("xgf", 0) / df["iceTime"] * 60
        df["xga_per60"] = df.get("xga", 0) / df["iceTime"] * 60
        df["hdcf_per60"] = df.get("hdcf", 0) / df["iceTime"] * 60
        df["hdca_per60"] = df.get("hdca", 0) / df["iceTime"] * 60
    else:
        df["xgf_per60"] = 0
        df["xga_per60"] = 0
        df["hdcf_per60"] = 0
        df["hdca_per60"] = 0

    # ratios
    df["xgf_pct"] = df["xgf"] / (df["xgf"] + df["xga"]).replace(0, 1)

    return df

def build_full_dataset(teams, seasons, weights):
    team_df = normalize_season(build_team_stats_dataset(teams, seasons))
    mp_df = build_moneypuck_dataset(seasons)
    mp_df = normalize_season(mp_df)

    merged = pd.merge(team_df, mp_df, on=["team","season"], how="inner")
    merged = merged.rename(columns=COLUMN_MAP)

    merged["games_played"] = merged["games_played_x"]
    merged = merged.drop(columns=["games_played_x"], errors="ignore")

    merged = add_derived_features(merged)
    merged["games"] = merged["games_played"].replace(0, 1)

    merged["goals_for_per_game"] = merged["goals_for"] / merged["games_played"]
    merged["goals_against_per_game"] = merged["goals_against"] / merged["games_played"]

    assert "season" in team_df.columns, "team_df missing season"
    assert "season" in mp_df.columns, "nst_df missing season"

    rows = []

    for col in ["goalie_sv_pct", "goalie_gsax"]:
        if col not in merged.columns:
            merged[col] = 0

    for team in teams:
        df = merged[merged["team"] == team]

        row = {"team": team}

        for col in TEAM_COLS + MP_COLS:
            total = 0
            wsum = 0

            for season in seasons:
                s = df[df["season"] == season]
                if s.empty:
                    continue

                val = s[col].iloc[0] if (col in s.columns and not s.empty) else 0

                if pd.isna(val):
                    val = 0
                w = weights.get(season, 0)

                total += val * w
                wsum += w

            row[col] = total / wsum if wsum > 0 else 0

        rows.append(row)


    REQUIRED_COLS = [
        "team", "season",
        "win_pct", "goal_diff_per_game",
        "shots_per_game", "shots_against_per_game",
        "corsi_pct",
        "xgf_per60", "xga_per60",
        "hdcf_per60", "hdca_per60",
        "goalie_sv_pct", "goalie_gsax",
        "xgf_pct"
    ]

    missing = [c for c in REQUIRED_COLS if c not in merged.columns]

    if missing:
        raise ValueError(f"Missing columns in full_df: {missing}")
    return merged
