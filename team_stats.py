from nhlpy import NHLClient
import pandas as pd
from src.utils.cache import get_or_create

client = NHLClient()

SEASONS = [
    "20202021",
    "20212022",
    "20222023",
    "20232024",
    "20242025",
    "20252026"
]

SEASON_WEIGHTS = {
    "20252026": 1.0,
    "20242025": 0.8,
    "20232024": 0.6,
    "20222023": 0.4,
    "20212022": 0.2,
    "20202021": 0.1
}

TEAM_ACTIVE_FROM = {
    "SEA": "20212022",
    "UTA": "20242025",
}

TEAM_NAME_TO_ABBR = {
    "Anaheim Ducks": "ANA",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Carolina Hurricanes": "CAR",
    "Calgary Flames": "CGY",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Chicago Blackhawks": "CHI",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Montreal Canadiens": "MTL",
    "Minnesota Wild": "MIN",
    "New York Rangers": "NYR",
    "New York Islanders": "NYI",
    "New Jersey Devils": "NJD",
    "Nashville Predators": "NSH",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "Seattle Kraken": "SEA",
    "San Jose Sharks": "SJS",
    "St. Louis Blues": "STL",
    "Toronto Maple Leafs": "TOR",
    "Tampa Bay Lightning": "TBL",
    "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}

"""
SEA started in 2021-2022, so should have 0 stats for 2020-2021 and be excluded from any season-based features for that season.
ARI had its final season in 2023-2024, and was relocated to UTA, but the two are not considered the same franchise.
UTA starts in 2024-2025, so should have 0 stats for all prior seasons and be excluded from season-based features for those seasons.
"""

def is_team_active(team, season):
    start = TEAM_ACTIVE_FROM.get(team, "00000000")
    return season >= start

def clean_team_column(df):
    df = df.copy()

    df["team"] = df["team"].astype(str).str.split(",").str[0]

    return df

# NHL API pulls for team statistics

def get_team_stats_season(season):
    key = f"nhl_team_summary_{season}"

    def build():
        data = client.stats.team_summary(
            start_season=season,
            end_season=season
        )
        return pd.DataFrame(data)

    df = get_or_create(key, build)

    if df is None or df.empty:
        return pd.DataFrame()

    # rename AFTER cache load
    df = df.rename(columns={
        "teamAbbrevs": "team",
        "gamesPlayed": "games_played",
        "wins": "wins",
        "losses": "losses",
        "otLosses": "ot_losses",
        "points": "points",
        "goalsFor": "goals_for",
        "goalsAgainst": "goals_against",
        "powerPlayPct": "pp_pct",
        "penaltyKillPct": "pk_pct",
        "shotsForPerGame": "shots_per_game",
        "shotsAgainstPerGame": "shots_against_per_game"
    })

    df["team"] = df["teamFullName"].map(TEAM_NAME_TO_ABBR)

    df["season"] = season

    return df

def build_team_stats_dataset(teams, seasons):
    frames = []

    for season in seasons:
        df = get_team_stats_season(season)

        if df.empty:
            continue

        # keep only valid teams if provided
        if teams is not None:
            df = df[df["team"].isin(teams)]

        # derived features
        df["games_played"] = df["games_played"].replace(0, 1)
        df["win_pct"] = df["wins"] / df["games_played"]
        df["goal_diff_per_game"] = (
            (df["goals_for"] - df["goals_against"]) /
            df["games_played"]
        )

        frames.append(df)

    return pd.concat(frames, ignore_index=True)