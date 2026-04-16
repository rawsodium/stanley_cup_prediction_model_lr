import pandas as pd
import requests

def clean_games(df):
    df = df.copy()
    df["homeGoals"] = pd.to_numeric(df["homeGoals"], errors="coerce")
    df["awayGoals"] = pd.to_numeric(df["awayGoals"], errors="coerce")
    df = df.dropna(subset=["homeGoals", "awayGoals"])
    return df

# REGULAR SEASON H2H
def regular_season_h2h(df, team1, team2):
    df = df.copy()

    # ensure numeric safety
    df = df.dropna(subset=["homeGoals", "awayGoals"])

    df["homeGoals"] = df["homeGoals"].astype(int)
    df["awayGoals"] = df["awayGoals"].astype(int)

    mask = (
        ((df["homeAbbrev"] == team1) & (df["awayAbbrev"] == team2)) |
        ((df["homeAbbrev"] == team2) & (df["awayAbbrev"] == team1))
    )

    games = df.loc[mask].copy()

    if games.empty:
        return games

    games["winner"] = games.apply(
        lambda r: r["homeAbbrev"] if r["homeGoals"] > r["awayGoals"] else r["awayAbbrev"],
        axis=1
    )

    return games

# PLAYOFF H2H
def playoff_h2h(df, team1, team2):
    if df.empty:
        return pd.DataFrame()

    games = df[
        (
            ((df["homeAbbrev"] == team1) & (df["awayAbbrev"] == team2)) |
            ((df["homeAbbrev"] == team2) & (df["awayAbbrev"] == team1))
        )
    ].copy()

    if games.empty:
        return games

    games["winner"] = games.apply(
        lambda r: r["homeAbbrev"] if r["homeGoals"] > r["awayGoals"] else r["awayAbbrev"],
        axis=1
    )

    return games

def get_team_schedule(team, season):
    url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
    r = requests.get(url, timeout=10)

    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json()

    games = []

    for g in data.get("games", []):
        games.append({
            "gameId": g.get("id"),
            "season": season,
            "homeAbbrev": g["homeTeam"]["abbrev"],
            "awayAbbrev": g["awayTeam"]["abbrev"],
            "homeGoals": g.get("homeScore"),
            "awayGoals": g.get("awayScore"),
        })

    return pd.DataFrame(games)

def canonicalize_games(df):
    df = df.copy()

    df["teamA"] = df[["homeAbbrev", "awayAbbrev"]].min(axis=1)
    df["teamB"] = df[["homeAbbrev", "awayAbbrev"]].max(axis=1)

    return df

def summarize_series(df, team1, team2):
    if df.empty:
        return {
            "series_played": 0,
            "team1_series_wins": 0,
            "team2_series_wins": 0,
            "avg_games_per_series": 0
        }

    # derive a fake series key from playoff structure
    df = df.copy()
    df["series_id"] = df["gameId"].astype(str).str[:-1]

    series_stats = []

    for sid, g in df.groupby("series_id"):
        wins = g["winner"].value_counts()

        t1 = wins.get(team1, 0)
        t2 = wins.get(team2, 0)

        series_stats.append({
            "series_id": sid,
            "winner": team1 if t1 > t2 else team2,
            "games": len(g)
        })

    series_df = pd.DataFrame(series_stats)

    return {
        "series_played": len(series_df),
        "team1_series_wins": (series_df["winner"] == team1).sum(),
        "team2_series_wins": (series_df["winner"] == team2).sum(),
        "avg_games_per_series": series_df["games"].mean()
    }

def add_series_id(df):
    """
    Adds a series_id column by trimming gameId.
    """
    df = df.copy()
    df['series_id'] = df['gameId'].astype(str).str[:-1]
    return df

def add_winner(df):
    df = df.copy()

    df['winner'] = df.apply(
        lambda row: row['homeAbbrev']
        if row['homeGoals'] > row['awayGoals']
        else row['awayAbbrev'],
        axis=1
    )

    return df
