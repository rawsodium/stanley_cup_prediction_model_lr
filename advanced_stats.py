import pandas as pd
import requests
from src.utils.cache import get_or_create
from io import StringIO
import time

# spoof browser requests, essentially
def read_moneypuck_csv(url, retries=3, delay=2):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": "https://moneypuck.com/",
    }

    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            return pd.read_csv(StringIO(r.text))
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise e

def format_season_mp(season: str):
    return season[:4]

def get_moneypuck_team_stats(season):
    key = f"moneypuck_team_{season}"

    def build():
        season_year = season[:4]
        url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season_year}/regular/teams.csv"

        df = read_moneypuck_csv(url)

        df = df[df["situation"] == "all"].copy()

        df.columns = df.columns.str.strip()
        df["team"] = df["team"].str.upper()
        df["season"] = season

        return df

    return get_or_create(key, build)

def get_moneypuck_goalie_stats(season):
    key = f"moneypuck_goalie_{season}"

    def build():
        season_year = season[:4]
        url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season_year}/regular/goalies.csv"

        df = read_moneypuck_csv(url)

        df.columns = df.columns.str.strip()
        df["team"] = df["team"].str.upper()
        df["season"] = season

        # safety numeric conversion
        for col in ["icetime", "xGoals", "goals", "ongoal"]:
            if col not in df.columns:
                df[col] = 0

        df["gsax"] = df["xGoals"] - df["goals"]

        if df["ongoal"].sum() > 0:
            df["save_pct"] = 1 - (df["goals"] / df["ongoal"].replace(0, pd.NA))
        else:
            df["save_pct"] = 0

        grouped = df.groupby("team")

        rows = []

        for team, g in grouped:
            total_time = g["icetime"].sum()
            if total_time == 0:
                continue

            g = g.copy()
            g["w"] = g["icetime"] / total_time

            rows.append({
                "team": team,
                "season": season,
                "goalie_gsax": (g["gsax"] * g["w"]).sum(),
                "goalie_sv_pct": (g["save_pct"] * g["w"]).sum(),
            })

        return pd.DataFrame(rows)

    return get_or_create(key, build)

def build_moneypuck_dataset(seasons):
    frames = []

    for season in seasons:
        team = get_moneypuck_team_stats(season)
        goalie = get_moneypuck_goalie_stats(season)

        team["team"] = team["team"].str.upper()
        team["season"] = season

        goalie["team"] = goalie["team"].str.upper()
        goalie["season"] = season

        df = team.merge(goalie, on=["team", "season"], how="left")

        frames.append(df)

    return pd.concat(frames, ignore_index=True)