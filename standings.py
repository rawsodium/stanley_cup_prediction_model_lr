import requests
import pandas as pd

# cached version
def get_standings(date):

    from src.utils.cache import get_or_create

    key = f"standings_{date}"

    def build():
        return get_standings_from_api(date)

    return get_or_create(key, build)

# gets the NHL standings for the current date
def get_standings_from_api(date):
    url = f"https://api-web.nhle.com/v1/standings/{date}"
    response = requests.get(url)
    data = response.json()

    teams = []

    for team in data["standings"]:
        teams.append({
            "team": team["teamAbbrev"]["default"],
            "conference": team["conferenceName"],
            "division": team["divisionName"],
            "points": team["points"],
            "wins": team["wins"],
            "losses": team["losses"],
            "ot_losses": team["otLosses"],
            "goal_diff": team["goalDifferential"],

            "season": str(team["seasonId"]),

            "games_played": team["gamesPlayed"],
            "goals_for": team["goalFor"],
            "goals_against": team["goalAgainst"],
            "win_pct": team["winPctg"],
            "point_pct": team["pointPctg"],
        })

    return pd.DataFrame(teams)