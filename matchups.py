def extract_first_round_matchups(bracket):
    matchups = []

    for series in bracket.get("series", []):
        # only first round
        if series.get("playoffRound") != 1:
            continue

        team1 = series.get("topSeedTeam", {}).get("abbrev")
        team2 = series.get("bottomSeedTeam", {}).get("abbrev")

        if team1 and team2:
            matchups.append((team1, team2))

    return matchups