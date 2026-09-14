"""Re-measure how long injured players stay out, for pipeline/availability.py.

    python tools/return_curves.py [season ...]

Prints RETURN_CURVES, the share of players back k weeks after a move to
injured reserve, by injury, and CARRYOVER, how often a fantasy player listed
on an injury report plays the following game. Both come from nflverse's
weekly rosters, injury reports and stats, 2022-2025 unless seasons are given.
"""

from __future__ import annotations

import sys

import nflreadpy as nfl
import numpy as np
import pandas as pd

SEASONS = [2022, 2023, 2024, 2025]
WEEKS = 18
# Cases' worth of pull toward the all-injury curve, for injuries with few moves.
PRIOR = 50
INJURIES = ("hamstring", "ankle", "knee")


def main() -> None:
    seasons = [int(s) for s in sys.argv[1:]] or SEASONS
    rosters = nfl.load_rosters_weekly(seasons).to_pandas()
    inj = nfl.load_injuries(seasons).to_pandas()
    inj["hurt"] = inj["report_primary_injury"].fillna(inj["practice_primary_injury"])
    hurt = inj.dropna(subset=["hurt"])

    moves, pup = [], []
    for (pid, season), g in rosters.sort_values("week").groupby(["gsis_id", "season"]):
        s = list(zip(g["week"], g["status"], g["status_description_abbr"]))
        if s[0][0] == 1 and s[0][1] == "RES" and s[0][2] == "R04":
            back = next((w for w, st, _ in s if st == "ACT"), None)
            pup.append(back - 1 if back else 99)
        for i in range(1, len(s)):
            w0, st, code = s[i]
            if s[i - 1][1] == "ACT" and st == "RES" and code in ("R01", "R48") and w0 <= 10:
                back = next((w for w, later, _ in s[i + 1:] if later == "ACT"), None)
                h = hurt[(hurt.gsis_id == pid) & (hurt.season == season) & hurt.week.between(w0 - 3, w0)]
                kind = str(h["hurt"].iloc[-1]).lower() if len(h) else ""
                moves.append((back - w0 if back else 99, next((k for k in INJURIES if k in kind), "other")))
                break

    df = pd.DataFrame(moves, columns=["missed", "hurt"])
    curve = lambda m: np.array([(np.asarray(m) <= k).mean() for k in range(WEEKS)])
    fmt = lambda c: "(" + ", ".join("0" if v == 0 else f"{v:.2f}".lstrip("0") for v in c) + ")"
    base = curve(df.missed)
    print(f"# {len(df)} moves to injured reserve; {len(pup)} seasons started on PUP")
    print("RETURN_CURVES = {")
    print(f'    "all": {fmt(base)},')
    for k in INJURIES:
        m = df.missed[df.hurt == k]
        print(f'    "{k}": {fmt((len(m) * curve(m) + PRIOR * base) / (len(m) + PRIOR))},  # {len(m)} cases')
    print(f'    "pup": {fmt(curve(pup))},')
    print("}")

    stats = nfl.load_player_stats(seasons).to_pandas()
    stats = stats[stats["season_type"] == "REG"]
    played = set(zip(stats.player_id, stats.season, stats.week))
    games = set(zip(stats.team, stats.season, stats.week))
    rep = inj[(inj.game_type == "REG") & inj.position.isin(["QB", "RB", "WR", "TE"])
              & inj.report_status.notna() & (inj.week <= 17)]
    carry = {}
    for status in ("Out", "Doubtful", "Questionable"):
        sub = rep[rep.report_status == status]
        nxt = [(g, s, w + 1) for g, s, w, t in zip(sub.gsis_id, sub.season, sub.week, sub.team)
               if (t, s, w + 1) in games]
        carry[status.lower()] = round(sum(k in played for k in nxt) / len(nxt), 2)
    print(f"CARRYOVER = {carry}")


if __name__ == "__main__":
    main()
