"""Who will actually be on the field.

A projection is only worth as much as the chance a player plays. Two free
nflverse sources carry most of that: the depth chart, which says who starts,
and the weekly injury report, which says who might not. Both are read for the
season in progress only.

Without this, a team's backup quarterback carries his old starter's form and
projects as a starter, and two quarterbacks on the same team both rank among
the league's best.
"""

from __future__ import annotations

import pandas as pd

from .simulate import DESIGNATION_PLAY_PROB

# How often a backup quarterback plays in a given week. He only plays when
# the starter is hurt, which no one can schedule.
BACKUP_QB_PLAY = 0.08

# Injured reserve keeps a player out for at least this many games.
RESERVE_WEEKS = 4


def load(season: int) -> dict:
    """The latest depth chart and this season's injury reports.

    Either can be missing early in the week or when nflverse is slow. Players
    then keep their baseline availability rather than the build failing.
    """
    import nflreadpy as nfl

    info: dict = {"starters": None, "qb_teams": set(), "reports": {}}
    try:
        dc = nfl.load_depth_charts([season]).to_pandas()
        dc["dt"] = pd.to_datetime(dc["dt"])
        qbs = dc[(dc["dt"] == dc["dt"].max()) & (dc["pos_abb"] == "QB")]
        info["starters"] = set(qbs.loc[qbs["pos_rank"] == 1, "gsis_id"].dropna())
        info["qb_teams"] = set(qbs["team"])
    except Exception as err:
        print(f"  depth chart unavailable ({type(err).__name__}); quarterbacks keep baseline availability")
    try:
        inj = nfl.load_injuries([season]).to_pandas()
        inj = inj[inj["report_status"].notna()]
        info["reports"] = {(r.gsis_id, int(r.week)): str(r.report_status).lower()
                           for r in inj.itertuples()}
    except Exception as err:
        print(f"  injury reports unavailable ({type(err).__name__}); designations default to healthy")
    return info


def by_week(player_id: str, position: str, team: str, status, base: float,
            weeks: list[int], from_week: int, info: dict) -> dict[int, float]:
    """Probability of playing in each of the given weeks."""
    starters = info.get("starters")
    backup_qb = (position == "QB" and starters is not None
                 and team in info["qb_teams"] and player_id not in starters)
    out = {}
    for week in weeks:
        p = min(base, BACKUP_QB_PLAY) if backup_qb else base
        if status == "RES" and week < from_week + RESERVE_WEEKS:
            p = 0.0
        report = info["reports"].get((player_id, week))
        if report in DESIGNATION_PLAY_PROB:
            p = min(p, DESIGNATION_PLAY_PROB[report])
        out[week] = round(p, 3)
    return out
