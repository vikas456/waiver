"""Who will actually be on the field.

A projection is only worth as much as the chance a player plays. Free sources
carry most of that: nflverse's depth chart, which says who starts; its weekly
rosters and injury reports, which say who is hurt; and Sleeper's player list,
which catches a move to injured reserve within hours instead of at nflverse's
next weekly release. All describe the season in progress only.

Without this, a team's backup quarterback carries his old starter's form and
projects as a starter, and a receiver moved to injured reserve on Monday keeps
projecting as if he will play on Sunday.
"""

from __future__ import annotations

import pandas as pd

from . import sleeper
from .simulate import DESIGNATION_PLAY_PROB

# How often a backup quarterback plays in a given week. He only plays when
# the starter is hurt, which no one can schedule.
BACKUP_QB_PLAY = 0.08

# Share of players back on the field k weeks after a move to injured reserve,
# k = 0 being his first week on it. Measured on every in-season move in weeks
# 1-10 of 2022-2025 (613 players, all positions): nobody returns inside the
# four-game minimum, and about half never return that season. Injuries with
# enough cases get their own curve, pulled toward the overall one by 50 cases'
# worth. tools/return_curves.py re-measures them.
RETURN_CURVES = {
    "all":       (0, 0, 0, 0, .11, .21, .29, .36, .40, .43, .46, .47, .48, .48, .49, .50, .50, .51),
    "hamstring": (0, 0, 0, 0, .15, .30, .38, .51, .58, .60, .63, .64, .65, .65, .65, .66, .66, .66),
    "ankle":     (0, 0, 0, 0, .14, .25, .31, .34, .42, .45, .47, .49, .50, .51, .51, .51, .52, .52),
    "knee":      (0, 0, 0, 0, .08, .13, .22, .30, .35, .37, .41, .41, .41, .41, .42, .43, .43, .43),
    # Players who start the season physically unable to perform (119 cases).
    "pup":       (0, 0, 0, 0, .08, .13, .24, .29, .34, .39, .45, .50, .54, .56, .59, .64, .65, .67),
}

# How often a fantasy player played the next game after each designation,
# 2022-2025. Sleeper's designation can be last week's, standing until this
# week's report is out, so a designation from Sleeper alone is read as one.
CARRYOVER = {"out": 0.26, "doubtful": 0.39, "questionable": 0.69}

# Questionable covers everything from a knock to a player who has not
# practised all week, and what he did in practice splits it: measured on
# regulars, meaning players taking at least a third of their team's snaps that
# season, a questionable player who practised fully played about nine times in
# ten and one who sat out every session about half the time. Both halves of
# 2022-2025 agree (dnp 0.530 then 0.534, full 0.886 then 0.933, limited 0.730
# then 0.672), which is why these three are used and nothing else is: a player
# with no designation played 96% of the time whether he practised or not, and
# out and doubtful are already decided by the designation itself.
PRACTICE_PLAY = {
    ("questionable", "dnp"): 0.53,
    ("questionable", "limited"): 0.70,
    ("questionable", "full"): 0.90,
}

# A player on injured reserve who turns up on a practice report has had his
# 21-day window opened, which no return curve knows about. Share back on the
# field within one, two and three weeks of that first session, 2022-2025:
# about ten times the rate of a reserve player who is not practising.
WINDOW_RETURN = (0.20, 0.36, 0.45)

# How the injury report spells out what a player did in practice.
PRACTICE = {"Did Not Participate In Practice": "dnp",
            "Limited Participation in Practice": "limited",
            "Full Participation in Practice": "full"}

# Sleeper's designations, as the site names them.
LABELS = {"IR": "IR", "PUP": "PUP", "Out": "Out", "Doubtful": "Doubtful",
          "Questionable": "Questionable", "Sus": "Suspended", "NA": "Inactive",
          "DNR": "Did not report"}

RESERVE = ("ir", "pup")


def load(season: int) -> dict:
    """The latest depth chart, this season's injury reports and reserve lists,
    and Sleeper's current designations.

    Any of them can be missing early in the week or when a source is slow.
    Players then keep what the others say rather than the build failing.
    """
    import nflreadpy as nfl

    info: dict = {"starters": None, "qb_teams": set(), "reports": {}, "practice": {},
                  "hurt": {}, "reserve": {}, "news": None}
    try:
        dc = nfl.load_depth_charts([season]).to_pandas()
        dc["dt"] = pd.to_datetime(dc["dt"])
        qbs = dc[(dc["dt"] == dc["dt"].max()) & (dc["pos_abb"] == "QB")]
        info["starters"] = set(qbs.loc[qbs["pos_rank"] == 1, "gsis_id"].dropna())
        info["qb_teams"] = set(qbs["team"])
    except Exception as err:
        print(f"  depth chart unavailable ({type(err).__name__}); quarterbacks keep baseline availability")
    try:
        inj = nfl.load_injuries([season]).to_pandas().sort_values("week")
        rep = inj[inj["report_status"].notna()]
        info["reports"] = {(r.gsis_id, int(r.week)): str(r.report_status).lower()
                           for r in rep.itertuples()}
        prac = inj[inj["practice_status"].notna()]
        info["practice"] = {(r.gsis_id, int(r.week)): PRACTICE.get(str(r.practice_status), "none")
                            for r in prac.itertuples()}
        hurt = inj["report_primary_injury"].fillna(inj["practice_primary_injury"])
        info["hurt"] = dict(zip(inj.loc[hurt.notna(), "gsis_id"], hurt.dropna()))
    except Exception as err:
        print(f"  injury reports unavailable ({type(err).__name__}); designations default to healthy")
    try:
        info["reserve"] = _reserve_spells(nfl.load_rosters_weekly([season]).to_pandas())
    except Exception as err:
        print(f"  weekly rosters unavailable ({type(err).__name__}); reserve lists come from Sleeper alone")
    try:
        info["news"] = sleeper.statuses()
    except Exception as err:
        print(f"  Sleeper statuses unavailable ({type(err).__name__}); reserve lists come from nflverse alone")
    return info


def _reserve_spells(rosters: pd.DataFrame) -> dict[str, dict]:
    """Players on reserve in the latest weekly roster, and the week it began."""
    latest = rosters["week"].max()
    out = {}
    for pid, g in rosters.sort_values("week").groupby("gsis_id"):
        rows = list(zip(g["week"], g["status"], g["status_description_abbr"]))
        if rows[-1][0] != latest or rows[-1][1] != "RES":
            continue
        since = rows[-1][0]
        for week, status, _ in reversed(rows):
            if status != "RES":
                break
            since = week
        out[pid] = {"kind": "pup" if rows[-1][2] == "R04" else "ir", "since": int(since)}
    return out


def current(player_id: str, from_week: int, info: dict) -> dict | None:
    """What keeps a player off the field now, if anything: its kind ("ir",
    "pup" or a designation), the week a reserve spell began, a label for the
    site and the injury behind it."""
    spell = info["reserve"].get(player_id)
    hurt = info["hurt"].get(player_id, "")
    news = info.get("news")
    # Sleeper is fresher, so where it knows the player it has the last word,
    # including that he has come back since nflverse's last release.
    if news is not None and player_id in news:
        status, detail = news[player_id]["status"], news[player_id]["detail"] or hurt
        if status not in LABELS or detail == "Coach's Decision":
            return None
        kind = status.lower()
        since = spell["since"] if spell and kind in RESERVE else from_week
        return {"kind": kind, "since": since, "label": LABELS[status], "detail": detail}
    if spell:
        return {"kind": spell["kind"], "since": spell["since"], "label": spell["kind"].upper(),
                "detail": hurt}
    return None


def on_reserve(info: dict, from_week: int) -> set[str]:
    """Everyone currently on injured reserve or the PUP list."""
    ids = set(info["reserve"]) | set(info.get("news") or {})
    return {pid for pid in ids
            if (now := current(pid, from_week, info)) and now["kind"] in RESERVE}


def return_chance(now: dict, week: int, from_week: int) -> float:
    """Chance a player on reserve is back by the given week, knowing he is
    still out as of from_week."""
    if now["kind"] == "pup":
        curve = RETURN_CURVES["pup"]
    else:
        hurt = (now["detail"] or "").lower()
        curve = next((RETURN_CURVES[k] for k in ("hamstring", "ankle", "knee") if k in hurt),
                     RETURN_CURVES["all"])
    back_by = lambda k: 0.0 if k < 0 else curve[min(k, len(curve) - 1)]
    # Everyone who would have been back before now is ruled out already.
    gone = back_by(from_week - now["since"] - 1)
    # A curve that has everyone back by now says nothing about a player who
    # is still out; take him as due back rather than divide by zero.
    if gone >= 1.0:
        return 1.0
    return max(0.0, (back_by(week - now["since"]) - gone) / (1 - gone))


def by_week(player_id: str, position: str, team: str, base: float,
            weeks: list[int], from_week: int, info: dict) -> dict[int, float]:
    """Probability of playing in each of the given weeks."""
    starters = info.get("starters")
    backup_qb = (position == "QB" and starters is not None
                 and team in info["qb_teams"] and player_id not in starters)
    now = current(player_id, from_week, info)
    practice = info.get("practice", {})
    on_reserve_now = bool(now and now["kind"] in RESERVE)
    # A reserve player practising this week has had his 21-day window opened.
    window = on_reserve_now and practice.get((player_id, from_week)) in ("limited", "full")
    out = {}
    for week in weeks:
        p = min(base, BACKUP_QB_PLAY) if backup_qb else base
        if on_reserve_now:
            chance = return_chance(now, week, from_week)
            if window:
                # News about this player beats an average over hundreds of
                # reserve spells, and the curve never sees a window open.
                chance = max(chance, WINDOW_RETURN[min(week - from_week, len(WINDOW_RETURN) - 1)])
            p *= chance
        elif now and week == from_week:
            p = min(p, CARRYOVER.get(now["kind"], 0.0))
        report = info["reports"].get((player_id, week))
        did = practice.get((player_id, week))
        if not on_reserve_now and did and (report or "none", did) in PRACTICE_PLAY:
            # What he did in practice is the sharper signal, so where it is
            # known it stands in for the designation rather than joining it.
            p = min(p, PRACTICE_PLAY[(report or "none", did)])
        elif report in DESIGNATION_PLAY_PROB:
            p = min(p, DESIGNATION_PLAY_PROB[report])
        out[week] = round(p, 3)
    return out
