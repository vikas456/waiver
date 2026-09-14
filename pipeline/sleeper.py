"""Sleeper's public API, for news nflverse has not caught up with yet.

nflverse publishes rosters and injury reports in weekly batches, so a player
moved to injured reserve on Monday is still listed as active until its next
release. Sleeper's player list updates within hours. Sleeper asks that the
list be fetched at most once a day, which a build three times a week respects.
"""

from __future__ import annotations

import functools
import json
import urllib.request

PLAYERS = "https://api.sleeper.app/v1/players/nfl"
TRENDING = "https://api.sleeper.app/v1/players/nfl/trending/add?lookback_hours=48&limit=50"


def _get(url: str, timeout: int = 60):
    with urllib.request.urlopen(url, timeout=timeout) as res:
        return json.load(res)


@functools.cache
def to_gsis() -> dict[str, str]:
    """Sleeper player id to the gsis id the rest of the pipeline uses."""
    import nflreadpy as nfl

    ids = nfl.load_ff_playerids().to_pandas()[["sleeper_id", "gsis_id"]].dropna()
    sleeper = ids["sleeper_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return dict(zip(sleeper, ids["gsis_id"]))


def statuses() -> dict[str, dict]:
    """Every rostered player Sleeper knows, by gsis id, with his current
    designation (None when healthy) and the injury behind it."""
    ids = to_gsis()
    out = {}
    for sid, p in _get(PLAYERS).items():
        gsis = ids.get(sid) or (p.get("gsis_id") or "").strip()
        if gsis and p.get("team"):
            out[gsis] = {"status": p.get("injury_status"), "detail": p.get("injury_body_part") or ""}
    return out


def trending_adds() -> list[tuple[str, int]]:
    """(gsis id, adds) for the players most added over the last two days."""
    ids = to_gsis()
    return [(ids[a["player_id"]], a["count"]) for a in _get(TRENDING, 20) if a["player_id"] in ids]
