"""Where the model and the experts rank a player furthest apart.

The site blends the model with FantasyPros' consensus, so to see where the
model itself disagrees, its own projection is recovered by taking the
consensus pull back out. Both sides count games a player is expected to miss:
the model through his chance of playing each week, the experts in their rank.

The disagreement page shows these every week, and the scorecard grades them
once the games are played, so the model's boldest calls are on the record
whichever way they go.
"""

from __future__ import annotations

from .config import POSITIONS

# How deep each position's list reaches: roughly the players a 12-team league
# rosters. A disagreement about the 90th receiver is not a call anyone makes.
RELEVANT = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}


def _points(week: dict, weights: dict) -> float:
    return sum(v * weights.get(k, 0) for k, v in week["c"].items())


def model_ppg(player: dict, weights: dict) -> float:
    """Expected points per scheduled game from the model alone, before the
    pull toward consensus, counting the chance he misses each game."""
    weeks = player["weeks"]
    if not weeks:
        return 0.0
    total = sum((_points(w, weights) - w.get("drivers", {}).get("market", 0.0))
                * w.get("p", player["playProb"]) for w in weeks)
    return total / len(weeks)


def reasons(player: dict, weights: dict, higher: bool) -> list[tuple[str, float]]:
    """The model's strongest factors in the direction of the disagreement, in
    points per game against a typical player at his position, with games he
    is expected to miss counted as their own factor."""
    weeks = player["weeks"]
    if not weeks:
        return []
    totals: dict[str, float] = {}
    for w in weeks:
        for group, value in w.get("drivers", {}).items():
            if group != "market":
                totals[group] = totals.get(group, 0.0) + value / len(weeks)
    if_plays = sum(_points(w, weights) - w.get("drivers", {}).get("market", 0.0) for w in weeks) / len(weeks)
    totals["missed_games"] = model_ppg(player, weights) - if_plays
    pointing = [(g, v) for g, v in totals.items() if (v > 0.1 if higher else v < -0.1)]
    return sorted(pointing, key=lambda t: abs(t[1]), reverse=True)[:2]


def disagreements(payload: dict) -> dict[str, list[dict]]:
    """Per position, every relevant player the consensus ranks, with his rank
    by the model alone and by the experts, sorted from where the model is
    most above the experts to where it is most below."""
    weights = payload["meta"]["scoringFormats"]["ppr"]
    out = {}
    for pos in POSITIONS:
        pool = [p for p in payload["players"] if p["position"] == pos]
        ppg = {p["id"]: model_ppg(p, weights) for p in pool}
        order = sorted(pool, key=lambda p: ppg[p["id"]], reverse=True)
        model_rank = {p["id"]: i for i, p in enumerate(order, start=1)}
        rows = []
        for p in pool:
            experts = p.get("consensusRank")
            mine = model_rank[p["id"]]
            if experts is None or min(experts, mine) > RELEVANT[pos]:
                continue
            rows.append({"player": p, "modelRank": mine, "consensusRank": experts,
                         "gap": experts - mine, "modelPpg": ppg[p["id"]]})
        out[pos] = sorted(rows, key=lambda r: r["gap"], reverse=True)
    return out


def biggest(table: dict[str, list[dict]], per_side: int) -> dict[str, dict[str, list[dict]]]:
    """The largest gaps each way, per position.

    A player carrying an injury or absence designation is kept off the side
    where the model ranks him higher: there the experts may simply know more
    about when he will be back than the model can read."""
    return {pos: {"higher": [r for r in rows if r["gap"] > 0 and not r["player"].get("injury")][:per_side],
                  "lower": [r for r in reversed(rows) if r["gap"] < 0][:per_side]}
            for pos, rows in table.items()}
