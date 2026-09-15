"""Grade the site's own projections against what happened, week by week.

Every build archives the projections the site showed, as
web/data/history/week-N.json, and the last one before week N's games is what
people saw. Once week N has been played, it is graded on that week's games:

  - the order the site gave each position, a missed game counting as the
    zero it is in a lineup, against FantasyPros' order and the last four
    games, by the same pairwise accuracy as the backtest;
  - that build's biggest disagreements with the experts: whether each
    player finished the week nearer the model's rank or the experts'.

Losing weeks stay on the record, which is the point of keeping one.

    python -m pipeline.scorecard     writes web/data/scorecard.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import disagree
from .backtest import pairwise_accuracy
from .config import CURRENT_SEASON, POSITIONS
from .current_week import current_week
from .scoring import actual_points

HISTORY = Path("web/data/history")
OUT = Path("web/data/scorecard.json")
# Disagreements graded each way per position: the first ones the page shows.
PER_SIDE = 3
METHODS = ("site", "consensus", "last4")


def _points(week: dict, weights: dict) -> float:
    return sum(v * weights.get(k, 0) for k, v in week["c"].items())


def grade(payload: dict, weekly: pd.DataFrame, teams_played: set[str]) -> dict:
    """Grade one archived build on the games of the week it was made for.

    weekly holds every game played, with PPR points in a pts column."""
    meta = payload["meta"]
    week, season = meta["fromWeek"], meta["season"]
    weights = meta["scoringFormats"]["ppr"]
    scored = (weekly[(weekly["season"] == season) & (weekly["week"] == week)]
              .groupby("player_id")["pts"].sum().to_dict())
    before = weekly[(weekly["season"] < season) | ((weekly["season"] == season) & (weekly["week"] < week))]
    last4 = (before.sort_values(["season", "week"]).groupby("player_id").tail(4)
             .groupby("player_id")["pts"].mean())

    rows = []
    for p in payload["players"]:
        entry = next((w for w in p["weeks"] if w["w"] == week), None)
        # Players whose team had no game that week are left out; a player
        # whose team played and who did not is in, with the zero he scored.
        if entry is None or p["team"] not in teams_played or p["id"] not in last4.index:
            continue
        rows.append({"id": p["id"], "position": p["position"],
                     "site": _points(entry, weights) * entry.get("p", p["playProb"]),
                     "rank": p.get("consensusRank"), "last4": float(last4[p["id"]]),
                     "actual": float(scored.get(p["id"], 0.0))})
    f = pd.DataFrame(rows, columns=["id", "position", "site", "rank", "last4", "actual"])

    positions = {}
    for pos, g in f.groupby("position"):
        if pos not in POSITIONS or len(g) < 12 or g["rank"].isna().all():
            continue
        # Anyone the experts leave unranked sits below everyone they rank.
        consensus = -g["rank"].fillna(g["rank"].max() + 1)
        actual = g["actual"].to_numpy()
        acc = {k: pairwise_accuracy(v.to_numpy(), actual)
               for k, v in {"site": g["site"], "consensus": consensus, "last4": g["last4"]}.items()}
        positions[pos] = {k: round(a, 4) for k, (a, _) in acc.items()} | {"pairs": acc["site"][1]}
    overall = ({k: round(float(np.mean([v[k] for v in positions.values()])), 4) for k in METHODS}
               if positions else {})

    # Each disagreement is judged by where the player finished that week among
    # everyone at his position whose team played, a missed game included.
    finish = {}
    for _, g in f.groupby("position"):
        finish.update({pid: i for i, pid in
                       enumerate(g.sort_values("actual", ascending=False)["id"], start=1)})
    calls = []
    for pos, sides in disagree.biggest(disagree.disagreements(payload), PER_SIDE).items():
        for side, picks in sides.items():
            for r in picks:
                pid = r["player"]["id"]
                if pid not in finish:
                    continue
                to_model = abs(finish[pid] - r["modelRank"])
                to_experts = abs(finish[pid] - r["consensusRank"])
                calls.append({
                    "id": pid, "name": r["player"]["name"], "position": pos, "side": side,
                    "modelRank": r["modelRank"], "consensusRank": r["consensusRank"],
                    "finish": finish[pid], "points": round(scored.get(pid, 0.0), 1),
                    "closer": ("model" if to_model < to_experts
                               else "experts" if to_experts < to_model else "even"),
                })
    return {"week": week, "overall": overall, "positions": positions, "calls": calls}


def _week_of(path: Path) -> int:
    return int(path.stem.split("-")[1])


def main() -> None:
    season = CURRENT_SEASON
    # The first week with a game still to play; everything before it is done.
    played_through = current_week(season=season) - 1
    archives = sorted(HISTORY.glob("week-*.json"), key=_week_of)
    due = [a for a in archives if _week_of(a) <= played_through]

    weeks = []
    if due:
        from .ingest import load_all
        data = load_all([season - 1, season])
        weekly = data["weekly"].assign(pts=lambda d: actual_points(d, "ppr"))
        team_week = data["team_week"]
        for path in due:
            payload = json.loads(path.read_text())
            if payload["meta"].get("season") != season:
                continue
            week = payload["meta"]["fromWeek"]
            teams = set(team_week.loc[(team_week["season"] == season) & (team_week["week"] == week), "team"])
            if not teams:
                print(f"  week {week} is not in the data yet; it will be graded next run")
                continue
            weeks.append(grade(payload, weekly, teams))

    calls = [c for w in weeks for c in w["calls"]]
    card = {
        "season": season,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weeks": weeks,
        "calls": {k: sum(c["closer"] == k for c in calls) for k in ("model", "experts", "even")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(card, indent=1) + "\n")
    print(f"Wrote {OUT}: {len(weeks)} graded weeks, {len(calls)} graded calls")


if __name__ == "__main__":
    main()
