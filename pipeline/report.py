"""The weekly report: how the site did, against the experts and against
another site's projections.

The scorecard already grades an archived build against what happened, for the
site itself, for expert consensus and for a recent-points baseline. This adds
the one comparison it cannot make on its own — Sleeper's published projections
for the same players — and writes the result as a report the site publishes.

Only players both sides have an opinion about are scored, which is the only
fair way to do it, and the report says how many that was.

    python -m pipeline.report --week 2
    python -m pipeline.report --week 10 --season 2025 --archive /tmp/w10.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import scorecard
from .backtest import pairwise_accuracy
from .config import POSITIONS
from .scoring import actual_points

REPORTS_DIR = Path("web/data/reports")
SNAPSHOTS = Path("web/data/sleeper")
HISTORY = Path("web/data/history")

# A pair of players counts when their scores finished clearly apart, the same
# bar the scorecard uses, so the two numbers can sit in one table.
MIN_GAP = 2.0


def sleeper_numbers(season: int, week: int, snapshot_dir: Path = SNAPSHOTS,
                    payload: Path | None = None) -> tuple[dict, str]:
    """Sleeper's projections for a week, and where they came from.

    A snapshot taken before the games is the honest source. Falling back to
    their API means grading against numbers that may have been revised since,
    so the report says which one it used.
    """
    path = snapshot_dir / f"week-{week}.json"
    saved = None
    if path.exists():
        try:
            saved = json.loads(path.read_text())
        except ValueError:
            saved = None
    if saved and int(saved.get("season", season)) == season and int(saved.get("week", week)) == week:
        return saved.get("projected", {}), "captured before the games"

    from tools.sleeper_snapshot import snapshot

    try:
        # Mapping their ids to ours needs the build being graded, not whatever
        # the site happens to be showing now: a past week is a past roster.
        fetched = (snapshot(season, week, payload) if payload else snapshot(season, week))
    except Exception as err:
        print(f"  Sleeper projections unavailable ({type(err).__name__})")
        return {}, "unavailable"
    return fetched.get("projected", {}), "read back from Sleeper afterwards"


def grade_week(payload: dict, weekly: pd.DataFrame, teams_played: set[str],
               theirs: dict) -> dict:
    """The scorecard's grading, plus Sleeper over the same players."""
    card = scorecard.grade(payload, weekly, teams_played)

    meta = payload["meta"]
    week, season = meta["fromWeek"], meta["season"]
    weights = meta["scoringFormats"]["ppr"]
    scored = (weekly[(weekly["season"] == season) & (weekly["week"] == week)]
              .groupby("player_id")["pts"].sum().to_dict())

    rows = []
    for p in payload["players"]:
        entry = next((w for w in p["weeks"] if w["w"] == week), None)
        if entry is None or p["team"] not in teams_played or p["id"] not in theirs:
            continue
        rows.append({"position": p["position"],
                     "site": scorecard._points(entry, weights) * entry.get("p", p["playProb"]),
                     "sleeper": float(theirs[p["id"]]),
                     "actual": float(scored.get(p["id"], 0.0))})

    head = pd.DataFrame(rows, columns=["position", "site", "sleeper", "actual"])
    per_position, pairs_total = {}, 0
    for pos, group in head.groupby("position"):
        if pos not in POSITIONS or len(group) < 12:
            continue
        actual = group["actual"].to_numpy()
        site, pairs = pairwise_accuracy(group["site"].to_numpy(), actual, min_gap=MIN_GAP)
        other, _ = pairwise_accuracy(group["sleeper"].to_numpy(), actual, min_gap=MIN_GAP)
        per_position[pos] = {"site": round(site, 4), "sleeper": round(other, 4), "pairs": pairs}
        pairs_total += pairs
    head_to_head = {}
    if per_position:
        head_to_head = {
            "site": round(float(np.mean([v["site"] for v in per_position.values()])), 4),
            "sleeper": round(float(np.mean([v["sleeper"] for v in per_position.values()])), 4),
            "players": len(head), "pairs": pairs_total, "byPosition": per_position,
        }
    return {"card": card, "sleeper": head_to_head}


def results(season: int, week: int) -> tuple[pd.DataFrame, set[str] | None]:
    """Every game played, with PPR points, and who played in the week asked
    about. Gathered exactly as pipeline/scorecard.py does, so the two grade the
    same games the same way."""
    from .ingest import load_all

    data = load_all([season - 1, season])
    weekly = data["weekly"].assign(pts=lambda d: actual_points(d, "ppr"))
    team_week = data["team_week"]
    teams = set(team_week.loc[(team_week["season"] == season)
                              & (team_week["week"] == week), "team"])
    return weekly, (teams or None)


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def write_report(graded: dict, season: int, week: int, source: str,
                 out_dir: Path = REPORTS_DIR) -> Path:
    """Turn a graded week into the report the site publishes."""
    card, head = graded["card"], graded["sleeper"]
    overall = card.get("overall") or {}
    sections = []

    if overall:
        sections.append({
            "heading": "Against the experts",
            "paragraphs": [
                "Every pair of players at the same position whose scores finished at least two "
                "points apart, and how often each source put them in the right order. A player "
                "whose team played and who did not counts as the zero he scored."],
            "table": {"columns": ["Source", "Right order"],
                      "rows": [["Waiver", _pct(overall.get("site"))],
                               ["Expert consensus", _pct(overall.get("consensus"))],
                               ["Going by recent points", _pct(overall.get("last4"))]]},
        })

    if head:
        sections.append({
            "heading": "Against Sleeper's projections",
            "paragraphs": [
                f"Over the {head['players']} players both Waiver and Sleeper projected, scored the "
                f"same way on the same games. Sleeper's numbers were {source}."],
            "table": {"columns": ["Source", "Right order"],
                      "rows": [["Waiver", _pct(head.get("site"))],
                               ["Sleeper", _pct(head.get("sleeper"))]]},
            "notes": [f"{head['pairs']} pairs counted."],
        })

    calls = card.get("calls") or []
    if calls:
        rows = [[c["name"], f"{c['position']}{c['modelRank']}", f"{c['position']}{c['consensusRank']}",
                 f"{c['position']}{c['finish']} · {c['points']:.1f}",
                 {"model": "Waiver", "experts": "Experts", "even": "Even"}[c["closer"]]]
                for c in calls]
        won = sum(1 for c in calls if c["closer"] == "model")
        sections.append({
            "heading": "The calls against consensus",
            "paragraphs": [
                f"The players Waiver ranked furthest from the experts before the week. It was "
                f"nearer the finish on {won} of {len(calls)}."],
            "table": {"columns": ["Player", "Waiver", "Experts", "Finished", "Nearer"], "rows": rows},
        })

    site, experts = overall.get("site"), overall.get("consensus")
    if site is not None and experts is not None:
        verdict = ("ahead of expert consensus" if site > experts
                   else "behind expert consensus" if site < experts else "level with expert consensus")
        summary = (f"In week {week}, Waiver ordered players correctly {_pct(site)} of the time, "
                   f"{verdict} on {_pct(experts)}.")
    else:
        summary = f"Week {week} has been graded; not every comparison had enough players to score."

    report = {
        "week": week, "season": season,
        "title": f"Week {week}: how the picks actually did",
        "date": date.today().strftime("%d %B %Y"),
        "summary": summary,
        "sections": sections,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"week-{week}.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the weekly report")
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season", type=int)
    ap.add_argument("--archive", type=Path, help="the build to grade; defaults to the week's archive")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR)
    args = ap.parse_args()

    archive = args.archive or HISTORY / f"week-{args.week}.json"
    if not archive.exists():
        raise SystemExit(f"no archived build at {archive}; nothing to grade")
    payload = json.loads(archive.read_text())
    season = args.season or int(payload["meta"]["season"])

    weekly, teams = results(season, args.week)
    if teams is None:
        raise SystemExit(f"week {args.week} of {season} has no results yet")

    theirs, source = sleeper_numbers(season, args.week, payload=archive)
    graded = grade_week(payload, weekly, teams, theirs)
    path = write_report(graded, season, args.week, source, args.out)
    overall = graded["card"].get("overall") or {}
    print(f"Wrote {path} — site {_pct(overall.get('site'))}, "
          f"consensus {_pct(overall.get('consensus'))}, "
          f"Sleeper {_pct((graded['sleeper'] or {}).get('sleeper'))}")


if __name__ == "__main__":
    main()
