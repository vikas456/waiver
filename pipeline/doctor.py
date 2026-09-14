"""Check each nflverse loader against one season and report what breaks.

The ingestion path was first written without network access to nflverse, so
real runs are where renamed columns and mismatched join keys surface. Loaders
run in dependency order, each on its own, so one failure does not hide the
next, and every failure prints its traceback.

Raising is not the only failure worth catching. A merge that turns targets
into targets_x, or a join that multiplies rows, runs cleanly and quietly
corrupts training. So every table is also checked for the columns downstream
code reads, for how many of them are empty, and for duplicate keys.

    python -m pipeline.doctor                  # the last completed season
    python -m pipeline.doctor --season 2026    # the season in progress

The season in progress fails some checks for its first two or three weeks,
and that is expected: with one game played every rolling feature is still at
its prior, and snap counts trail the stat files by a day or two.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd

from . import features, ingest
from .config import CURRENT_SEASON, STAT_COMPONENTS
from .scoring import actual_points

# Per table: the key that must identify a row, the largest plausible share of
# nulls in each column downstream code reads, and columns that must vary.
# A column that is present but constant usually means its source was missing
# and a prior filled the gap.
CHECKS = {
    "pbp": {"key": ["game_id", "play_id"],
            "nulls": {"game_id": 0, "season": 0, "week": 0, "posteam": 0.1}},
    # Cut and practice-squad players often lack a birth date. None of them
    # play, so their missing age never reaches a projection.
    "rosters": {"key": ["player_id"],
                "nulls": {"player_id": 0, "position": 0, "age": 0.12}},
    "schedule": {"key": ["season", "week", "home_team"],
                 "nulls": {"home_team": 0, "away_team": 0, "roof": 0.05}},
    "weekly": {
        "key": ["player_id", "season", "week"],
        "nulls": {
            **{c: 0 for c in STAT_COMPONENTS},
            "player_id": 0, "player_name": 0, "position": 0, "team": 0,
            "opponent_team": 0, "targets": 0, "carries": 0,
            "team_pass_att": 0.15, "target_share": 0.15, "carry_share": 0.15,
            "air_yards": 0.15, "pfr_id": 0.05, "years_exp": 0.05,
            "draft_number": 0.4, "snap_share": 0.1, "routes": 0.2,
        },
        "varies": ["snap_share", "target_share", "carry_share", "routes"],
    },
    "team_week": {"key": ["game_id", "team"],
                  "nulls": {"plays": 0, "pass_rate": 0, "off_epa": 0.01,
                            "proe": 0.05, "neutral_plays": 0.05, "sack_rate": 0}},
    "defense": {"key": ["season", "week", "defteam", "position"],
                "nulls": {"defteam": 0, "pts_allowed": 0, "targets_allowed": 0}},
    "features": {
        "key": ["player_id", "season", "week"],
        "nulls": {"opponent": 0.02, "implied_total": 0.02, "team_proe": 0.1},
        "varies": ["target_share_adj", "carry_share_adj", "snap_share_adj",
                   "route_participation_adj", "wopr_adj", "yards_per_route_adj",
                   "def_factor", "implied_total", "team_proe"],
    },
}


def _table_problems(frame: pd.DataFrame, spec: dict) -> list[str]:
    problems = []
    nulls = spec.get("nulls", {})
    missing = [c for c in nulls if c not in frame.columns]
    if missing:
        problems.append(f"missing columns: {', '.join(missing)}")
    for col, limit in nulls.items():
        if col in frame.columns:
            share = frame[col].isna().mean()
            if share > limit:
                problems.append(f"{col}: {share:.1%} null, expected at most {limit:.0%}")
    key = spec.get("key", [])
    if key and all(c in frame.columns for c in key):
        dups = int(frame.duplicated(key).sum())
        if dups:
            problems.append(f"{dups:,} duplicate rows on ({', '.join(key)})")
    suffixed = [c for c in frame.columns if c.endswith(("_x", "_y"))]
    if suffixed:
        problems.append(f"merge collision columns: {', '.join(suffixed)}")
    for col in spec.get("varies", []):
        if col in frame.columns and frame[col].nunique(dropna=True) <= 1:
            problems.append(f"{col} is constant, so its source is probably missing")
    return problems


def _weekly_problems(weekly: pd.DataFrame) -> list[str]:
    problems = []
    if weekly["week"].max() > 18:
        problems.append("postseason weeks present")
    # nflverse publishes its own PPR total. If scoring.py disagrees with it,
    # a stat column has been mapped onto the wrong component.
    gap = (actual_points(weekly, "ppr") - weekly["nflverse_ppr"]).abs()
    agree = (gap < 0.05).mean()
    if agree < 0.99:
        worst = weekly.assign(gap=gap).nlargest(3, "gap")
        sample = "; ".join(f"{r.player_name} wk{r.week} off by {r.gap:.2f}"
                           for r in worst.itertuples())
        problems.append(f"PPR matches nflverse on only {agree:.1%} of rows ({sample})")
    return problems


def _teams_problems(frame: pd.DataFrame, col: str) -> list[str]:
    n = frame[col].nunique()
    return [] if n == 32 else [f"{n} distinct {col} values, expected 32"]


def _stages(season: int) -> list[tuple]:
    s = [season]
    return [
        ("pbp", [], lambda t: ingest.load_pbp(s), None),
        ("rosters", [], lambda t: ingest.load_rosters(s), None),
        ("schedule", [], lambda t: ingest.load_schedule(s), None),
        ("weekly", ["pbp", "rosters"], lambda t: ingest.load_weekly(s), _weekly_problems),
        ("team_week", ["pbp"], lambda t: ingest.load_team_week(s),
         lambda f: _teams_problems(f, "team")),
        ("defense", ["weekly"], lambda t: ingest.load_defense(t["weekly"]),
         lambda f: _teams_problems(f, "defteam")),
        ("features", ["weekly", "team_week", "defense", "schedule"],
         lambda t: features.build_features(
             t["weekly"], t["team_week"], t["defense"], t["schedule"]), None),
    ]


def run(season: int) -> bool:
    tables: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for name, needs, load, extra in _stages(season):
        blocked = [n for n in needs if n not in tables]
        if blocked:
            print(f"  {name:<10} skipped, needs {', '.join(blocked)}")
            failed.append(name)
            continue
        try:
            frame = load(tables)
        except Exception:
            print(f"  {name:<10} FAIL   raised")
            print("    " + traceback.format_exc().replace("\n", "\n    ").rstrip())
            failed.append(name)
            continue
        problems = _table_problems(frame, CHECKS[name])
        if not frame.empty and extra:
            problems += extra(frame)
        if frame.empty:
            problems.append("no rows")
        status = "FAIL" if problems else "ok"
        print(f"  {name:<10} {status:<6} {len(frame):>8,} rows")
        for p in problems:
            print(f"               {p}")
        if problems:
            failed.append(name)
        tables[name] = frame
    print()
    if failed:
        print(f"{len(failed)} of {len(_stages(season))} stages failed: {', '.join(failed)}")
    else:
        print("All loaders passed.")
    return not failed


def main() -> None:
    ap = argparse.ArgumentParser(description="Check the nflverse loaders")
    ap.add_argument("--season", type=int, default=CURRENT_SEASON - 1)
    ap.add_argument("--use-cache", action="store_true",
                    help=f"check what is in {ingest.CACHE_DIR} instead of downloading fresh")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if not args.use_cache:
            ingest.CACHE_DIR = Path(tmp)
        print(f"Checking nflverse loaders against the {args.season} season")
        print(f"cache: {ingest.CACHE_DIR}\n")
        ok = run(args.season)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
