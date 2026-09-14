"""Build the static projections file the website reads.

The whole site is static. This script runs once a week, writes one JSON file,
and the browser does the rest: week range, scoring format, league size and
value over replacement are all computed client side from the same numbers.
That keeps hosting free and makes every knob in the interface instant.

What gets written, per player per remaining week:
  components   predicted stat line, so any scoring format can be applied
  sd           calibrated spread for the simulation
  play_prob    probability he is available that week
  drivers      SHAP contributions grouped into readable factors
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import explain, features, simulate, train
from .config import (CURRENT_SEASON, FULL_FORM_FROM_WEEK, POSITIONS,
                     SCORING_FORMATS, STABILISATION_GAMES, STAT_COMPONENTS,
                     TRAIN_SEASONS)
from .scoring import score_components

OUT_PATH = Path("web/data/projections.json")


def early_season_form(rows: pd.DataFrame, rosters: pd.DataFrame, season: int,
                      from_week: int) -> pd.DataFrame:
    """Form for weeks 1 and 2, before any player has a game with a game
    behind it this season.

    Each player starts from his last game of an earlier season, the most
    recent full view of his role. A rookie starts from his first game this
    season, which carries only his role prior. Games already played this
    season are then blended in with the shrinkage the feature layer uses, so
    one game moves a quickly stabilising metric such as snap share further
    than a slow one such as yards after catch. Training is untouched: rolling
    windows still reset at the season boundary, and this only decides where an
    early projection starts.

    Only players on an active roster are kept, so retired, released and
    injured-reserve players do not reappear, and each takes his current team so
    an offseason move projects against the new schedule.
    """
    def latest(frame: pd.DataFrame) -> pd.DataFrame:
        return (frame.sort_values(["season", "week"])
                .groupby("player_id", observed=True).tail(1)
                .set_index("player_id"))

    current = rows[(rows["season"] == season) & (rows["week"] < from_week)]
    this_season = latest(current)
    form = latest(rows[(rows["season"] < season) & (rows["games_played"] >= 1)])
    form = pd.concat([form, this_season.drop(form.index, errors="ignore")])

    if "status" in rosters.columns:
        rosters = rosters[rosters["status"] == "ACT"]
    form = form[form.index.isin(rosters["player_id"])].copy()
    form["season"] = season
    form["team"] = (this_season["team"]
                    .combine_first(rosters.set_index("player_id")["team"])
                    .reindex(form.index).fillna(form["team"]))

    games = current.groupby("player_id", observed=True).size().reindex(form.index).fillna(0)
    means = (current.groupby("player_id", observed=True).mean(numeric_only=True)
             .reindex(form.index))
    for col in features.SHARE_COLS + features.EFF_COLS + ["td_oe"]:
        if f"{col}_adj" in form.columns and col in means.columns:
            form[f"{col}_adj"] = features.shrink(
                means[col], games, form[f"{col}_adj"], STABILISATION_GAMES.get(col, 6.0))

    # Display only, never a model input. The site reads these as this season's
    # touchdowns, so last season's totals must not carry over.
    totals = current.groupby("player_id", observed=True)[["actual_td", "expected_td"]].sum()
    form["actual_td_season"] = totals["actual_td"].reindex(form.index)
    form["expected_td_season"] = totals["expected_td"].reindex(form.index)
    return form


def upcoming_rows(rows: pd.DataFrame, schedule: pd.DataFrame, season: int,
                  from_week: int, through_week: int,
                  rosters: pd.DataFrame) -> pd.DataFrame:
    """Build one feature row per player per remaining game.

    Each future row carries the player's latest known form joined to that
    specific week's opponent and game script, which is what makes the
    projection matchup-aware rather than a flat season average.
    """
    if from_week < FULL_FORM_FROM_WEEK:
        latest = early_season_form(rows, rosters, season, from_week)
    else:
        # Form must come from games already played. Taking the last row of the
        # season would quietly pull in weeks the model is supposed to predict,
        # which inflates every accuracy number downstream.
        played = rows[(rows["season"] == season) & (rows["week"] < from_week)
                      & (rows["games_played"] >= 1)]
        latest = (played.sort_values(["player_id", "week"])
                  .groupby("player_id", observed=True).tail(1)
                  .set_index("player_id"))

    gs = features.game_script(schedule)
    gs = gs[(gs["season"] == season) & gs["week"].between(from_week, through_week)]

    rows = []
    for pid, form in latest.iterrows():
        team_games = gs[gs["team"] == form["team"]]
        for _, game in team_games.iterrows():
            row = form.copy()
            row["player_id"] = pid
            row["week"] = int(game["week"])
            row["opponent"] = game["opponent"]
            row["implied_total"] = game["implied_total"]
            row["spread"] = game["spread"]
            row["is_dome"] = game["is_dome"]
            row["is_favourite"] = game["is_favourite"]
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values(["player_id", "week"])
            .reset_index(drop=True))


def apply_future_defense(future: pd.DataFrame, defense: pd.DataFrame,
                         season: int) -> pd.DataFrame:
    """Attach each upcoming opponent's strength against the player's position.

    Season-to-date defensive factors are carried forward, shrunk toward
    neutral. This is where a player's remaining schedule stops being a talking
    point and becomes a number in the projection.
    """
    opp = features.opponent_strength(defense)
    recent = (opp[opp["season"] == season]
              .sort_values("week")
              .groupby(["defteam", "position"], observed=True)["def_factor"]
              .mean().reset_index())
    out = future.drop(columns=["def_factor"], errors="ignore").merge(
        recent.rename(columns={"defteam": "opponent"}),
        on=["opponent", "position"], how="left")
    out["def_factor"] = out["def_factor"].fillna(1.0)
    return out


def build(data: dict, from_week: int, season: int = CURRENT_SEASON,
          rounds: int = 300, verbose: bool = True,
          source: str = "nflverse") -> dict:
    rows = features.build_features(
        data["weekly"], data["team_week"], data["defense"], data["schedule"])
    hist = rows[rows["games_played"] >= 1]

    train_mask = ~((hist["season"] == season) & (hist["week"] >= from_week))
    train_df = hist[train_mask]
    if verbose:
        print(f"  training rows: {len(train_df):,}")

    comp_bundle = train.train_components(train_df, rounds=rounds)
    rank_bundle = train.train_ranker(train_df, rounds=rounds)
    variance = train.fit_variance(train_df, comp_bundle)

    sched = data["schedule"]
    through = int(sched.loc[sched["season"] == season, "week"].max())
    future = upcoming_rows(rows, sched, season, from_week, through, data["rosters"])
    if future.empty:
        raise RuntimeError(f"no player has form to project from week {from_week} of {season}")
    future = apply_future_defense(future, data["defense"], season)

    preds = train.predict_components(future, comp_bundle)
    for stat in STAT_COMPONENTS:
        if stat not in preds.columns:
            preds[stat] = 0.0
    # Matchup adjustment on top of the model's own view of the opponent.
    scale = future["def_factor"].to_numpy().clip(0.75, 1.3)
    for stat in ["rec_yd", "rush_yd", "pass_yd", "rec_td", "rush_td", "pass_td", "reception"]:
        preds[stat] = preds[stat].to_numpy() * scale

    ppr_pts = score_components(preds, "ppr", positions=future["position"])
    drivers = explain.shap_drivers(future, comp_bundle, variance,
                                   SCORING_FORMATS["ppr"])

    # Availability, once per player rather than once per week.
    roster = data["rosters"].set_index("player_id")
    missed = (data["weekly"][data["weekly"]["season"] == season]
              .groupby("player_id")["week"].count())
    # Before week one nobody has had a game to miss.
    weeks_so_far = from_week - 1

    # Early rows carry last season's sample size and role changes, which the
    # site would otherwise describe as this season's.
    early = from_week < FULL_FORM_FROM_WEEK
    weekly = data["weekly"]
    so_far = (weekly[(weekly["season"] == season) & (weekly["week"] < from_week)]
              .groupby("player_id")["week"].count())

    players: dict[str, dict] = {}
    for i, row in future.iterrows():
        pid = row["player_id"]
        if pid not in players:
            info = roster.loc[pid] if pid in roster.index else {}
            played = int(missed.get(pid, 0))
            play_prob = simulate.availability(
                position=row["position"],
                age=float(info.get("age", 26) if hasattr(info, "get") else 26),
                games_missed_last_2y=max(0, weeks_so_far - played),
                snap_load=float(row.get("snap_share_adj", 0.5) or 0.5),
                designation="healthy",
            )
            players[pid] = {
                "id": pid,
                "name": row["player_name"],
                "position": row["position"],
                "team": row["team"],
                "playProb": round(play_prob, 3),
                "weeks": [],
                "stats": {
                    "snapShare": _r(row.get("snap_share_adj")),
                    "snapTrend": _r(row.get("snap_share_trend")),
                    "targetShare": _r(row.get("target_share_adj")),
                    "targetTrend": _r(row.get("target_share_trend")),
                    "routeRate": _r(row.get("route_participation_adj")),
                    "wopr": _r(row.get("wopr_adj")),
                    "carryShare": _r(row.get("carry_share_adj")),
                    "glShare": _r(row.get("gl_touch_share_adj")),
                    "adot": _r(row.get("adot_adj")),
                    "yprr": _r(row.get("yards_per_route_adj")),
                    "actualTd": _r(row.get("actual_td_season")),
                    "expectedTd": _r(row.get("expected_td_season")),
                    "tdOverExpected": _r(row.get("td_oe_adj")),
                    "roleChange": 0 if early else int(row.get("role_change", 0) or 0),
                    "gamesPlayed": (int(so_far.get(pid, 0)) if early
                                    else int(row.get("games_played", 0) or 0)),
                },
            }
        mean = float(max(ppr_pts.iloc[i], 0.0))
        players[pid]["weeks"].append({
            "w": int(row["week"]),
            "opp": row["opponent"],
            "c": {s: round(float(preds[s].iloc[i]), 3) for s in STAT_COMPONENTS},
            "sd": round(simulate.sd_for(mean, row["position"], variance), 2),
            "drivers": {g: round(float(drivers[g].iloc[i]), 3)
                        for g in drivers.columns if abs(drivers[g].iloc[i]) > 0.01},
        })

    # Bye weeks, so a zero in the range is never hidden inside an average.
    for pid, p in players.items():
        weeks_present = {w["w"] for w in p["weeks"]}
        p["byeWeeks"] = [w for w in range(from_week, through + 1) if w not in weeks_present]

    return {
        "meta": {
            "season": season,
            "source": source,
            "fromWeek": from_week,
            "throughWeek": through,
            "earlySeason": early,
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scoringFormats": {k: {kk: vv for kk, vv in v.items() if kk != "label"}
                               | {"label": v["label"]}
                               for k, v in SCORING_FORMATS.items()},
            "replacementRankPerTeam": {p: __import__("pipeline.config", fromlist=["x"])
                                       .REPLACEMENT_RANK_PER_TEAM[p] for p in POSITIONS},
            "driverLabels": explain.DRIVER_LABELS,
        },
        "players": list(players.values()),
    }


def _r(v, nd: int = 4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build projections.json")
    ap.add_argument("--source", choices=["nflverse", "synthetic"], default="nflverse")
    ap.add_argument("--from-week", type=int, required=True,
                    help="first week to project, usually the upcoming one")
    ap.add_argument("--season", type=int, default=CURRENT_SEASON)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if args.source == "synthetic":
        from .synth import generate
        print("Generating synthetic data")
        data = generate(TRAIN_SEASONS + [args.season])
    else:
        from .ingest import load_all
        print("Loading nflverse data (first run downloads several seasons)")
        data = load_all(TRAIN_SEASONS + [args.season])

    print("Building features and training")
    payload = build(data, from_week=args.from_week, season=args.season,
                    rounds=args.rounds, source=args.source)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    size = args.out.stat().st_size / 1_000_000
    print(f"Wrote {args.out} — {len(payload['players'])} players, {size:.1f} MB")


if __name__ == "__main__":
    main()
