"""Walk-forward validation.

A fantasy model that reports in-sample accuracy is worthless, and a model
scored only on squared error is measuring the wrong thing, because the product
outputs an order rather than a number. So this measures what the site actually
does: given a handful of players, how often does it put them in the right
order.

Every week of the test season the model is retrained on what had been played
by then and asked for each player's next `horizon` games through the same
forecast the site uses, matchups and all. A change that helps here therefore
helps what people see.

The model has to beat these, all built from the same pre-week information:
  last4      the player's last four games, reaching back into last season
             early in the year. What most people do in their head.
  season     season-to-date points per game, or last season's before any games
  volume     targets plus carries over the last four games, no model
  market     FantasyPros' rest-of-season consensus, from the last scrape before
             the week began

Training never sees a week at or after the week being predicted.
"""

from __future__ import annotations

import argparse
import itertools
from datetime import timedelta

import numpy as np
import pandas as pd

from . import market, project
from .config import CURRENT_SEASON, POSITIONS, TRAIN_SEASONS
from .current_week import SEASON_OPENER
from .scoring import actual_points, score_components

METHODS = ["model", "blend", "blend50", "market", "last4", "season_avg", "volume"]


def pairwise_accuracy(pred: np.ndarray, actual: np.ndarray,
                      min_gap: float = 2.0) -> tuple[float, int]:
    """Share of player pairs placed in the correct order.

    Pairs whose real outcomes finished within min_gap points are excluded:
    getting two effectively tied players in a particular order is luck, not
    skill, and including them drags every method toward 50%.
    """
    correct = total = 0
    for i, j in itertools.combinations(range(len(pred)), 2):
        gap = actual[i] - actual[j]
        if abs(gap) < min_gap:
            continue
        total += 1
        if np.sign(pred[i] - pred[j]) == np.sign(gap):
            correct += 1
    return (correct / total if total else float("nan")), total


def spearman(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) < 3:
        return float("nan")
    a = pd.Series(pred).rank()
    b = pd.Series(actual).rank()
    return float(a.corr(b))


def baselines(weekly: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    w = weekly.assign(pts=actual_points(weekly, "ppr"))
    past = w[(w["season"] < season) | ((w["season"] == season) & (w["week"] < week))]
    past = past.sort_values(["season", "week"])
    recent = past.groupby("player_id").tail(4).groupby("player_id")
    this_season = past[past["season"] == season].groupby("player_id")["pts"].mean()
    last_season = past[past["season"] == season - 1].groupby("player_id")["pts"].mean()
    out = pd.DataFrame({"last4": recent["pts"].mean(),
                        "volume": recent[["targets", "carries"]].sum().sum(axis=1)})
    out["season_avg"] = this_season.reindex(out.index).fillna(last_season)
    return out


def run(data: dict, season: int, start_week: int = 1, end_week: int = 17,
        horizon: int = 4, rounds: int = 250, market_weight: float = 0.25,
        use_market: bool = True, verbose: bool = True,
        frames: list | None = None) -> pd.DataFrame:
    weekly = data["weekly"]
    opener = SEASON_OPENER.get(season)
    results = []

    for week in range(start_week, end_week - horizon + 2):
        through = week + horizon - 1
        fc = project.forecast(data, season, week, through, rounds=rounds)
        future = fc["future"]
        pts = score_components(fc["model_preds"], "ppr", positions=future["position"]).to_numpy()
        by = future.assign(pts=pts).groupby("player_id")
        frame = pd.DataFrame({"position": by["position"].first(), "model": by["pts"].mean()})

        rank = pd.Series(dtype=float)
        if use_market and opener is not None:
            rank = market.ranks_before(pd.Timestamp(opener + timedelta(days=7 * (week - 1))))
        frame["market_rank"] = rank.reindex(frame.index)
        frame["blend"] = market.blend(frame["model"], frame["position"],
                                      frame["market_rank"], market_weight)
        frame["blend50"] = market.blend(frame["model"], frame["position"],
                                        frame["market_rank"], 0.5)
        # Anyone the market leaves unranked sits below everyone it does rank.
        floor = frame.groupby("position")["market_rank"].transform("max").fillna(0) + 1
        frame["market"] = -frame["market_rank"].fillna(floor)
        if frame["market_rank"].isna().all():
            frame["market"] = np.nan

        frame = frame.join(baselines(weekly, season, week))
        truth = weekly[(weekly["season"] == season) & weekly["week"].between(week, through)]
        played = truth.assign(pts=actual_points(truth, "ppr")).groupby("player_id")["pts"]
        frame["actual"] = played.mean()
        # What the site ranks on is points per scheduled game, with a missed
        # game as zero, so the saved frames keep players who never took the
        # field, with their scheduled games and total points.
        frame["games"] = future.groupby("player_id").size()
        frame["total"] = played.sum().reindex(frame.index).fillna(0.0)
        if frames is not None:
            frames.append(frame.assign(week=week))
        frame = frame.dropna(subset=["actual", "model", "last4", "season_avg"])

        for pos in POSITIONS:
            grp = frame[frame["position"] == pos]
            if len(grp) < 12:
                continue
            actual = grp["actual"].to_numpy()
            row = {"week": week, "position": pos, "n": len(grp)}
            for method in METHODS:
                if grp[method].isna().all():
                    row[f"{method}_pair"] = row[f"{method}_rho"] = np.nan
                    continue
                acc, row["pairs"] = pairwise_accuracy(grp[method].to_numpy(), actual)
                row[f"{method}_pair"] = acc
                row[f"{method}_rho"] = spearman(grp[method].to_numpy(), actual)
            for method in ["model", "blend", "last4", "season_avg"]:
                row[f"{method}_mae"] = float(np.mean(np.abs(grp[method] - actual)))
            results.append(row)
        if verbose:
            print(f"  week {week}: scored {len(frame)} players", flush=True)

    return pd.DataFrame(results)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        rows.append({
            "method": method,
            "pairwise_accuracy": res[f"{method}_pair"].mean(),
            "spearman": res[f"{method}_rho"].mean(),
            "mae": res[f"{method}_mae"].mean() if f"{method}_mae" in res else np.nan,
        })
    return pd.DataFrame(rows).set_index("method").round(4)


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward backtest")
    ap.add_argument("--source", choices=["nflverse", "synthetic"], default="nflverse")
    ap.add_argument("--season", type=int, default=CURRENT_SEASON - 1)
    ap.add_argument("--start-week", type=int, default=1)
    ap.add_argument("--end-week", type=int, default=17)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--market-weight", type=float, default=0.25)
    ap.add_argument("--out", help="write per-week, per-position results to this CSV")
    ap.add_argument("--frames", help="write every scored player-week to this parquet file, "
                                     "so blends can be compared without retraining")
    args = ap.parse_args()

    # Testing an earlier season must not train on the seasons after it.
    seasons = [s for s in TRAIN_SEASONS if s <= args.season]
    if args.source == "synthetic":
        from .synth import generate
        data = generate(seasons)
    else:
        from .ingest import load_all
        data = load_all(seasons)

    frames: list | None = [] if args.frames else None
    res = run(data, season=args.season, start_week=args.start_week, end_week=args.end_week,
              horizon=args.horizon, rounds=args.rounds, market_weight=args.market_weight,
              use_market=args.source != "synthetic", frames=frames)
    if res.empty:
        print("No comparable weeks produced results.")
        return
    if args.out:
        res.to_csv(args.out, index=False)
    if frames:
        pd.concat(frames).rename_axis("player_id").reset_index().to_parquet(args.frames, index=False)

    print("\nAccuracy by method, averaged over walk-forward weeks")
    print(summarise(res).to_string())
    for label, part in (("weeks 1-2", res[res["week"] <= 2]), ("week 3 on", res[res["week"] >= 3])):
        if not part.empty:
            print(f"\n{label}")
            print(summarise(part)[["pairwise_accuracy", "spearman"]].to_string())
    print("\nPairwise accuracy by position")
    print(res.groupby("position")[[f"{m}_pair" for m in METHODS]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
