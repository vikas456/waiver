"""Walk-forward validation.

A fantasy model that reports in-sample accuracy is worthless, and a model
scored only on squared error is measuring the wrong thing, because the product
outputs an order rather than a number. So this measures what the site actually
does: given a handful of players, how often does it put them in the right
order, and are the floor and ceiling bands honest.

Three baselines have to be beaten before any claim of being better holds:
  last4      the player's trailing four-game average, which is what most
             people do in their head and is a genuinely hard baseline
  season     season-to-date average points per game
  volume     trailing target and carry volume alone, no model

Training never sees a week at or after the week being predicted.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from . import features, train
from .config import CURRENT_SEASON, POSITIONS, TRAIN_SEASONS
from .scoring import score_components


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


def run(data: dict, season: int = CURRENT_SEASON, start_week: int = 6,
        end_week: int = 17, horizon: int = 4, rounds: int = 250,
        verbose: bool = True) -> pd.DataFrame:
    """Retrain each week and score the next `horizon` weeks of real outcomes."""
    hist = features.build_features(
        data["weekly"], data["team_week"], data["defense"], data["schedule"])
    hist = hist[hist["games_played"] >= 1]
    results = []

    for week in range(start_week, end_week - horizon + 2):
        train_df = hist[~((hist["season"] == season) & (hist["week"] >= week))]
        test_df = hist[(hist["season"] == season)
                       & hist["week"].between(week, week + horizon - 1)]
        if len(train_df) < 500 or test_df.empty:
            continue

        bundle = train.train_components(train_df, rounds=rounds)

        # Model projection for the test window, using only pre-week form.
        form = (train_df[train_df["season"] == season]
                .sort_values("week").groupby("player_id", observed=True).tail(1))
        if form.empty:
            continue
        preds = train.predict_components(form, bundle)
        form = form.assign(
            model=score_components(preds, "ppr", positions=form["position"]).to_numpy())

        # Baselines, all computed from the same pre-week information.
        form["last4"] = form.get("fantasy_points_ppr_r5", form["fantasy_points_ppr"])
        prior = train_df[train_df["season"] == season]
        season_avg = prior.groupby("player_id")["fantasy_points_ppr"].mean()
        last4 = (prior.sort_values("week").groupby("player_id")["fantasy_points_ppr"]
                 .apply(lambda s: s.tail(4).mean()))
        volume = (prior.sort_values("week").groupby("player_id")
                  .apply(lambda g: g.tail(4)[["targets", "carries"]].sum().sum(),
                         include_groups=False))
        form["season_avg"] = form["player_id"].map(season_avg)
        form["last4"] = form["player_id"].map(last4)
        form["volume"] = form["player_id"].map(volume)

        truth = (test_df.groupby("player_id")["fantasy_points_ppr"].mean()
                 .rename("actual").reset_index())
        joined = form.merge(truth, on="player_id", how="inner").dropna(
            subset=["actual", "model", "last4", "season_avg"])
        if len(joined) < 30:
            continue

        for pos in POSITIONS:
            grp = joined[joined["position"] == pos]
            if len(grp) < 12:
                continue
            actual = grp["actual"].to_numpy()
            row = {"week": week, "position": pos, "n": len(grp)}
            for method in ["model", "last4", "season_avg", "volume"]:
                acc, pairs = pairwise_accuracy(grp[method].to_numpy(), actual)
                row[f"{method}_pair"] = acc
                row[f"{method}_rho"] = spearman(grp[method].to_numpy(), actual)
                row["pairs"] = pairs
            row["model_mae"] = float(np.mean(np.abs(grp["model"] - actual)))
            row["last4_mae"] = float(np.mean(np.abs(grp["last4"] - actual)))
            results.append(row)
        if verbose:
            print(f"  week {week}: scored {len(joined)} players")

    return pd.DataFrame(results)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    methods = ["model", "last4", "season_avg", "volume"]
    rows = []
    for method in methods:
        rows.append({
            "method": method,
            "pairwise_accuracy": res[f"{method}_pair"].mean(),
            "spearman": res[f"{method}_rho"].mean(),
        })
    out = pd.DataFrame(rows).set_index("method")
    out["mae"] = [res["model_mae"].mean(), res["last4_mae"].mean(), np.nan, np.nan]
    return out.round(4)


def calibration(res_intervals: pd.DataFrame) -> float:
    """Share of outcomes falling inside the stated 80% band.

    If this is not close to 0.80 the floor and ceiling numbers on the site are
    decoration rather than information, so it is checked explicitly.
    """
    inside = ((res_intervals["actual"] >= res_intervals["p10"])
              & (res_intervals["actual"] <= res_intervals["p90"]))
    return float(inside.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward backtest")
    ap.add_argument("--source", choices=["nflverse", "synthetic"], default="nflverse")
    ap.add_argument("--season", type=int, default=CURRENT_SEASON - 1)
    ap.add_argument("--start-week", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=250)
    args = ap.parse_args()

    if args.source == "synthetic":
        from .synth import generate
        data = generate(TRAIN_SEASONS)
    else:
        from .ingest import load_all
        data = load_all(TRAIN_SEASONS)

    res = run(data, season=args.season, start_week=args.start_week,
              horizon=args.horizon, rounds=args.rounds)
    if res.empty:
        print("No comparable weeks produced results.")
        return

    print("\nAccuracy by method, averaged over walk-forward weeks")
    print(summarise(res).to_string())
    print("\nBy position")
    print(res.groupby("position")[["model_pair", "last4_pair"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
