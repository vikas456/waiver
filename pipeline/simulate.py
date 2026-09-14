"""Simulation, availability and value over replacement.

A single projected number hides the thing that actually decides waiver claims:
how wide the range of outcomes is. A back with a 40% chance of taking over a
backfield and a 40% chance of losing his job is a different asset from a
receiver who will quietly produce eleven points every week, even when both
project to the same mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import replacement_rank
from .scoring import score_components

RNG = np.random.default_rng(20260913)

# Baseline per-game probability that a player misses the next game, before any
# adjustment. Running backs take more contact and miss more time.
BASE_MISS_RATE = {"QB": 0.055, "RB": 0.105, "WR": 0.075, "TE": 0.085}

# How an injury designation changes the odds of playing.
DESIGNATION_PLAY_PROB = {
    "healthy": 1.0, "questionable": 0.72, "doubtful": 0.22,
    "out": 0.0, "ir": 0.0, "suspended": 0.0,
}


def availability(position: str, age: float, games_missed_last_2y: int,
                 snap_load: float, designation: str = "healthy") -> float:
    """Probability a player is available for a given future game.

    A hazard-style estimate rather than a coin flip: age, recent injury
    history and workload all raise the rate, and a current designation
    overrides everything for the coming week.
    """
    base = BASE_MISS_RATE.get(position, 0.08)
    age_factor = 1.0 + max(0.0, (age - 27.0)) * 0.06
    history_factor = 1.0 + min(games_missed_last_2y, 12) * 0.045
    load_factor = 1.0 + max(0.0, snap_load - 0.75) * 0.25
    miss = min(0.85, base * age_factor * history_factor * load_factor)
    prob = 1.0 - miss
    if designation and designation.lower() in DESIGNATION_PLAY_PROB:
        prob = min(prob, DESIGNATION_PLAY_PROB[designation.lower()])
    return float(np.clip(prob, 0.0, 1.0))


def simulate_player(week_means: np.ndarray, week_sds: np.ndarray,
                    play_probs: np.ndarray, n: int = 4000,
                    rng: np.random.Generator = RNG) -> dict:
    """Simulate a week-range outcome for one player.

    Each week is drawn independently: first whether he plays, then how he
    performs given that he plays. Outcomes are drawn from a gamma so the
    distribution is right-skewed and cannot go negative, which matches how
    fantasy scoring actually behaves — the ceiling is far from the mean but
    the floor is bounded at roughly zero.
    """
    weeks = len(week_means)
    if weeks == 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "games": 0.0}

    means = np.clip(week_means, 0.05, None)
    sds = np.clip(week_sds, 0.5, None)
    shape = (means / sds) ** 2
    scale = sds**2 / means

    draws = rng.gamma(shape[None, :], scale[None, :], size=(n, weeks))
    played = rng.random((n, weeks)) < play_probs[None, :]
    weekly = draws * played

    total = weekly.sum(axis=1)
    games = played.sum(axis=1)
    per_game = np.divide(total, np.maximum(games, 1))
    per_game[games == 0] = 0.0

    return {
        "mean": float(per_game.mean()),
        "p10": float(np.percentile(per_game, 10)),
        "p50": float(np.percentile(per_game, 50)),
        "p90": float(np.percentile(per_game, 90)),
        "total_mean": float(total.mean()),
        "total_p10": float(np.percentile(total, 10)),
        "total_p90": float(np.percentile(total, 90)),
        "games": float(games.mean()),
    }


def sd_for(mean: float, position: str, variance: dict) -> float:
    params = variance.get(position, {"sd_intercept": 4.0, "sd_slope": 0.5})
    return params["sd_intercept"] + params["sd_slope"] * max(mean, 0.0)


# ---------------------------------------------------------------------------
# Value over replacement
# ---------------------------------------------------------------------------

def replacement_levels(projections: pd.DataFrame, league_size: int,
                       points_col: str = "proj_ppg") -> dict:
    """Points per game of the best freely available player at each position.

    This is what makes cross-position comparison meaningful. A tight end at
    eleven points can be worth more than a receiver at twelve and a half,
    because the tight end you would otherwise start is far worse than the
    receiver you would otherwise start. Without this, every ranking that mixes
    positions is just sorting by raw points, which is wrong.
    """
    levels = {}
    for pos, grp in projections.groupby("position"):
        rank = replacement_rank(pos, league_size)
        ordered = grp[points_col].sort_values(ascending=False).to_numpy()
        if len(ordered) == 0:
            levels[pos] = 0.0
        elif len(ordered) >= rank:
            # Average a small band around the cut line rather than taking one
            # player, so the baseline is not hostage to a single outlier.
            lo, hi = max(0, rank - 3), min(len(ordered), rank + 2)
            levels[pos] = float(np.mean(ordered[lo:hi]))
        else:
            levels[pos] = float(ordered[-1])
    return levels


def add_vorp(projections: pd.DataFrame, league_size: int,
             points_col: str = "proj_ppg") -> pd.DataFrame:
    levels = replacement_levels(projections, league_size, points_col)
    out = projections.copy()
    out["replacement_ppg"] = out["position"].map(levels)
    out["vorp"] = out[points_col] - out["replacement_ppg"]
    return out


def bye_weeks_in_range(schedule: pd.DataFrame, team: str, season: int,
                       weeks: range) -> list[int]:
    """Weeks in the requested range where a team does not play.

    A bye is a zero that a per-game average hides, so it is surfaced
    separately rather than silently folded into the mean.
    """
    played = set(
        schedule[
            (schedule["season"] == season)
            & ((schedule["home_team"] == team) | (schedule["away_team"] == team))
        ]["week"].tolist()
    )
    return [w for w in weeks if w not in played]
