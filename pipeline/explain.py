"""Explanation layer.

Two rules shape everything here.

First, the explanation must come from the same numbers that produced the
ranking. Anything else is a story written after the fact, which is what most
ranking sites publish. Contributions are taken straight from SHAP values on
the fitted models, aggregated into groups a person can reason about.

Second, the explanation must be comparative. The product answers "which of
these should I pick up", so the reasoning has to say why this player beat
those specific players, not describe him in isolation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import DRIVER_LABELS

# Which feature belongs to which driver group. Matching is by substring so
# that every rolling and shrunk variant of a metric lands in the right bucket.
DRIVER_PATTERNS = {
    "opportunity": ["target_share", "air_yards_share", "wopr", "carry_share"],
    "route_role": ["route_participation", "snap_share", "yards_per_route"],
    "goal_line": ["gl_touch_share", "rz_touch_share"],
    "efficiency": ["yac_oe", "ryoe", "catch_rate_oe", "cpoe", "adot", "epa"],
    "td_regression": ["td_oe", "expected_td"],
    "trend": ["_trend", "role_change", "weeks_since_change"],
    "production": ["attempts", "completions", "pass_yd", "pass_td", "interception",
                   "rush_yd", "rush_td", "reception", "rec_yd", "rec_td",
                   "fantasy_points", "actual_td"],
    "offense": ["team_proe", "team_off_epa", "team_neutral_plays",
                "team_sack_rate", "implied_total", "is_favourite"],
    "schedule": ["def_factor", "spread", "is_dome"],
    # How many games the model has to judge him on, not whether he plays:
    # missed games are their own factor, worked out on the site.
    "sample": ["games_played"],
}


def _group_for(feature: str) -> str:
    for group, patterns in DRIVER_PATTERNS.items():
        if any(p in feature for p in patterns):
            return group
    return "prior"


def _log_link(model: xgb.Booster) -> bool:
    """Poisson models add their contributions up on the log scale."""
    return json.loads(model.save_config())["learner"]["objective"]["name"] == "count:poisson"


def _to_output_scale(contrib: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Share a log-scale model's movement from its baseline among the drivers.

    A Poisson model predicts exp(bias + sum of contributions), so each
    contribution is a log ratio, not a count. Splitting the real change,
    exp(bias + total) - exp(bias), in proportion to the log contributions
    keeps the drivers additive and in the stat's own units.
    """
    total = contrib.sum(axis=1)
    change = np.exp(bias + total) - np.exp(bias)
    # Where the contributions cancel out, the ratio tends to exp(bias).
    scale = np.divide(change, total, out=np.exp(bias), where=np.abs(total) > 1e-9)
    return contrib * scale[:, None]


def shap_drivers(df: pd.DataFrame, bundle: dict, variance: dict,
                 scoring_weights: dict) -> pd.DataFrame:
    """Per-row driver contributions, expressed in fantasy points.

    SHAP values are computed on each component model, converted to points
    using the scoring format, then summed into groups. The result is additive
    and in the units the user sees, so the bars genuinely add up to the gap
    between a player and the field.
    """
    cols = bundle["features"]
    groups = sorted(set(DRIVER_LABELS.keys()))
    out = pd.DataFrame(0.0, index=df.index, columns=groups)

    for pos, models in bundle["models"].items():
        mask = (df["position"] == pos).to_numpy()
        if not mask.any():
            continue
        d = xgb.DMatrix(df.loc[mask, cols].astype(float),
                        feature_names=cols, missing=np.nan)
        for stat, model in models.items():
            weight = scoring_weights.get(stat, 0.0)
            if weight == 0.0:
                continue
            raw = model.predict(d, pred_contribs=True)
            # Last column is the bias term, which is the baseline rather than
            # a driver, so it is dropped.
            contrib, bias = raw[:, :-1], raw[:, -1]
            if _log_link(model):
                contrib = _to_output_scale(contrib, bias)
            contrib = contrib * weight
            for j, feature in enumerate(cols):
                out.loc[mask, _group_for(feature)] += contrib[:, j]
    return out
