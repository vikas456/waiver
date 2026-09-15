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
    "availability": ["games_played"],
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


def relative_drivers(drivers: pd.DataFrame) -> pd.DataFrame:
    """Re-centre contributions against the other players being compared.

    Users want to know why B beat A, not why B is above a league-wide average
    they never see. Subtracting the group mean makes every bar read as "this
    is what separates him from the others in this comparison".
    """
    return drivers - drivers.mean(axis=0)


# ---------------------------------------------------------------------------
# Prose
# ---------------------------------------------------------------------------

PHRASES = {
    "opportunity": ("commands a larger share of his offence's targets",
                    "sees a smaller share of his offence's targets"),
    "route_role": ("is on the field and running routes far more often",
                   "runs routes on too few dropbacks to hold a steady floor"),
    "goal_line": ("owns the goal-line work near the end zone",
                  "rarely gets the ball inside the five"),
    "efficiency": ("is producing more than his opportunities alone would suggest",
                   "has been inefficient with the touches he does get"),
    "td_regression": ("has scored close to what his usage supports",
                      "has scored far more than his usage supports"),
    "trend": ("has been gaining role over the past few weeks",
              "has been losing role over the past few weeks"),
    "production": ("has been putting up bigger stat lines",
                   "has been putting up smaller stat lines"),
    "offense": ("plays in a faster, more productive offence",
                "is stuck in an offence that does not generate enough volume"),
    "schedule": ("draws a friendlier set of remaining defences",
                 "faces a harder set of remaining defences"),
    "availability": ("has a longer track record this season",
                     "has a thin sample to judge from"),
    "prior": ("profiles well for his role", "profiles poorly for his role"),
}


def _fmt_pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.0f}%"


def describe_player(row: pd.Series, rel: pd.Series, rank: int,
                    others: list[str], stats: dict) -> str:
    """One paragraph explaining this player's position in this comparison."""
    ordered = rel.reindex(rel.abs().sort_values(ascending=False).index)
    top = [g for g in ordered.index[:3] if abs(ordered[g]) > 0.15]
    name = row["player_name"].split()[-1]

    clauses = []
    for g in top:
        up, down = PHRASES.get(g, ("rates well here", "rates poorly here"))
        clauses.append(up if ordered[g] > 0 else down)

    if not clauses:
        body = f"{name} sits close to the middle of this group on every factor the model weighs."
    elif rank == 1:
        body = f"{name} ranks first because he {clauses[0]}"
        if len(clauses) > 1:
            body += f" and {clauses[1]}"
        body += "."
    else:
        lead = clauses[0]
        body = f"{name} {lead}"
        if len(clauses) > 1:
            body += f", and {clauses[1]}"
        body += f", which is what drops him to {rank}."

    # The touchdown warning is worth stating explicitly, because scoring luck
    # is the most common reason a hot free agent is a trap.
    td_gap = stats.get("td_oe")
    if td_gap is not None and td_gap > 1.5:
        body += (f" He has {stats.get('actual_td', 0):.0f} touchdowns against "
                 f"{stats.get('expected_td', 0):.1f} expected, the kind of gap that "
                 "usually closes rather than continues.")
    elif td_gap is not None and td_gap < -1.5:
        body += (" He has scored less than his opportunities deserve, so some "
                 "positive regression is likely rather than priced in already.")

    if others:
        body += f" Compared with {' and '.join(others)}, that is the difference."
    return body


def comparison_summary(ranked: pd.DataFrame, rel: pd.DataFrame) -> str:
    """The headline verdict across the whole comparison."""
    top = ranked.iloc[0]
    gap = top["proj_ppg"] - ranked.iloc[1]["proj_ppg"] if len(ranked) > 1 else 0.0
    vorp_gap = top["vorp"] - ranked.iloc[1]["vorp"] if len(ranked) > 1 else 0.0

    if abs(vorp_gap) < 0.4:
        verdict = (f"{top['player_name']} and {ranked.iloc[1]['player_name']} are "
                   "close enough that either is defensible.")
    elif gap < 0 <= vorp_gap:
        verdict = (f"{top['player_name']} projects for fewer raw points but ranks "
                   "first because his position is thinner, so he replaces a worse "
                   "player on your bench.")
    else:
        verdict = (f"{top['player_name']} is the pick, by about "
                   f"{abs(gap):.1f} points a game over "
                   f"{ranked.iloc[1]['player_name']}.")

    spread = top.get("p90", 0) - top.get("p10", 0)
    if spread > 11:
        verdict += (" His range of outcomes is wide, so take him if you need "
                    "upside and the safer option if you are protecting a lead.")
    return verdict


def range_sensitivity(ranked_a: pd.DataFrame, ranked_b: pd.DataFrame,
                      label_a: str, label_b: str) -> str | None:
    """Say so when the answer changes under a different week range.

    Schedule adjustment is per week, so a player can be the right pick for the
    rest of the season and the wrong one for the fantasy playoffs alone. No
    other tool surfaces this because none of them let you change the range.
    """
    if ranked_a.empty or ranked_b.empty:
        return None
    if ranked_a.iloc[0]["player_id"] == ranked_b.iloc[0]["player_id"]:
        return None
    return (f"Worth noting: over {label_b} rather than {label_a}, "
            f"{ranked_b.iloc[0]['player_name']} moves ahead of "
            f"{ranked_a.iloc[0]['player_name']} on schedule alone.")
