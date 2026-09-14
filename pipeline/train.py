"""Model training.

Two models are trained per position and they do different jobs.

The component model predicts the pieces of a stat line: receptions, receiving
yards, carries, touchdowns and so on. Predicting components rather than points
is what lets one model serve every scoring format, and it makes the
explanation honest, because you can say "the targets are real, the touchdown
rate is not" instead of gesturing at a single number.

The ranking model is trained directly on the thing the product outputs: given
a set of players, get the order right. Squared error on points and accuracy on
ordering are not the same objective, and optimising the one you actually ship
is worth roughly a point of pairwise accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import POSITIONS, STAT_COMPONENTS
from .features import feature_columns

MODEL_DIR = Path("models")

COMPONENT_PARAMS = {
    "objective": "count:poisson",
    "max_depth": 5,
    "eta": 0.04,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 12,
    "reg_lambda": 2.0,
}

CONTINUOUS_PARAMS = {
    **COMPONENT_PARAMS,
    "objective": "reg:squarederror",
}

RANK_PARAMS = {
    "objective": "rank:pairwise",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 15,
    "reg_lambda": 3.0,
}

# Counting stats get a Poisson objective because they are non-negative and
# right-skewed; yardage is continuous.
COUNT_STATS = {"pass_td", "rush_td", "rec_td", "reception", "interception",
               "fumble_lost", "two_point"}


def _dmatrix(df: pd.DataFrame, cols: list[str], label=None, weight=None):
    return xgb.DMatrix(df[cols].astype(float), label=label, weight=weight,
                       feature_names=cols, missing=np.nan)


def train_components(df: pd.DataFrame, rounds: int = 400) -> dict:
    """One model per position per stat component."""
    cols = feature_columns(df)
    models: dict[str, dict[str, xgb.Booster]] = {}

    for pos in POSITIONS:
        sub = df[df["position"] == pos]
        sub = sub[sub["games_played"] >= 1]
        if len(sub) < 200:
            continue
        models[pos] = {}
        for stat in STAT_COMPONENTS:
            if stat not in sub.columns:
                continue
            y = sub[stat].fillna(0).clip(lower=0)
            if y.sum() < 50:
                continue
            params = COMPONENT_PARAMS if stat in COUNT_STATS else CONTINUOUS_PARAMS
            dtrain = _dmatrix(sub, cols, label=y, weight=sub["season_weight"])
            models[pos][stat] = xgb.train(params, dtrain, num_boost_round=rounds)
    return {"models": models, "features": cols}


def train_ranker(df: pd.DataFrame, rounds: int = 300) -> dict:
    """Pairwise ranking model, grouped by position and week.

    Each group is the set of players at one position in one week, so the model
    learns to order players against their real competition rather than against
    the league as a whole.
    """
    cols = feature_columns(df)
    models = {}
    for pos in POSITIONS:
        sub = df[(df["position"] == pos) & (df["games_played"] >= 1)].copy()
        if len(sub) < 200:
            continue
        sub = sub.sort_values(["season", "week"])
        groups = sub.groupby(["season", "week"], observed=True).size().to_numpy()
        # Relevance grades rather than raw points: the ranker only needs to
        # know who finished in which tier that week.
        sub["grade"] = (
            sub.groupby(["season", "week"], observed=True)["fantasy_points_ppr"]
            .transform(lambda s: pd.qcut(s.rank(method="first"), 5,
                                         labels=False, duplicates="drop"))
            .fillna(0)
        )
        dtrain = _dmatrix(sub, cols, label=sub["grade"])
        dtrain.set_group(groups)
        models[pos] = xgb.train(RANK_PARAMS, dtrain, num_boost_round=rounds)
    return {"models": models, "features": cols}


def fit_variance(df: pd.DataFrame, component_bundle: dict) -> dict:
    """Fit how much a projection actually varies around its mean.

    A floor and a ceiling are only useful if they are calibrated, so the
    spread is estimated from real residuals per position rather than assumed.
    Volatility scales with projected volume, so the model learns a slope on
    the mean rather than a single number.
    """
    preds = predict_components(df, component_bundle)
    from .scoring import score_components
    mean_pts = score_components(preds, "ppr", positions=df["position"])
    resid = df["fantasy_points_ppr"].to_numpy() - mean_pts.to_numpy()

    out = {}
    frame = pd.DataFrame({"position": df["position"].to_numpy(),
                          "mean": mean_pts.to_numpy(), "resid": resid}).dropna()
    for pos, grp in frame.groupby("position"):
        bins = pd.qcut(grp["mean"].rank(method="first"), 5, labels=False, duplicates="drop")
        sd_by_bin = grp.groupby(bins)["resid"].std()
        centres = grp.groupby(bins)["mean"].mean()
        if len(sd_by_bin) >= 2:
            slope, intercept = np.polyfit(centres.to_numpy(), sd_by_bin.to_numpy(), 1)
        else:
            slope, intercept = 0.5, 4.0
        out[pos] = {"sd_intercept": float(max(intercept, 1.0)),
                    "sd_slope": float(max(slope, 0.05))}
    return out


def predict_components(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Predicted stat line for every row."""
    cols = bundle["features"]
    out = pd.DataFrame(0.0, index=df.index, columns=STAT_COMPONENTS)
    for pos, models in bundle["models"].items():
        mask = df["position"] == pos
        if not mask.any():
            continue
        d = _dmatrix(df[mask], cols)
        for stat, model in models.items():
            out.loc[mask, stat] = model.predict(d)
    return out


def save(bundle_components: dict, bundle_rank: dict, variance: dict,
         path: Path = MODEL_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for pos, models in bundle_components["models"].items():
        for stat, model in models.items():
            model.save_model(str(path / f"comp_{pos}_{stat}.json"))
    for pos, model in bundle_rank["models"].items():
        model.save_model(str(path / f"rank_{pos}.json"))
    (path / "meta.json").write_text(json.dumps({
        "features": bundle_components["features"],
        "variance": variance,
        "positions": list(bundle_components["models"].keys()),
        "components": {p: list(m.keys()) for p, m in bundle_components["models"].items()},
    }, indent=2))


def load(path: Path = MODEL_DIR) -> tuple[dict, dict, dict]:
    meta = json.loads((path / "meta.json").read_text())
    comp = {"features": meta["features"], "models": {}}
    for pos, stats in meta["components"].items():
        comp["models"][pos] = {}
        for stat in stats:
            b = xgb.Booster()
            b.load_model(str(path / f"comp_{pos}_{stat}.json"))
            comp["models"][pos][stat] = b
    rank = {"features": meta["features"], "models": {}}
    for pos in meta["positions"]:
        f = path / f"rank_{pos}.json"
        if f.exists():
            b = xgb.Booster()
            b.load_model(str(f))
            rank["models"][pos] = b
    return comp, rank, meta["variance"]
