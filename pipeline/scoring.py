"""Turn predicted stat components into fantasy points.

Kept deliberately separate from the model so that one set of projections
serves every scoring format. The same functions run in Python here and are
mirrored in the browser, so server and client always agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SCORING_FORMATS, STAT_COMPONENTS


def score_components(
    components: pd.DataFrame,
    fmt: str = "ppr",
    te_premium: float = 0.0,
    positions: pd.Series | None = None,
) -> pd.Series:
    """Apply a scoring format to a frame of stat components.

    te_premium adds extra points per reception for tight ends only, which a
    surprising number of leagues use and no public tool bothers to support.
    """
    if fmt not in SCORING_FORMATS:
        raise ValueError(f"unknown scoring format: {fmt}")
    weights = SCORING_FORMATS[fmt]

    points = np.zeros(len(components), dtype=float)
    for stat in STAT_COMPONENTS:
        if stat in components.columns:
            points += components[stat].to_numpy(dtype=float) * weights[stat]

    if te_premium and positions is not None and "reception" in components.columns:
        is_te = (positions.to_numpy() == "TE").astype(float)
        points += components["reception"].to_numpy(dtype=float) * te_premium * is_te

    return pd.Series(points, index=components.index, name="fantasy_points")


def actual_points(weekly: pd.DataFrame, fmt: str = "ppr") -> pd.Series:
    """Score real historical weekly stat lines. Used to build training labels."""
    return score_components(weekly, fmt=fmt, positions=weekly.get("position"))
