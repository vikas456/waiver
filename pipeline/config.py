"""Central configuration for the projection pipeline."""

# Seasons used for training. Five years back from the current season.
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
CURRENT_SEASON = 2026

# Recency weighting: each season back from the most recent is worth this much
# less during training. 0.85 keeps five years useful without letting a 2021
# passing environment outvote last month.
SEASON_DECAY = 0.85

# FTN charting data only exists from 2022 onward.
FTN_FIRST_SEASON = 2022

POSITIONS = ["QB", "RB", "WR", "TE"]

REGULAR_SEASON_WEEKS = 18
FANTASY_PLAYOFF_WEEKS = (15, 17)

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
# The model never predicts "fantasy points". It predicts stat components, and
# scoring is applied afterwards. That is what lets one set of projections serve
# PPR, half PPR and standard without retraining.

SCORING_FORMATS = {
    "ppr": {
        "label": "Full PPR",
        "pass_yd": 0.04, "pass_td": 4.0, "interception": -2.0,
        "rush_yd": 0.1, "rush_td": 6.0,
        "reception": 1.0, "rec_yd": 0.1, "rec_td": 6.0,
        "fumble_lost": -2.0, "two_point": 2.0,
        "te_premium": 0.0,
    },
    "half_ppr": {
        "label": "Half PPR",
        "pass_yd": 0.04, "pass_td": 4.0, "interception": -2.0,
        "rush_yd": 0.1, "rush_td": 6.0,
        "reception": 0.5, "rec_yd": 0.1, "rec_td": 6.0,
        "fumble_lost": -2.0, "two_point": 2.0,
        "te_premium": 0.0,
    },
    "standard": {
        "label": "Standard",
        "pass_yd": 0.04, "pass_td": 4.0, "interception": -2.0,
        "rush_yd": 0.1, "rush_td": 6.0,
        "reception": 0.0, "rec_yd": 0.1, "rec_td": 6.0,
        "fumble_lost": -2.0, "two_point": 2.0,
        "te_premium": 0.0,
    },
}

# The stat components the model predicts. Everything downstream is derived.
STAT_COMPONENTS = [
    "pass_yd", "pass_td", "interception",
    "rush_yd", "rush_td",
    "reception", "rec_yd", "rec_td",
    "fumble_lost", "two_point",
]

# ---------------------------------------------------------------------------
# Replacement level (for value over replacement)
# ---------------------------------------------------------------------------
# Rank at each position that counts as "freely available on waivers", by league
# size. Derived from typical starting requirements plus bench depth: in a
# 12-team league roughly 30 RBs and 36 WRs are rostered, so the 31st RB is the
# guy you could pick up instead of the player being evaluated.
REPLACEMENT_RANK_PER_TEAM = {"QB": 1.5, "RB": 2.5, "WR": 3.0, "TE": 1.2}


def replacement_rank(position: str, league_size: int) -> int:
    """The positional rank that defines replacement level for a league size."""
    return max(1, int(round(REPLACEMENT_RANK_PER_TEAM[position] * league_size)))


# ---------------------------------------------------------------------------
# Feature stability
# ---------------------------------------------------------------------------
# Games of sample needed before a metric carries roughly half its full signal.
# Used to set Bayesian shrinkage strength: TD rate essentially never stabilises
# inside a season, so it gets shrunk hard toward a prior. Target share firms up
# in about four games and is trusted quickly.
STABILISATION_GAMES = {
    "snap_share": 3.0,
    "route_participation": 3.0,
    "target_share": 4.0,
    "air_yards_share": 5.0,
    "wopr": 4.5,
    "carry_share": 3.5,
    "rz_touch_share": 8.0,
    "gl_touch_share": 10.0,
    "adot": 6.0,
    "yac_oe": 9.0,
    "ryoe_per_carry": 11.0,
    "catch_rate_oe": 10.0,
    "td_rate": 40.0,
    "yards_per_route": 8.0,
    "cpoe": 9.0,
}

ROLLING_WINDOWS = [3, 5]

# Human-readable names for the explanation layer, keyed by the feature group
# that SHAP contributions are aggregated into.
DRIVER_LABELS = {
    "opportunity": "Opportunity share",
    "route_role": "Route participation",
    "goal_line": "Goal-line share",
    "efficiency": "Efficiency over expected",
    "td_regression": "Touchdown regression",
    "trend": "Usage trend",
    "offense": "Offensive environment",
    "schedule": "Remaining schedule",
    "coverage": "Coverage matchup",
    "availability": "Availability",
    "prior": "Baseline for role",
}
