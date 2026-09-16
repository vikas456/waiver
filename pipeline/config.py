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

# Kickers and team defences are projected by a different route: see
# pipeline/kdst.py. Almost nothing they do repeats week to week, so there is
# no usage to model, and the feature pipeline above deliberately ignores them.
KDST_POSITIONS = ["K", "DEF"]

# What a kicker or a defence is credited with in a week. The site multiplies
# these by the league's values, exactly as it does for the skill positions.
KDST_COMPONENTS = [
    "fg_0_39", "fg_40_49", "fg_50", "pat", "fg_miss",
    "sack", "interception_def", "fumble_recovery", "def_td", "safety", "block",
]

REGULAR_SEASON_WEEKS = 18
FANTASY_PLAYOFF_WEEKS = (15, 17)

# From this week on, projections start from form built within the current
# season. Before it no player has a game with a game behind it, so they start
# from earlier seasons instead; see project.early_season_form.
FULL_FORM_FROM_WEEK = 3

# Sportsbooks price the coming week and usually the one after it, so a forecast
# may use a real betting line that far out and no further. Backtests are held
# to the same limit, since a later week's final line is information the live
# site never has.
LINES_KNOWN_AHEAD = 1

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
        "fg_0_39": 3.0, "fg_40_49": 4.0, "fg_50": 5.0, "pat": 1.0, "fg_miss": -1.0,
        "sack": 1.0, "interception_def": 2.0, "fumble_recovery": 2.0,
        "def_td": 6.0, "safety": 2.0, "block": 2.0,
    },
    "half_ppr": {
        "label": "Half PPR",
        "pass_yd": 0.04, "pass_td": 4.0, "interception": -2.0,
        "rush_yd": 0.1, "rush_td": 6.0,
        "reception": 0.5, "rec_yd": 0.1, "rec_td": 6.0,
        "fumble_lost": -2.0, "two_point": 2.0,
        "te_premium": 0.0,
        "fg_0_39": 3.0, "fg_40_49": 4.0, "fg_50": 5.0, "pat": 1.0, "fg_miss": -1.0,
        "sack": 1.0, "interception_def": 2.0, "fumble_recovery": 2.0,
        "def_td": 6.0, "safety": 2.0, "block": 2.0,
    },
    "standard": {
        "label": "Standard",
        "pass_yd": 0.04, "pass_td": 4.0, "interception": -2.0,
        "rush_yd": 0.1, "rush_td": 6.0,
        "reception": 0.0, "rec_yd": 0.1, "rec_td": 6.0,
        "fumble_lost": -2.0, "two_point": 2.0,
        "te_premium": 0.0,
        "fg_0_39": 3.0, "fg_40_49": 4.0, "fg_50": 5.0, "pat": 1.0, "fg_miss": -1.0,
        "sack": 1.0, "interception_def": 2.0, "fumble_recovery": 2.0,
        "def_td": 6.0, "safety": 2.0, "block": 2.0,
    },
}

# Points allowed is scored in bands, not per point, so a defence's projection
# carries the chance of landing in each one and the site applies its league's
# values. Upper bound of each band, and what it is worth by default.
PA_TIERS = [(0, 10.0), (6, 7.0), (13, 4.0), (20, 1.0), (27, 0.0), (34, -1.0), (99, -4.0)]

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
REPLACEMENT_RANK_PER_TEAM = {"QB": 1.5, "RB": 2.5, "WR": 3.0, "TE": 1.2,
                             # One of each starts and almost nobody carries a
                             # second, so the replacement is the last starter.
                             "K": 1.0, "DEF": 1.0}


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
    "td_oe": 40.0,
    "yards_per_route": 8.0,
    "cpoe": 9.0,
    # Box-score production, alongside the usage above. Volume settles within
    # a few games; touchdowns and interceptions barely settle at all.
    "attempts": 3.0,
    "completions": 4.0,
    "reception": 4.0,
    "pass_yd": 5.0,
    "rush_yd": 5.0,
    "rec_yd": 5.0,
    "fantasy_points_ppr": 5.0,
    "pass_td": 12.0,
    "interception": 15.0,
    "rush_td": 20.0,
    "rec_td": 20.0,
}

ROLLING_WINDOWS = [3, 5]

# Share of each projection taken from FantasyPros' rest-of-season consensus.
# Set from the walk-forward backtest, which scores the model, the consensus
# and blends of the two on the same weeks. Before week three the model knows
# least and the market knows about offseason moves and rookies, so it gets
# more say: a 0.7 share did best in weeks one and two on both 2025 and the
# held-out 2024, about half a point of pairwise accuracy above an even blend
# and above the consensus alone. From week three a light blend stays best.
MARKET_WEIGHT = 0.25
MARKET_WEIGHT_EARLY = 0.7

# Human-readable names for the explanation layer, keyed by the feature group
# that SHAP contributions are aggregated into.
DRIVER_LABELS = {
    "opportunity": "Opportunity share",
    "route_role": "Route participation",
    "goal_line": "Goal-line share",
    "efficiency": "Efficiency over expected",
    "td_regression": "Touchdown regression",
    "trend": "Usage trend",
    "production": "Recent production",
    "market": "Expert consensus",
    "offense": "Offensive environment",
    "schedule": "Remaining schedule",
    "coverage": "Coverage matchup",
    "sample": "Sample size",
    # Worked out on the site from each week's chance of playing and the
    # league's replacement level, rather than from the model's SHAP values.
    "missed_games": "Missed games",
    "position": "Position scarcity",
    "prior": "Baseline for role",
}
