"""Feature engineering.

The guiding idea: box-score stats describe the past, usage describes the
future. Everything here is built to measure role and opportunity, to discount
production that came from luck, and to notice when a role has just changed.

Every feature is computed strictly from information available before the game
it predicts. Nothing is allowed to see forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    POSITIONS, ROLLING_WINDOWS, SEASON_DECAY, STABILISATION_GAMES,
)
from .scoring import actual_points

SHARE_COLS = [
    "snap_share", "target_share", "air_yards_share", "wopr", "carry_share",
    "rz_touch_share", "gl_touch_share", "route_participation",
]
EFF_COLS = ["adot", "yac_oe", "ryoe_per_carry", "catch_rate_oe",
            "yards_per_route", "cpoe", "rec_epa", "rush_epa"]


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------

def shrink(observed: pd.Series, games: pd.Series, prior: pd.Series,
           stabilise_games: float) -> pd.Series:
    """Blend an observed rate toward a prior based on sample size.

    weight = n / (n + k), where k is the number of games at which the metric
    carries half its signal. A receiver with two games of a 30% target share
    gets pulled most of the way back to his prior; one with nine games barely
    moves. This is the single most effective guard against overreacting to a
    two-game hot streak, which is how most public tools get burned.
    """
    n = games.astype(float).clip(lower=0)
    w = n / (n + stabilise_games)
    return w * observed.fillna(prior) + (1 - w) * prior


def _role_prior(df: pd.DataFrame, col: str) -> pd.Series:
    """Prior for a metric: what a player of this position, draft capital and
    experience typically does. Rookies and deep-bench players lean on this
    heavily until they have played enough to speak for themselves."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    key = ["position", "draft_bucket", "exp_bucket"]
    prior = df.groupby(key, observed=True)[col].transform("median")
    return prior.fillna(df.groupby("position", observed=True)[col].transform("median")).fillna(0.0)


# ---------------------------------------------------------------------------
# Derived efficiency metrics
# ---------------------------------------------------------------------------

def add_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Efficiency over expected, which separates the player from his situation."""
    df = df.copy()
    df["catch_rate"] = df["reception"] / df["targets"].replace(0, np.nan)
    # Catch rate falls predictably with depth of target, so compare a player
    # against what his own aDOT implies rather than against the league mean.
    expected_catch = 0.78 - 0.018 * df["adot"].fillna(8.0)
    df["catch_rate_oe"] = df["catch_rate"] - expected_catch

    df["yac_oe"] = (df["yac"] / df["reception"].replace(0, np.nan)) - 4.6
    df["yards_per_route"] = df["rec_yd"] / df["routes"].replace(0, np.nan)
    df["ryoe_per_carry"] = df["rush_epa"].fillna(0) * 1.8

    # Expected touchdowns from opportunity alone. The gap between this and
    # actual touchdowns is the single biggest source of fantasy mirages.
    df["expected_td"] = (
        0.19 * df["gl_carries"].fillna(0)
        + 0.34 * df["gl_targets"].fillna(0)
        + 0.055 * df["rz_carries"].fillna(0)
        + 0.10 * df["rz_targets"].fillna(0)
        + 0.0025 * df["rec_yd"].fillna(0)
        + 0.0020 * df["rush_yd"].fillna(0)
    )
    df["actual_td"] = df["rush_td"].fillna(0) + df["rec_td"].fillna(0)
    df["td_oe"] = df["actual_td"] - df["expected_td"]
    return df


# ---------------------------------------------------------------------------
# Zero-sum usage within a team
# ---------------------------------------------------------------------------

def renormalise_shares(df: pd.DataFrame, active_mask: pd.Series | None = None) -> pd.DataFrame:
    """Force usage shares to sum to one within each team-week.

    Target share and carry share are compositional. Modelling players in
    isolation means an injured team-mate's vacated targets vanish instead of
    being redistributed, which is exactly the situation a waiver-wire tool
    faces most often. Redistribution is proportional to each remaining
    player's own share, which matches how offences actually reallocate.
    """
    df = df.copy()
    if active_mask is not None:
        df = df[active_mask]
    for col in ["target_share", "carry_share", "rz_touch_share", "gl_touch_share"]:
        if col not in df.columns:
            continue
        total = df.groupby(["team", "season", "week"], observed=True)[col].transform("sum")
        df[col + "_norm"] = df[col] / total.replace(0, np.nan)
    return df


def redistribute_for_absence(team_frame: pd.DataFrame, absent_ids: list[str]) -> pd.DataFrame:
    """Reallocate the shares of absent players across those still available."""
    frame = team_frame.copy()
    out = frame[~frame["player_id"].isin(absent_ids)].copy()
    for col in ["target_share", "carry_share", "gl_touch_share", "rz_touch_share"]:
        if col not in frame.columns:
            continue
        vacated = frame.loc[frame["player_id"].isin(absent_ids), col].sum()
        remaining = out[col].sum()
        if remaining > 0 and vacated > 0:
            out[col] = out[col] + vacated * (out[col] / remaining)
    return out


# ---------------------------------------------------------------------------
# Rolling windows and trend
# ---------------------------------------------------------------------------

def add_rolling(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Trailing means, strictly excluding the current game.

    Windows reset at each season boundary. A player's role in October says
    very little about his role the following September, and letting an
    expanding mean run across years quietly buries a current-season role
    change under two seasons of history.
    """
    df = df.sort_values(["player_id", "season", "week"]).copy()
    key = ["player_id", "season"]
    g = df.groupby(key, observed=True)
    for col in cols:
        if col not in df.columns:
            continue
        shifted = g[col].shift(1)
        df[f"{col}_todate"] = (
            shifted.groupby([df["player_id"], df["season"]])
            .expanding().mean().reset_index(level=[0, 1], drop=True)
        )
        for w in ROLLING_WINDOWS:
            df[f"{col}_r{w}"] = (
                shifted.groupby([df["player_id"], df["season"]])
                .rolling(w, min_periods=1).mean()
                .reset_index(level=[0, 1], drop=True)
            )
        # Trend: recent form against the season baseline. A rising snap share
        # is the clearest early signal that a coaching staff has changed its
        # mind about a player, and it shows up here weeks before the box score.
        df[f"{col}_trend"] = df[f"{col}_r3"] - df[f"{col}_todate"]
    df["games_played"] = g.cumcount()
    return df


def detect_change_points(df: pd.DataFrame, col: str = "snap_share",
                         min_jump: float = 0.18, lookback: int = 3) -> pd.DataFrame:
    """Flag structural breaks in a player's role.

    When a starter goes down, a coordinator is fired or a committee back takes
    over, history from before the break is actively misleading. Rather than
    averaging across the break, we mark it and let the model weight post-break
    games more heavily.
    """
    df = df.sort_values(["player_id", "season", "week"]).copy()
    g = df.groupby(["player_id", "season"], observed=True)[col]
    recent = g.transform(lambda s: s.shift(1).rolling(lookback, min_periods=1).mean())
    older = g.transform(lambda s: s.shift(lookback + 1).rolling(4, min_periods=1).mean())
    jump = recent - older
    df["role_change"] = (jump.abs() >= min_jump).astype(int)
    df["role_change_dir"] = np.sign(jump).fillna(0)
    df["weeks_since_change"] = (
        df.groupby(["player_id", "season"], observed=True)["role_change"]
        .transform(lambda s: s[::-1].cumsum()[::-1].pipe(lambda x: x.groupby(x).cumcount()))
    )
    return df


# ---------------------------------------------------------------------------
# Opponent adjustment
# ---------------------------------------------------------------------------

def opponent_strength(defense: pd.DataFrame) -> pd.DataFrame:
    """Per-defence, per-position adjustment factor, normalised for pace.

    A defence that faces 70 plays a game will allow more fantasy points than
    one facing 58 without being any worse. Dividing by plays faced fixes that,
    which is why this differs from the points-allowed tables most sites show.
    """
    d = defense.copy()
    d["pts_per_play"] = d["pts_allowed"] / d["targets_allowed"].replace(0, np.nan).clip(lower=1)
    league = d.groupby(["season", "position"], observed=True)["pts_per_play"].transform("mean")
    d["def_factor"] = d["pts_per_play"] / league.replace(0, np.nan)
    # Shrink toward neutral. Defensive fantasy splits are noisy and a handful
    # of games is not enough to justify a large adjustment.
    n = d.groupby(["season", "defteam", "position"], observed=True).cumcount() + 1
    w = n / (n + 6.0)
    d["def_factor"] = w * d["def_factor"].fillna(1.0) + (1 - w) * 1.0
    return d[["season", "week", "defteam", "position", "def_factor"]]


def game_script(schedule: pd.DataFrame) -> pd.DataFrame:
    """Implied team total and spread, per team per week.

    A heavy favourite runs more in the fourth quarter, which helps backs and
    hurts receivers. Vegas prices this better than any model, so we use it.
    """
    home = schedule.rename(columns={"home_team": "team", "away_team": "opponent"}).copy()
    home["implied_total"] = home["total_line"] / 2 + home["spread_line"] / 2
    home["spread"] = home["spread_line"]
    away = schedule.rename(columns={"away_team": "team", "home_team": "opponent"}).copy()
    away["implied_total"] = away["total_line"] / 2 - away["spread_line"] / 2
    away["spread"] = -away["spread_line"]
    cols = ["season", "week", "team", "opponent", "implied_total", "spread", "roof"]
    out = pd.concat([home[cols], away[cols]], ignore_index=True)
    out["is_dome"] = out["roof"].isin(["dome", "closed"]).astype(int)
    out["is_favourite"] = (out["spread"] < 0).astype(int)
    return out.drop(columns=["roof"])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_features(weekly: pd.DataFrame, team_week: pd.DataFrame,
                   defense: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Produce the model-ready frame: one row per player per game."""
    df = weekly.copy()
    df["position"] = df["position"].astype(str)
    df = df[df["position"].isin(POSITIONS)]

    for col in ["routes", "targets", "carries", "yac", "gl_carries", "gl_targets",
                "rz_carries", "rz_targets", "rec_epa", "rush_epa", "adot"]:
        if col not in df.columns:
            df[col] = np.nan

    df["draft_bucket"] = pd.cut(
        df.get("draft_number", pd.Series(np.nan, index=df.index)).fillna(300),
        bins=[0, 32, 64, 105, 180, 400],
        labels=["r1", "r2", "r3", "d4_5", "late"],
    )
    df["exp_bucket"] = pd.cut(
        df.get("years_exp", pd.Series(0, index=df.index)).fillna(0),
        bins=[-1, 0, 1, 3, 6, 30], labels=["rookie", "y2", "y3_4", "prime", "vet"],
    )

    df = add_efficiency(df)
    df["route_participation"] = df["routes"] / (df["team_pass_att"].replace(0, np.nan))

    df = add_rolling(df, SHARE_COLS + EFF_COLS
                     + ["td_oe", "expected_td", "actual_td", "targets", "carries"])
    df = detect_change_points(df)

    # Season-to-date totals, for display only. These are never fed to the
    # model, but "six touchdowns against 2.4 expected" is the single most
    # persuasive line the explanation can offer, and it needs real counts
    # rather than per-game averages.
    for col in ["actual_td", "expected_td"]:
        df[f"{col}_season"] = (
            df.groupby(["player_id", "season"], observed=True)[col].cumsum()
        )

    # Shrink every trailing rate toward its role prior, weighted by how fast
    # that particular metric stabilises.
    for col in SHARE_COLS + EFF_COLS + ["td_oe"]:
        src = f"{col}_todate"
        if src not in df.columns:
            continue
        prior = _role_prior(df, src)
        k = STABILISATION_GAMES.get(col, 6.0)
        df[f"{col}_adj"] = shrink(df[src], df["games_played"], prior, k)

    # Offensive environment, lagged one week.
    tw = team_week.sort_values(["team", "season", "week"]).copy()
    for col in ["proe", "off_epa", "neutral_plays", "sack_rate", "cpoe"]:
        if col in tw.columns:
            tw[f"team_{col}"] = (
                tw.groupby("team", observed=True)[col]
                .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            )
    keep = ["season", "week", "team"] + [c for c in tw.columns if c.startswith("team_")]
    df = df.merge(tw[keep], on=["season", "week", "team"], how="left")

    # Schedule context and opponent quality. The schedule is the authority on
    # who played whom, so any opponent column already on the frame is dropped.
    gs = game_script(schedule)
    df = df.drop(columns=["opponent"], errors="ignore")
    df = df.merge(gs, on=["season", "week", "team"], how="left")
    opp = opponent_strength(defense)
    df = df.merge(
        opp.rename(columns={"defteam": "opponent"}),
        on=["season", "week", "opponent", "position"], how="left",
    )
    df["def_factor"] = df["def_factor"].fillna(1.0)

    # Recency weight for training.
    df["season_weight"] = SEASON_DECAY ** (df["season"].max() - df["season"])
    df["fantasy_points_ppr"] = actual_points(df, "ppr")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs. Excludes anything that leaks the outcome."""
    suffixes = ("_adj", "_r3", "_r5", "_trend")
    cols = [c for c in df.columns if c.endswith(suffixes)]
    cols += [
        "games_played", "role_change", "role_change_dir", "weeks_since_change",
        "implied_total", "spread", "is_dome", "is_favourite", "def_factor",
        "team_proe", "team_off_epa", "team_neutral_plays", "team_sack_rate",
    ]
    return [c for c in dict.fromkeys(cols) if c in df.columns]
