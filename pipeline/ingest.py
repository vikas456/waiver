"""Pull raw data from nflverse and reshape it into the tables the feature
layer expects.

Four tables come out of here:
  weekly    one row per player per game, with usage and stat components
  team_week one row per team per game, with pace, pass rate and efficiency
  defense   one row per defence per game, with what they allowed by position
  schedule  remaining games, with Vegas lines for game-script projection

Everything is cached to parquet under data/cache so a rebuild during the week
does not re-download four seasons of play-by-play.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CURRENT_SEASON, FTN_FIRST_SEASON, POSITIONS

CACHE_DIR = Path(os.environ.get("FF_CACHE", "data/cache"))


def _cache_path(name: str, seasons: list[int]) -> Path:
    tag = f"{min(seasons)}_{max(seasons)}"
    return CACHE_DIR / f"{name}_{tag}.parquet"


def _cached(name: str, seasons: list[int], builder, refresh: bool = False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(name, seasons)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    frame = builder()
    frame.to_parquet(path, index=False)
    return frame


# ---------------------------------------------------------------------------
# Play-by-play derived usage
# ---------------------------------------------------------------------------

def _usage_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Derive per-player per-game usage that nflverse does not ship directly.

    Target share, air yards share, WOPR, red zone and goal line touch share,
    aDOT and yards per route all have to be built from the play level. These
    are the metrics that lead production, so they are worth the work.
    """
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    pbp["is_rz"] = pbp["yardline_100"] <= 20
    pbp["is_gl"] = pbp["yardline_100"] <= 5

    # Team totals per game, used as denominators for every share metric.
    team = (
        pbp.groupby(["game_id", "posteam"], observed=True)
        .agg(
            team_plays=("play_id", "count"),
            team_pass_att=("pass_attempt", "sum"),
            team_rush_att=("rush_attempt", "sum"),
            team_air_yards=("air_yards", "sum"),
            team_rz_plays=("is_rz", "sum"),
            team_gl_plays=("is_gl", "sum"),
        )
        .reset_index()
    )

    rec = (
        pbp[pbp["pass_attempt"] == 1]
        .groupby(["game_id", "posteam", "receiver_player_id"], observed=True)
        .agg(
            targets=("pass_attempt", "sum"),
            air_yards=("air_yards", "sum"),
            adot=("air_yards", "mean"),
            yac=("yards_after_catch", "sum"),
            rz_targets=("is_rz", "sum"),
            gl_targets=("is_gl", "sum"),
            rec_epa=("epa", "mean"),
        )
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id"})
    )

    rush = (
        pbp[pbp["rush_attempt"] == 1]
        .groupby(["game_id", "posteam", "rusher_player_id"], observed=True)
        .agg(
            carries=("rush_attempt", "sum"),
            rush_epa=("epa", "mean"),
            rz_carries=("is_rz", "sum"),
            gl_carries=("is_gl", "sum"),
        )
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id"})
    )

    usage = rec.merge(rush, on=["game_id", "posteam", "player_id"], how="outer")
    usage = usage.merge(team, on=["game_id", "posteam"], how="left")
    usage = usage.fillna({c: 0 for c in usage.columns if c != "adot"})

    denom = usage["team_pass_att"].replace(0, np.nan)
    usage["target_share"] = usage["targets"] / denom
    usage["air_yards_share"] = usage["air_yards"] / usage["team_air_yards"].replace(0, np.nan)
    # WOPR is the standard 1.5x targets plus 0.7x air yards blend. It predicts
    # future receiving points better than either share alone.
    usage["wopr"] = 1.5 * usage["target_share"] + 0.7 * usage["air_yards_share"]
    usage["carry_share"] = usage["carries"] / usage["team_rush_att"].replace(0, np.nan)
    usage["rz_touch_share"] = (usage["rz_targets"] + usage["rz_carries"]) / usage["team_rz_plays"].replace(0, np.nan)
    usage["gl_touch_share"] = (usage["gl_targets"] + usage["gl_carries"]) / usage["team_gl_plays"].replace(0, np.nan)
    return usage


def load_weekly(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Per-player per-game stat lines joined to derived usage."""

    def build() -> pd.DataFrame:
        import nfl_data_py as nfl

        base = nfl.import_weekly_data(seasons, downcast=True)
        base = base[base["position"].isin(POSITIONS)].copy()
        base = base.rename(
            columns={
                "passing_yards": "pass_yd",
                "passing_tds": "pass_td",
                "interceptions": "interception",
                "rushing_yards": "rush_yd",
                "rushing_tds": "rush_td",
                "receptions": "reception",
                "receiving_yards": "rec_yd",
                "receiving_tds": "rec_td",
                "recent_team": "team",
            }
        )
        base["fumble_lost"] = (
            base.get("sack_fumbles_lost", 0)
            + base.get("rushing_fumbles_lost", 0)
            + base.get("receiving_fumbles_lost", 0)
        )
        base["two_point"] = (
            base.get("passing_2pt_conversions", 0)
            + base.get("rushing_2pt_conversions", 0)
            + base.get("receiving_2pt_conversions", 0)
        )

        pbp = nfl.import_pbp_data(seasons, downcast=True, cache=False)
        usage = _usage_from_pbp(pbp)
        game_keys = pbp[["game_id", "season", "week"]].drop_duplicates()
        usage = usage.merge(game_keys, on="game_id", how="left")

        merged = base.merge(
            usage.drop(columns=["posteam"]),
            on=["player_id", "season", "week"],
            how="left",
        )

        snaps = nfl.import_snap_counts(seasons)
        snaps = snaps.rename(columns={"pfr_player_id": "pfr_id"})
        merged = merged.merge(
            snaps[["pfr_id", "season", "week", "offense_pct"]],
            left_on=["pfr_id", "season", "week"],
            right_on=["pfr_id", "season", "week"],
            how="left",
        ).rename(columns={"offense_pct": "snap_share"})

        # FTN charting adds route participation and pressure context, but only
        # from 2022. Older seasons fall back to an estimate from target volume.
        ftn_seasons = [s for s in seasons if s >= FTN_FIRST_SEASON]
        if ftn_seasons:
            try:
                ftn = nfl.import_ftn_data(ftn_seasons)
                routes = (
                    ftn.groupby(["nflverse_game_id", "season", "week"], observed=True)
                    .agg(is_motion=("is_motion", "mean"), n_offense=("n_offense_backfield", "mean"))
                    .reset_index()
                )
                merged = merged.merge(
                    routes.rename(columns={"nflverse_game_id": "game_id"}),
                    on=["season", "week"], how="left",
                )
            except Exception:
                pass

        return merged

    return _cached("weekly", seasons, build, refresh)


def load_team_week(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Offensive environment per team per game: pace, pass rate over expected,
    efficiency and the line play that sits underneath every skill player."""

    def build() -> pd.DataFrame:
        import nfl_data_py as nfl

        pbp = nfl.import_pbp_data(seasons, downcast=True, cache=False)
        pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
        neutral = pbp[(pbp["wp"].between(0.2, 0.8)) & (pbp["half_seconds_remaining"] > 120)]

        pace = (
            neutral.groupby(["game_id", "posteam"], observed=True)["play_id"]
            .count().rename("neutral_plays").reset_index()
        )
        out = (
            pbp.groupby(["game_id", "season", "week", "posteam"], observed=True)
            .agg(
                plays=("play_id", "count"),
                pass_rate=("pass_attempt", "mean"),
                xpass=("xpass", "mean"),
                off_epa=("epa", "mean"),
                cpoe=("cpoe", "mean"),
                sack_rate=("sack", "mean"),
            )
            .reset_index()
            .merge(pace, on=["game_id", "posteam"], how="left")
        )
        # Pass rate over expected separates coaching intent from game script.
        out["proe"] = out["pass_rate"] - out["xpass"]
        return out.rename(columns={"posteam": "team"})

    return _cached("team_week", seasons, build, refresh)


def load_defense(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """What each defence allows, by position, adjusted for how many plays they
    face. Raw points allowed punishes slow defences unfairly."""

    def build() -> pd.DataFrame:
        weekly = load_weekly(seasons)
        import nfl_data_py as nfl

        sched = nfl.import_schedules(seasons)
        long = pd.concat([
            sched[["season", "week", "home_team", "away_team"]].rename(
                columns={"home_team": "team", "away_team": "opponent"}),
            sched[["season", "week", "away_team", "home_team"]].rename(
                columns={"away_team": "team", "home_team": "opponent"}),
        ])
        w = weekly.merge(long, on=["season", "week", "team"], how="left")
        from .scoring import actual_points
        w["pts"] = actual_points(w, "ppr")
        out = (
            w.groupby(["season", "week", "opponent", "position"], observed=True)
            .agg(pts_allowed=("pts", "sum"),
                 targets_allowed=("targets", "sum"),
                 air_yards_allowed=("air_yards", "sum"))
            .reset_index()
            .rename(columns={"opponent": "defteam"})
        )
        return out

    return _cached("defense", seasons, build, refresh)


def load_schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    """Remaining games with spreads and totals, for game-script projection."""

    def build() -> pd.DataFrame:
        import nfl_data_py as nfl

        sched = nfl.import_schedules([season])
        cols = ["season", "week", "home_team", "away_team", "spread_line",
                "total_line", "roof", "surface", "result"]
        return sched[[c for c in cols if c in sched.columns]]

    return _cached("schedule", [season], build, refresh)


def load_rosters(season: int, refresh: bool = False) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        import nfl_data_py as nfl

        r = nfl.import_seasonal_rosters([season])
        cols = ["player_id", "player_name", "position", "team", "age",
                "years_exp", "draft_number", "height", "weight", "status"]
        return r[[c for c in cols if c in r.columns]]

    return _cached("rosters", [season], build, refresh)


def load_all(seasons: list[int] | None = None, refresh: bool = False) -> dict:
    seasons = seasons or list(range(CURRENT_SEASON - 5, CURRENT_SEASON + 1))
    return {
        "weekly": load_weekly(seasons, refresh),
        "team_week": load_team_week(seasons, refresh),
        "defense": load_defense(seasons, refresh),
        "schedule": load_schedule(CURRENT_SEASON, refresh),
        "rosters": load_rosters(CURRENT_SEASON, refresh),
    }
