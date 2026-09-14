"""Pull raw data from nflverse and reshape it into the tables the feature
layer expects.

Five tables come out of here:
  weekly    one row per player per regular-season game, with usage and stat components
  team_week one row per team per game, with pace, pass rate and efficiency
  defense   one row per defence per game, with what they allowed by position
  schedule  every regular-season game, with Vegas lines for game-script projection
  rosters   one row per player in the latest season, for age and availability

Loading goes through nflreadpy, the maintained successor to nfl_data_py. The
older library reads a player-stats release that nflverse retired in 2025, so
it cannot load any season from 2025 onward.

Completed seasons are cached to parquet under data/cache, one file per season,
so a rebuild during the week does not re-download five seasons of
play-by-play. The season in progress is never cached to disk; see _season.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CURRENT_SEASON, POSITIONS
from .scoring import actual_points

CACHE_DIR = Path(os.environ.get("FF_CACHE", "data/cache"))

PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam",
    "play_type", "yardline_100", "pass_attempt", "rush_attempt", "sack",
    "air_yards", "yards_after_catch", "epa", "cpoe", "xpass", "wp",
    "half_seconds_remaining", "receiver_player_id", "rusher_player_id",
]

# nflverse stat names mapped onto the components the model predicts.
STAT_RENAMES = {
    "player_display_name": "player_name",
    "passing_yards": "pass_yd",
    "passing_tds": "pass_td",
    "passing_interceptions": "interception",
    "rushing_yards": "rush_yd",
    "rushing_tds": "rush_td",
    "receptions": "reception",
    "receiving_yards": "rec_yd",
    "receiving_tds": "rec_td",
    # Kept only so the doctor can check scoring.py against nflverse's total.
    "fantasy_points_ppr": "nflverse_ppr",
}
FUMBLES_LOST = ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]
TWO_POINT = ["passing_2pt_conversions", "rushing_2pt_conversions",
             "receiving_2pt_conversions"]
WEEKLY_COLUMNS = (["player_id", "position", "season", "week", "game_id", "team",
                   "opponent_team", "targets", "carries"]
                  + list(STAT_RENAMES) + FUMBLES_LOST + TWO_POINT)

# Per-player route counts are not published free, so routes are estimated
# from snap share on the team's dropbacks, using the same rate as synth.py.
# Route participation is therefore a proxy for snap share, not a separate
# signal. Models are fitted per position, so one rate for every position
# costs nothing.
ROUTES_PER_PASS_SNAP = 0.92

_LIVE: dict[tuple[str, int], pd.DataFrame] = {}


def _nfl():
    import nflreadpy
    return nflreadpy


@functools.cache
def _pfr_ids() -> pd.Series:
    """gsis id to Pro Football Reference id, from nflverse's master player
    table, which is unique on both. Held for the run rather than cached to
    disk, because it gains every new rookie."""
    players = _nfl().load_players().to_pandas()
    return players.dropna(subset=["gsis_id", "pfr_id"]).set_index("gsis_id")["pfr_id"]


def _season(name: str, season: int, builder, refresh: bool) -> pd.DataFrame:
    """One season of one table, read from cache where that is safe.

    The season in progress is rebuilt on every run and held only in memory.
    Its data changes weekly and CI restores the cache between runs, so a copy
    on disk would pin every later projection to the first download of the
    year. Memory still spares a second download within the same run.
    """
    if season >= CURRENT_SEASON:
        key = (name, season)
        if refresh or key not in _LIVE:
            _LIVE[key] = builder(season)
        return _LIVE[key]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    frame = builder(season)
    frame.to_parquet(path, index=False)
    return frame


def _by_season(name: str, seasons: list[int], builder, refresh: bool) -> pd.DataFrame:
    return pd.concat([_season(name, s, builder, refresh) for s in seasons],
                     ignore_index=True)


# ---------------------------------------------------------------------------
# Play-by-play derived usage
# ---------------------------------------------------------------------------

def load_pbp(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Regular-season play-by-play, trimmed to the columns used here.

    Usage and team environment are both built from it, so it is cached on its
    own and the largest download in the pipeline happens once.
    """

    def build(season: int) -> pd.DataFrame:
        pbp = _nfl().load_pbp([season]).select(PBP_COLUMNS).to_pandas()
        return pbp[pbp["season_type"] == "REG"].reset_index(drop=True)

    return _by_season("pbp", seasons, build, refresh)


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
    """Per-player per-game stat lines joined to derived usage and snap share."""

    def build(season: int) -> pd.DataFrame:
        nfl = _nfl()
        stats = nfl.load_player_stats([season], summary_level="week").to_pandas()
        base = stats.loc[(stats["season_type"] == "REG")
                         & stats["position"].isin(POSITIONS), WEEKLY_COLUMNS]
        base = base.rename(columns=STAT_RENAMES)
        base["fumble_lost"] = base[FUMBLES_LOST].fillna(0).sum(axis=1)
        base["two_point"] = base[TWO_POINT].fillna(0).sum(axis=1)
        base = base.drop(columns=FUMBLES_LOST + TWO_POINT)

        # The stat file already carries official target and carry counts, and
        # its own share columns were dropped above, so every share comes from
        # play-by-play against one set of team denominators. Overlapping
        # names would otherwise come back as targets_x and targets_y, which
        # the feature layer silently reads as missing.
        pbp = load_pbp([season], refresh)
        games = pbp[["game_id", "season", "week"]].drop_duplicates()
        usage = (_usage_from_pbp(pbp)
                 .merge(games, on="game_id", how="left", validate="m:1")
                 .drop(columns=["game_id", "posteam", "targets", "carries"]))
        merged = base.merge(usage, on=["player_id", "season", "week"],
                            how="left", validate="1:1")

        # Snap counts are keyed by Pro Football Reference id, which the stat
        # file lacks, so the master player table supplies it. Seasonal
        # rosters carry one too, but it is missing for a fifth of 2022 and
        # sometimes wrong (in 2023 Tyler Conklin carries Ryan Izzo's), so the
        # roster only fills gaps, alongside the draft capital and experience
        # the role priors need. Team is part of the snap join because the
        # snap file itself reuses an id: in 2021 two players share DaviJa06.
        roster = load_rosters([season], refresh)[
            ["player_id", "pfr_id", "years_exp", "draft_number"]]
        merged = merged.merge(roster, on="player_id", how="left", validate="m:1")
        merged["pfr_id"] = merged["player_id"].map(_pfr_ids()).fillna(merged["pfr_id"])

        snaps = nfl.load_snap_counts([season]).to_pandas()
        snaps = (snaps.loc[(snaps["game_type"] == "REG") & snaps["pfr_player_id"].notna(),
                           ["pfr_player_id", "season", "week", "team", "offense_pct"]]
                 .rename(columns={"pfr_player_id": "pfr_id", "offense_pct": "snap_share"}))
        merged = merged.merge(snaps, on=["pfr_id", "season", "week", "team"],
                              how="left", validate="m:1")

        merged["routes"] = merged["team_pass_att"] * merged["snap_share"] * ROUTES_PER_PASS_SNAP
        return merged

    return _by_season("weekly", seasons, build, refresh)


def load_team_week(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Offensive environment per team per game: pace, pass rate over expected,
    efficiency and the line play that sits underneath every skill player."""

    def build(season: int) -> pd.DataFrame:
        pbp = load_pbp([season], refresh)
        pbp = pbp[pbp["play_type"].isin(["pass", "run"])]
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

    return _by_season("team_week", seasons, build, refresh)


def load_defense(weekly: pd.DataFrame) -> pd.DataFrame:
    """What each defence allows, by position. The adjustment for how many
    plays they face happens in features.opponent_strength.

    Derived from the weekly table rather than downloaded, so it needs no cache.
    """
    w = weekly.assign(pts=actual_points(weekly, "ppr"))
    return (
        w.groupby(["season", "week", "opponent_team", "position"], observed=True)
        .agg(pts_allowed=("pts", "sum"),
             targets_allowed=("targets", "sum"),
             air_yards_allowed=("air_yards", "sum"))
        .reset_index()
        .rename(columns={"opponent_team": "defteam"})
    )


def load_schedule(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Every regular-season game with spreads and totals, for game-script
    projection. Past seasons are needed too: without them every training row
    loses its opponent and its Vegas context."""

    def build(season: int) -> pd.DataFrame:
        sched = _nfl().load_schedules([season]).to_pandas()
        cols = ["season", "week", "home_team", "away_team", "spread_line",
                "total_line", "roof", "surface", "result"]
        return sched.loc[sched["game_type"] == "REG", cols].reset_index(drop=True)

    return _by_season("schedule", seasons, build, refresh)


def load_rosters(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    def build(season: int) -> pd.DataFrame:
        r = _nfl().load_rosters([season]).to_pandas()
        r = r.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
        r = r.dropna(subset=["player_id"]).drop_duplicates("player_id", keep="last")
        born = pd.to_datetime(r["birth_date"], errors="coerce")
        r["age"] = (pd.Timestamp(season, 9, 1) - born).dt.days / 365.25
        cols = ["player_id", "season", "player_name", "position", "team", "pfr_id",
                "age", "years_exp", "draft_number", "status"]
        return r[cols].reset_index(drop=True)

    return _by_season("rosters", seasons, build, refresh)


def load_all(seasons: list[int] | None = None, refresh: bool = False) -> dict:
    # project.py appends the target season to the training seasons, so a
    # historical run would otherwise load that season twice.
    seasons = sorted(set(seasons or range(CURRENT_SEASON - 5, CURRENT_SEASON + 1)))
    weekly = load_weekly(seasons, refresh)
    rosters = load_rosters(seasons, refresh)
    return {
        "weekly": weekly,
        "team_week": load_team_week(seasons, refresh),
        "defense": load_defense(weekly),
        "schedule": load_schedule(seasons, refresh),
        # project.py indexes this by player id, so it holds one season only.
        "rosters": rosters[rosters["season"] == max(seasons)],
    }
