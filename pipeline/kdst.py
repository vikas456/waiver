"""Kickers and team defences.

These two are not projected the way the skill positions are, because there is
almost no usage to model and almost nothing repeats. Measured over 2022-2025,
ordering kickers by what they go on to score over the next four games is right
about 51% of the time going by their last four games, 54% going by the season,
and 58% going by expert consensus. Defences run 53%, 54% and 57%. The skill
positions reach 86%. A kicker's first half of a season predicts his second at
r = 0.14, a defence's at r = 0.16.

What does carry information is the game: a team expected to score more kicks
more extra points, and a defence facing a weaker offence collects more sacks
and allows fewer points. So both are projected from the betting line, nudged a
little by the unit's own season, and blended with consensus. The site says so
plainly rather than dressing up a coin flip.

Points allowed are scored in bands, so a defence carries the chance of landing
in each one rather than an average nobody scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import KDST_COMPONENTS, LINES_KNOWN_AHEAD, PA_TIERS

# Seasons of history behind the fits. Kicking rules and defensive scoring have
# been stable across these.
MIN_ROWS = 200

# How much of the unit's own season to mix into the line's verdict. A defence's
# season-to-date scoring predicts the rest of its season at r = 0.16 and a
# kicker's at r = 0.14, so this is deliberately small.
FORM_WEIGHT = {"K": 0.15, "DEF": 0.15}

# How much of the consensus rank to blend in, swept over 2022-2025: kickers
# peak around 0.5 and defences around 0.4, and neither is the worst choice in
# any single season at those weights.
MARKET_WEIGHT = {"K": 0.5, "DEF": 0.4}

# nflverse, FantasyPros and Sleeper spell two teams differently.
TEAM_ALIASES = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "ARZ": "ARI",
                "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA", "SD": "LAC", "OAK": "LV"}
# The other direction, for matching a Sleeper roster.
SLEEPER_TEAM = {"LA": "LAR"}

# Bands the points a defence allows fall into, matching PA_TIERS.
PA_EDGES = [t[0] for t in PA_TIERS]


def team(name) -> str:
    """One spelling per team, the one nflverse uses."""
    t = str(name).upper()
    return TEAM_ALIASES.get(t, t)


# ---------------------------------------------------------------------------
# The weekly record, rebuilt from the stat tables
# ---------------------------------------------------------------------------

def kicking(seasons: list[int]) -> pd.DataFrame:
    """One row per kicker per game: what he actually did."""
    import nflreadpy as nfl

    ps = nfl.load_player_stats(seasons).to_pandas()
    k = ps[ps["position"] == "K"].copy()
    k["team"] = k["team"].map(team)
    k["fg_0_39"] = k["fg_made_0_19"] + k["fg_made_20_29"] + k["fg_made_30_39"]
    k["fg_40_49"] = k["fg_made_40_49"]
    k["fg_50"] = k["fg_made_50_59"] + k["fg_made_60_"]
    k["fg_miss"] = k["fg_missed"]
    k["pat"] = k["pat_made"]
    return k.rename(columns={"player_id": "id", "player_display_name": "name"})[
        ["season", "week", "team", "id", "name", "fg_0_39", "fg_40_49", "fg_50",
         "fg_miss", "pat", "fg_att", "fg_made"]]


def defences(seasons: list[int]) -> pd.DataFrame:
    """One row per team defence per game, points allowed included.

    The scores come straight from nflverse rather than the pipeline's own
    schedule, which keeps the betting lines and the result but not the two
    teams' scores, and points allowed is the whole of a defence's biggest
    scoring category.
    """
    import nflreadpy as nfl

    ts = nfl.load_team_stats(seasons).to_pandas()
    ts["team"] = ts["team"].map(team)
    games = nfl.load_schedules(seasons).to_pandas()
    games = games[games["game_type"] == "REG"]
    allowed = pd.concat([
        games.rename(columns={"home_team": "team", "away_score": "allowed"})[
            ["season", "week", "team", "allowed"]],
        games.rename(columns={"away_team": "team", "home_score": "allowed"})[
            ["season", "week", "team", "allowed"]],
    ], ignore_index=True).dropna(subset=["allowed"])
    allowed["team"] = allowed["team"].map(team)

    d = ts.merge(allowed, on=["season", "week", "team"], how="inner")
    out = pd.DataFrame({
        "season": d["season"], "week": d["week"], "team": d["team"],
        "sack": d["def_sacks"].fillna(0),
        "interception_def": d["def_interceptions"].fillna(0),
        "fumble_recovery": d["fumble_recovery_opp"].fillna(0),
        # A return touchdown counts for the defence in every common ruleset.
        "def_td": d["def_tds"].fillna(0) + d["special_teams_tds"].fillna(0),
        "safety": d["def_safeties"].fillna(0),
        "block": (d["def_punt_blocks"].fillna(0) + d["def_pat_blocks"].fillna(0)
                  + d["def_fg_blocks"].fillna(0)),
        "allowed": d["allowed"],
    })
    out["id"] = out["team"]
    out["name"] = out["team"] + " D/ST"
    return out


def game_lines(schedule: pd.DataFrame) -> pd.DataFrame:
    """Implied total for each team and for the side it faces."""
    home = schedule.rename(columns={"home_team": "team", "away_team": "opponent"}).copy()
    home["implied"] = home["total_line"] / 2 + home["spread_line"] / 2
    away = schedule.rename(columns={"away_team": "team", "home_team": "opponent"}).copy()
    away["implied"] = away["total_line"] / 2 - away["spread_line"] / 2
    cols = ["season", "week", "team", "opponent", "implied"]
    both = pd.concat([home[cols], away[cols]], ignore_index=True)
    both["team"] = both["team"].map(team)
    both["opponent"] = both["opponent"].map(team)
    facing = both.rename(columns={"team": "opponent", "opponent": "team",
                                  "implied": "opp_implied"})[["season", "week", "team", "opp_implied"]]
    return both.merge(facing, on=["season", "week", "team"], how="left")


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

def _line_fit(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Slope and intercept of y on x, or a flat mean when there is too little."""
    ok = x.notna() & y.notna()
    if ok.sum() < MIN_ROWS:
        return 0.0, float(y[ok].mean() if ok.any() else 0.0)
    slope, intercept = np.polyfit(x[ok].to_numpy(), y[ok].to_numpy(), 1)
    return float(slope), float(intercept)


def fit(kicks: pd.DataFrame, defs: pd.DataFrame, lines: pd.DataFrame) -> dict:
    """Everything the projection needs, measured rather than assumed."""
    k = kicks.merge(lines, on=["season", "week", "team"], how="left")
    d = defs.merge(lines, on=["season", "week", "team"], how="left")

    params: dict = {"K": {}, "DEF": {}, "pa": {}, "sd": {}}
    # A kicker's extra points follow his offence closely; his field goals
    # barely move with it, because a field goal is a drive that stalled.
    for col in ("fg_0_39", "fg_40_49", "fg_50", "fg_miss", "pat"):
        params["K"][col] = _line_fit(k["implied"], k[col])
    for col in ("sack", "interception_def", "fumble_recovery", "def_td", "safety", "block"):
        params["DEF"][col] = _line_fit(d["opp_implied"], d[col])

    # Points allowed, as the chance of landing in each scoring band, by what
    # the opponent was expected to score. Averages are no use here: no defence
    # allows 22.5 points, it allows 17 or 31.
    bands = pd.cut(d["opp_implied"], [0, 18, 21, 24, 27, 99])
    # Seven bands need eight edges: everything up to nothing at all, then each
    # tier's upper bound, then anything above the last one.
    tiers = pd.cut(d["allowed"], [-1] + PA_EDGES[:-1] + [999],
                   labels=[str(t[0]) for t in PA_TIERS])
    table = pd.crosstab(bands, tiers, normalize="index")
    params["pa"] = {str(band): [float(table.loc[band].get(str(t[0]), 0.0)) for t in PA_TIERS]
                    for band in table.index}
    params["pa_bands"] = [18, 21, 24, 27]

    # How far a week's score strays, which the site turns into a floor and a
    # ceiling. Measured on what each unit actually scored, in the default
    # values, rather than inferred from any one component.
    params["sd"] = {"K": float(weekly_points(k, "K").std()),
                    "DEF": float(weekly_points(d, "DEF").std())}

    # A kicker whose game never gave him a chance, a defence that was run over:
    # 7% of kicker games and 18% of defence games are worth a point or less,
    # which is far more often than one smooth curve allows. Fitted on the
    # projection's own size, as train.fit_dud does for the skill positions.
    from .config import SCORING_FORMATS

    weights = SCORING_FORMATS["ppr"]
    params["dud"] = {}
    for position, frame, line_col in (("K", k, "implied"), ("DEF", d, "opp_implied")):
        actual = weekly_points(frame, position).to_numpy()
        projected = np.array([
            score(components(params, position, float(x), None),
                  pa_odds(params, float(x)) if position == "DEF" else None, weights)
            if not np.isnan(x) else np.nan for x in frame[line_col].to_numpy()])
        ok = ~np.isnan(projected) & ~np.isnan(actual)
        if ok.sum() < MIN_ROWS:
            continue
        x = np.column_stack([np.ones(ok.sum()), np.log(np.maximum(projected[ok], 0.2))])
        y = (actual[ok] <= 1.0).astype(float)
        b = np.zeros(2)
        for _ in range(60):
            p = 1 / (1 + np.exp(-x @ b))
            w = np.maximum(p * (1 - p), 1e-6)
            b += np.linalg.solve(x.T @ (x * w[:, None]) + 1e-6 * np.eye(2), x.T @ (y - p))
        params["dud"][position] = {"intercept": float(b[0]), "slope": float(b[1])}
    return params


def weekly_points(frame: pd.DataFrame, position: str) -> pd.Series:
    """What a unit actually scored each week, in the default values. Used to
    measure spread, never to project."""
    from .config import SCORING_FORMATS

    w = SCORING_FORMATS["ppr"]
    pts = sum(frame[c] * w[c] for c in KDST_COMPONENTS if c in frame.columns)
    if position == "DEF" and "allowed" in frame.columns:
        edges = [t[0] for t in PA_TIERS]
        values = [t[1] for t in PA_TIERS]
        allowed = frame["allowed"].to_numpy()
        pts = pts + np.select([allowed <= e for e in edges], values, default=values[-1])
    return pd.Series(pts, index=frame.index)


def pa_odds(params: dict, opp_implied: float) -> list[float]:
    """Chance of each points-allowed band for one game."""
    bands = params["pa_bands"]
    edges = [0] + bands + [99]
    for lo, hi in zip(edges, edges[1:]):
        if lo < opp_implied <= hi:
            key = f"({lo}, {hi}]"
            if key in params["pa"]:
                return params["pa"][key]
    # A line outside anything seen before takes the nearest band.
    return list(params["pa"].values())[-1 if opp_implied > bands[-1] else 0]


def components(params: dict, position: str, implied: float, form: dict | None) -> dict:
    """The stat line a kicker or defence is projected for in one game."""
    out = {}
    for col, (slope, intercept) in params[position].items():
        value = slope * implied + intercept
        # The unit's own season is worth a little, and no more than a little.
        if form and col in form and not np.isnan(form[col]):
            value = (1 - FORM_WEIGHT[position]) * value + FORM_WEIGHT[position] * form[col]
        out[col] = max(0.0, round(float(value), 3))
    return {k: v for k, v in out.items() if k in KDST_COMPONENTS}


def with_typical_lines(lines: pd.DataFrame, season: int, from_week: int) -> pd.DataFrame:
    """Real lines where the book has posted them, each team's typical game
    everywhere else.

    Only the coming week or two are priced when a forecast is made, and most of
    the season is not. Dropping those weeks would leave a kicker with two games
    to his name; using the final lines would hand the site information it will
    not have on the day.
    """
    out = lines.copy()
    priced = out.dropna(subset=["implied"])
    this_season = priced[(priced["season"] == season) & (priced["week"] < from_week)].groupby("team")
    typical = this_season[["implied", "opp_implied"]].mean()[this_season.size() >= 3]
    typical = typical.combine_first(
        priced[priced["season"] == season - 1].groupby("team")[["implied", "opp_implied"]].mean())
    unknown = (out["week"] > from_week + LINES_KNOWN_AHEAD) | out["implied"].isna()
    for col in ("implied", "opp_implied"):
        fill = out.loc[unknown, "team"].map(typical[col]) if col in typical else None
        if fill is not None:
            out.loc[unknown, col] = fill
    # An opponent's typical game is what it usually concedes, not what it
    # usually scores, so it is taken from the other side of the fixture.
    missing = out["opp_implied"].isna() & out["opponent"].notna()
    if "implied" in typical:
        out.loc[missing, "opp_implied"] = out.loc[missing, "opponent"].map(typical["implied"])
    return out


def _recent(frame: pd.DataFrame, season: int, from_week: int, key: str, cols: list[str]) -> dict:
    """Each unit's per-game rate so far this season, or last season's when the
    season is too young to say anything."""
    now = frame[(frame["season"] == season) & (frame["week"] < from_week)]
    before = frame[frame["season"] == season - 1]
    source = now if now[key].nunique() and len(now) >= 32 else before
    if source.empty:
        return {}
    return source.groupby(key)[cols].mean().to_dict("index")


def _blend_to_market(points: dict, ranks: dict, weight: float) -> dict:
    """Pull each projection toward what its consensus rank implies.

    A rank carries no points, so it borrows the projections' own scale: the
    unit ranked third is worth what the third-best projection is worth. Mirrors
    market.blend for the skill positions.
    """
    if not ranks or weight <= 0:
        return {k: 1.0 for k in points}
    scale = sorted(points.values(), reverse=True)
    out = {}
    for unit, value in points.items():
        rank = ranks.get(unit)
        if rank is None or value <= 0:
            out[unit] = 1.0
            continue
        implied = scale[min(int(rank) - 1, len(scale) - 1)]
        out[unit] = ((1 - weight) * value + weight * implied) / value
    return out


def build(season: int, from_week: int, through_week: int, schedule: pd.DataFrame,
          train_seasons: list[int], ranks: dict | None = None,
          avail: dict | None = None) -> dict:
    """Payload rows for every kicker and team defence, week by week, and the
    dud odds the site's floor needs.

    ranks is {"K": {player id: rank}, "DEF": {team: rank}} from the consensus,
    and avail the injury picture from pipeline/availability.py, which only a
    kicker can be caught by: a defence is never out.
    """
    seasons = sorted(set(train_seasons + [season]))
    kicks, defs = kicking(seasons), defences(seasons)
    lines = game_lines(schedule)
    played = lambda f: f[(f["season"] < season) | (f["week"] < from_week)]
    params = fit(played(kicks), played(defs), lines)
    lines = with_typical_lines(lines, season, from_week)
    ranks = ranks or {}

    # Who holds the job now: the last kicker each team used, this season if it
    # has started, otherwise last season's.
    recent_k = kicks[(kicks["season"] == season) & (kicks["week"] < from_week)]
    if recent_k.empty:
        recent_k = kicks[kicks["season"] == season - 1]
    holders = (recent_k.sort_values(["season", "week"]).groupby("team").tail(1)
               .set_index("team")[["id", "name"]].to_dict("index"))

    form_k = _recent(kicks, season, from_week, "id", ["fg_0_39", "fg_40_49", "fg_50", "fg_miss", "pat"])
    form_d = _recent(defs, season, from_week, "team",
                     ["sack", "interception_def", "fumble_recovery", "def_td", "safety", "block"])

    games = lines[(lines["season"] == season) & lines["week"].between(from_week, through_week)]
    units: dict[str, dict] = {}
    for _, game in games.iterrows():
        tm, week = game["team"], int(game["week"])
        for position in ("K", "DEF"):
            if position == "K":
                who = holders.get(tm)
                if not who:
                    continue
                uid, name = who["id"], who["name"]
                implied, form = game["implied"], form_k.get(uid)
            else:
                uid, name, implied, form = tm, f"{tm} D/ST", game["opp_implied"], form_d.get(tm)
            if pd.isna(implied):
                continue
            line = components(params, position, float(implied), form)
            week_row = {
                "w": week, "opp": game["opponent"],
                "p": 1.0, "c": line, "sd": round(params["sd"][position], 2),
            }
            if position == "DEF":
                week_row["pa"] = [round(p, 3) for p in pa_odds(params, float(implied))]
            unit = units.setdefault(uid, {"id": uid, "name": name, "position": position,
                                          "team": tm, "playProb": 1.0, "weeks": []})
            unit["weeks"].append(week_row)

    # The consensus knows about a change of kicker or a defence's new signing,
    # so each projection is pulled part of the way toward its rank.
    from .config import SCORING_FORMATS
    weights = SCORING_FORMATS["ppr"]
    for position in ("K", "DEF"):
        group = {u: v for u, v in units.items() if v["position"] == position}
        points = {u: (sum(score(w["c"], w.get("pa"), weights) for w in v["weeks"]) / len(v["weeks"]))
                  for u, v in group.items() if v["weeks"]}
        factors = _blend_to_market(points, ranks.get(position, {}), MARKET_WEIGHT[position])
        for uid, unit in group.items():
            factor = factors.get(uid, 1.0)
            before = points.get(uid, 0.0)
            for week in unit["weeks"]:
                week["c"] = {k: round(v * factor, 3) for k, v in week["c"].items()}
            after = (sum(score(w["c"], w.get("pa"), weights) for w in unit["weeks"]) / len(unit["weeks"]))
            # The site explains every projection by what moved it. For these
            # two that is the game itself, the unit's own season, and the
            # consensus, all in points against a typical week at the position.
            typical = sum(points.values()) / len(points) if points else 0.0
            drivers = {"matchup": round(before - typical, 2),
                       "market": round(after - before, 2)}
            unit["drivers"] = drivers
            unit["stats"] = {"gamesPlayed": len(unit["weeks"])}
            # The site reads a week's drivers, not a unit's, because a skill
            # player's change from week to week. These do not, so every week
            # carries the same pair.
            for week in unit["weeks"]:
                week["drivers"] = drivers
    # Games are collected fixture by fixture, so they need putting in order.
    for unit in units.values():
        unit["weeks"].sort(key=lambda w: w["w"])
    _attach_sleeper_ids(units.values())
    if avail:
        _attach_play_chance(units.values(), from_week, avail)
    return {"units": sorted(units.values(), key=lambda u: (u["position"], u["name"])),
            "dudOdds": params.get("dud", {})}


def _attach_play_chance(units, from_week: int, avail: dict) -> None:
    """A kicker on injured reserve is not kicking. Nothing else here misses a
    game: a defence plays every week its team does."""
    from . import availability

    for unit in units:
        if unit["position"] != "K":
            continue
        weeks = [w["w"] for w in unit["weeks"]]
        chance = availability.by_week(unit["id"], "K", unit["team"], 1.0, weeks, from_week, avail)
        for week in unit["weeks"]:
            week["p"] = chance.get(week["w"], 1.0)
        now = availability.current(unit["id"], from_week, avail)
        if now:
            unit["injury"] = {"status": now["label"], "detail": now["detail"]}


def _attach_sleeper_ids(units) -> None:
    """So a connected Sleeper league can tell whether a kicker or a defence is
    already taken. Sleeper keys a defence by the team it is."""
    try:
        import nflreadpy as nfl

        ids = nfl.load_ff_playerids().to_pandas()
        kickers = ids[ids["position"] == "PK"][["gsis_id", "sleeper_id"]].dropna()
        to_sleeper = {g: str(s).replace(".0", "") for g, s in
                      zip(kickers["gsis_id"], kickers["sleeper_id"])}
    except Exception as err:
        print(f"  kicker Sleeper ids unavailable ({type(err).__name__})")
        to_sleeper = {}
    for unit in units:
        if unit["position"] == "DEF":
            unit["sleeperId"] = SLEEPER_TEAM.get(unit["team"], unit["team"])
        elif unit["id"] in to_sleeper:
            unit["sleeperId"] = to_sleeper[unit["id"]]


def score(line: dict, pa: list[float] | None, weights: dict) -> float:
    """Fantasy points for one projected week, in one league's scoring."""
    pts = sum(v * weights.get(k, 0.0) for k, v in line.items())
    if pa:
        pts += sum(p * value for p, (_, value) in zip(pa, PA_TIERS))
    return pts
