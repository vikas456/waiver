"""Synthetic NFL data with the same shape as the nflverse tables.

This exists so the whole pipeline can be built and tested without network
access, and so anyone can run the project end to end before wiring up real
data. The generator deliberately builds in the structures the model is meant
to find: usage that persists, touchdown rates that do not, mid-season role
changes, and defences that differ in how much they allow.

Nothing here should be mistaken for real football. Swap in ingest.load_all()
and the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
         "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA",
         "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
         "TEN", "WAS"]

FIRST = ["Jalen", "Marcus", "Deshaun", "Tyler", "Amari", "Rashid", "Cade",
         "Quentin", "Darnell", "Elijah", "Trey", "Malik", "Isaiah", "Kenny",
         "Brock", "Xavier", "Jaylen", "Davante", "Tank", "Wan'Dale", "Bo",
         "Roman", "Chigoziem", "Anders", "Keon", "Dorian", "Zamir", "Tyjae"]
LAST = ["Okafor", "Whitfield", "Reyes", "Kincaid", "Brooks", "Vance", "Otton",
        "Halliday", "Mensah", "Sorensen", "Ibarra", "Castellano", "Nwosu",
        "Pritchard", "Dunlap", "Achane", "Robinson", "Bigsby", "Tolliver",
        "Ferreira", "Aiyuk", "Njoku", "Lindstrom", "Vaughan", "Moreau"]

ROSTER_SHAPE = {"QB": 2, "RB": 4, "WR": 6, "TE": 3}


def _names(rng: np.random.Generator, n: int) -> list[str]:
    seen: set[str] = set()
    out = []
    while len(out) < n:
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def build_players(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    total = sum(ROSTER_SHAPE.values()) * len(TEAMS)
    names = _names(rng, total)
    i = 0
    for team in TEAMS:
        for pos, count in ROSTER_SHAPE.items():
            for depth in range(count):
                # Talent drives efficiency; role drives volume. They are
                # correlated but not identical, which is the whole point.
                talent = rng.normal(0, 1) - depth * 0.35
                rows.append({
                    "player_id": f"P{i:04d}",
                    "player_name": names[i],
                    "position": pos,
                    "team": team,
                    "depth": depth,
                    "talent": talent,
                    "age": float(np.clip(rng.normal(26, 3), 21, 36)),
                    "years_exp": int(np.clip(rng.poisson(3.5), 0, 15)),
                    "draft_number": int(np.clip(rng.exponential(90) + depth * 30, 1, 300)),
                })
                i += 1
    return pd.DataFrame(rows)


def build_schedule(rng: np.random.Generator, season: int, weeks: int = 18) -> pd.DataFrame:
    rows = []
    for week in range(1, weeks + 1):
        order = list(rng.permutation(TEAMS))
        # Four teams on bye each week in the middle of the season.
        bye = order[:4] if 5 <= week <= 14 else []
        playing = [t for t in order if t not in bye]
        for a, b in zip(playing[::2], playing[1::2]):
            spread = float(np.round(rng.normal(0, 5.5) * 2) / 2)
            total = float(np.round(rng.normal(44, 4) * 2) / 2)
            rows.append({"season": season, "week": week, "home_team": a,
                         "away_team": b, "spread_line": spread,
                         "total_line": total,
                         "roof": rng.choice(["outdoors", "dome"], p=[0.72, 0.28])})
    return pd.DataFrame(rows)


def build_weekly(rng: np.random.Generator, players: pd.DataFrame,
                 schedule: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate weekly stat lines from a latent role that evolves over time."""
    # Latent team quality and pace, stable across a season.
    team_pace = {t: rng.normal(63, 4) for t in TEAMS}
    team_proe = {t: rng.normal(0, 0.05) for t in TEAMS}
    team_quality = {t: rng.normal(0, 0.12) for t in TEAMS}
    def_strength = {t: rng.normal(1.0, 0.13) for t in TEAMS}

    # Base role by depth chart position.
    base_target = {("WR", 0): 0.24, ("WR", 1): 0.18, ("WR", 2): 0.12,
                   ("WR", 3): 0.07, ("WR", 4): 0.04, ("WR", 5): 0.02,
                   ("TE", 0): 0.17, ("TE", 1): 0.07, ("TE", 2): 0.03,
                   ("RB", 0): 0.11, ("RB", 1): 0.07, ("RB", 2): 0.03,
                   ("RB", 3): 0.015, ("QB", 0): 0.0, ("QB", 1): 0.0}
    base_carry = {("RB", 0): 0.55, ("RB", 1): 0.27, ("RB", 2): 0.11,
                  ("RB", 3): 0.04, ("QB", 0): 0.09, ("QB", 1): 0.05}

    state = {}
    for _, p in players.iterrows():
        key = (p["position"], min(p["depth"], 5))
        state[p["player_id"]] = {
            "target": base_target.get(key, 0.01) * float(np.exp(rng.normal(0, 0.22))),
            "carry": base_carry.get(key, 0.0) * float(np.exp(rng.normal(0, 0.22))),
            "snap": float(np.clip(0.92 - p["depth"] * 0.17 + rng.normal(0, 0.06), 0.05, 0.99)),
            # Some players get a role change part way through the season. This
            # is what change-point detection is supposed to catch.
            "change_week": int(rng.integers(4, 14)) if rng.random() < 0.14 else None,
            "change_size": float(rng.choice([-1, 1]) * rng.uniform(0.25, 0.7)),
        }

    long = pd.concat([
        schedule.rename(columns={"home_team": "team", "away_team": "opponent"})[
            ["season", "week", "team", "opponent", "spread_line", "total_line", "roof"]],
        schedule.rename(columns={"away_team": "team", "home_team": "opponent"})[
            ["season", "week", "team", "opponent", "spread_line", "total_line", "roof"]],
    ], ignore_index=True)

    rows, team_rows = [], []
    for (week, team, opponent), grp in long.groupby(["week", "team", "opponent"]):
        roster = players[players["team"] == team]
        plays = max(40, int(rng.normal(team_pace[team], 4)))
        pass_rate = float(np.clip(0.58 + team_proe[team] + rng.normal(0, 0.05), 0.35, 0.78))
        pass_att = int(plays * pass_rate)
        rush_att = plays - pass_att
        rz_plays = max(1, int(rng.poisson(4 + team_quality[team] * 8)))
        gl_plays = max(0, int(rng.poisson(1.6 + team_quality[team] * 3)))
        opp_def = def_strength[opponent]

        team_rows.append({
            "season": season, "week": week, "team": team, "plays": plays,
            "pass_rate": pass_rate, "xpass": 0.58,
            "off_epa": team_quality[team] * 0.5 + rng.normal(0, 0.05),
            "cpoe": rng.normal(0, 2.2), "sack_rate": float(np.clip(rng.normal(0.068, 0.02), 0.01, 0.2)),
            "neutral_plays": int(plays * 0.55), "proe": pass_rate - 0.58,
        })

        # Shares are drawn as a composition so they sum to one, which is how
        # a real offence allocates its touches.
        for kind, attempts in (("target", pass_att), ("carry", rush_att)):
            weights = []
            for _, p in roster.iterrows():
                s = state[p["player_id"]]
                w = s[kind]
                if s["change_week"] is not None and week >= s["change_week"]:
                    w = max(0.002, w * (1 + s["change_size"]))
                weights.append(w)
            weights = np.array(weights, dtype=float)
            if weights.sum() <= 0:
                weights = np.ones(len(roster))
            alpha = weights / weights.sum() * 24
            draw = rng.dirichlet(np.maximum(alpha, 0.02))
            for idx, (_, p) in enumerate(roster.iterrows()):
                state[p["player_id"]][f"{kind}_share_week"] = draw[idx]

        for _, p in roster.iterrows():
            s = state[p["player_id"]]
            pos, pid = p["position"], p["player_id"]
            snap = float(np.clip(s["snap"] + rng.normal(0, 0.05), 0.02, 1.0))
            if s["change_week"] is not None and week >= s["change_week"]:
                snap = float(np.clip(snap * (1 + s["change_size"] * 0.6), 0.02, 1.0))

            t_share = s.get("target_share_week", 0.0) if pos != "QB" else 0.0
            c_share = s.get("carry_share_week", 0.0) if pos in ("RB", "QB") else 0.0
            targets = rng.poisson(max(0.0, pass_att * t_share))
            carries = rng.poisson(max(0.0, rush_att * c_share))

            adot = {"WR": 11.0, "TE": 7.5, "RB": 1.5, "QB": 0.0}[pos] + rng.normal(0, 2.2)
            catch_rate = float(np.clip(0.78 - 0.018 * adot + p["talent"] * 0.02, 0.2, 0.95))
            receptions = rng.binomial(targets, catch_rate) if targets else 0
            ypr = max(1.0, adot * 0.55 + 4.6 + p["talent"] * 0.8 + rng.normal(0, 2.0))
            rec_yd = float(receptions * ypr) if receptions else 0.0
            ypc = max(0.5, 4.3 + p["talent"] * 0.25 + rng.normal(0, 1.1)) / max(opp_def, 0.5)
            rush_yd = float(carries * ypc) if carries else 0.0

            gl_c = rng.binomial(gl_plays, min(0.9, c_share)) if pos == "RB" else 0
            gl_t = rng.binomial(max(gl_plays - gl_c, 0), min(0.9, t_share)) if pos != "QB" else 0
            rz_c = rng.binomial(rz_plays, min(0.9, c_share)) if pos in ("RB", "QB") else 0
            rz_t = rng.binomial(max(rz_plays - rz_c, 0), min(0.9, t_share)) if pos != "QB" else 0

            # Touchdowns depend on opportunity, with heavy noise on top. The
            # noise is the point: it is what the model must learn to discount.
            td_lambda = 0.19 * gl_c + 0.34 * gl_t + 0.055 * rz_c + 0.10 * rz_t \
                + 0.0025 * rec_yd + 0.0020 * rush_yd
            total_td = rng.poisson(max(td_lambda, 0.005))
            rec_td = rng.binomial(total_td, 0.55) if pos != "RB" else rng.binomial(total_td, 0.25)
            rush_td = total_td - rec_td if pos in ("RB", "QB") else 0

            pass_yd = pass_td = interception = 0.0
            if pos == "QB" and p["depth"] == 0:
                pass_yd = float(max(0, rng.normal(7.0 + p["talent"] * 0.6, 1.6) * pass_att))
                pass_td = rng.poisson(max(0.4, 1.55 + team_quality[team] * 2))
                interception = rng.poisson(0.75)
                carries = rng.poisson(4.0)
                rush_yd = float(carries * max(0.0, rng.normal(4.5, 2.0)))

            if p["depth"] > 0 and pos == "QB":
                snap, targets, carries = 0.02, 0, 0

            rows.append({
                "player_id": pid, "player_name": p["player_name"], "position": pos,
                "team": team, "opponent": opponent, "season": season, "week": week,
                "age": p["age"], "years_exp": p["years_exp"], "draft_number": p["draft_number"],
                "snap_share": snap, "targets": int(targets), "carries": int(carries),
                "routes": float(pass_att * min(1.0, snap * 0.92)),
                "team_pass_att": pass_att, "team_rush_att": rush_att,
                "adot": float(adot), "air_yards": float(targets * adot),
                "yac": float(receptions * max(0.0, rng.normal(4.6, 1.8))),
                "gl_carries": int(gl_c), "gl_targets": int(gl_t),
                "rz_carries": int(rz_c), "rz_targets": int(rz_t),
                "rec_epa": float(rng.normal(p["talent"] * 0.1, 0.35)),
                "rush_epa": float(rng.normal(p["talent"] * 0.06, 0.3)),
                "target_share": t_share, "carry_share": c_share,
                "air_yards_share": t_share * float(np.clip(rng.normal(1.0, 0.2), 0.2, 2.5)),
                "rz_touch_share": (rz_c + rz_t) / max(rz_plays, 1),
                "gl_touch_share": (gl_c + gl_t) / max(gl_plays, 1),
                "pass_yd": pass_yd, "pass_td": float(pass_td),
                "interception": float(interception),
                "rush_yd": rush_yd, "rush_td": float(rush_td),
                "reception": float(receptions), "rec_yd": rec_yd, "rec_td": float(rec_td),
                "fumble_lost": float(rng.random() < 0.028),
                "two_point": float(rng.random() < 0.012),
            })

    weekly = pd.DataFrame(rows)
    weekly["wopr"] = 1.5 * weekly["target_share"] + 0.7 * weekly["air_yards_share"]
    return weekly, pd.DataFrame(team_rows)


def build_defense(weekly: pd.DataFrame) -> pd.DataFrame:
    from .scoring import actual_points

    w = weekly.copy()
    w["pts"] = actual_points(w, "ppr")
    return (w.groupby(["season", "week", "opponent", "position"], observed=True)
            .agg(pts_allowed=("pts", "sum"), targets_allowed=("targets", "sum"),
                 air_yards_allowed=("air_yards", "sum"))
            .reset_index().rename(columns={"opponent": "defteam"}))


def generate(seasons: list[int], seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    players = build_players(rng)
    weeklies, team_weeks, schedules = [], [], []
    for season in seasons:
        sched = build_schedule(rng, season)
        wk, tw = build_weekly(rng, players, sched, season)
        weeklies.append(wk)
        team_weeks.append(tw)
        schedules.append(sched)
    weekly = pd.concat(weeklies, ignore_index=True)
    return {
        "weekly": weekly,
        "team_week": pd.concat(team_weeks, ignore_index=True),
        "defense": build_defense(weekly),
        "schedule": pd.concat(schedules, ignore_index=True),
        "rosters": players[["player_id", "player_name", "position", "team",
                            "age", "years_exp", "draft_number"]],
    }
