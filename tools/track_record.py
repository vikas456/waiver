"""Write web/data/track-record.json, the tested record the scorecard shows.

    python -m pipeline.backtest --season 2025 --frames f2025.parquet
    python -m pipeline.backtest --season 2024 --frames f2024.parquet
    python tools/track_record.py 2025=f2025.parquet 2024=f2024.parquet

Scores what the site ships (the model blended with the consensus by the
shares config.py sets), FantasyPros' consensus and the last four games
exactly as the backtest does: pairwise accuracy within each week and
position, predicting each player's next four games, averaged over the groups.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import market  # noqa: E402
from pipeline.backtest import pairwise_accuracy  # noqa: E402
from pipeline.config import (FULL_FORM_FROM_WEEK, MARKET_WEIGHT,  # noqa: E402
                             MARKET_WEIGHT_EARLY, POSITIONS)

OUT = ROOT / "web" / "data" / "track-record.json"
METHODS = {"site": "shipped", "consensus": "market", "last4": "last4"}


def _shipped(f: pd.DataFrame) -> pd.DataFrame:
    """Frames saved before the backtest scored the shipped blend get it here."""
    parts = []
    for week, g in f.groupby("week"):
        g = g.set_index("player_id")
        share = MARKET_WEIGHT_EARLY if week < FULL_FORM_FROM_WEEK else MARKET_WEIGHT
        parts.append(market.blend(g["model"], g["position"], g["market_rank"], share)
                     .rename("shipped").to_frame().assign(week=week))
    blended = pd.concat(parts).rename_axis("player_id").reset_index()
    return f.drop(columns=["shipped"], errors="ignore").merge(blended, on=["player_id", "week"], how="left")


def season_record(f: pd.DataFrame) -> dict:
    f = f.dropna(subset=["actual", "model", "last4", "season_avg"])
    if "shipped" not in f.columns or f["shipped"].isna().all():
        f = _shipped(f)
    rows = []
    for (week, pos), g in f.groupby(["week", "position"]):
        if pos not in POSITIONS or len(g) < 12:
            continue
        actual = g["actual"].to_numpy()
        rows.append({"week": int(week), "position": pos,
                     **{k: pairwise_accuracy(g[c].to_numpy(), actual)[0] for k, c in METHODS.items()}})
    r = pd.DataFrame(rows)
    four = lambda x: round(float(x), 4)
    return {
        "overall": {k: four(r[k].mean()) for k in METHODS},
        "byPosition": {pos: {k: four(g[k].mean()) for k in METHODS} for pos, g in r.groupby("position")},
        "byWeek": [{"week": int(w), **{k: four(g[k].mean()) for k in METHODS}} for w, g in r.groupby("week")],
    }


def main() -> None:
    seasons = {}
    for arg in sys.argv[1:]:
        season, path = arg.split("=", 1)
        seasons[season] = season_record(pd.read_parquet(path))
    record = {
        "horizon": 4,
        "seasons": dict(sorted(seasons.items(), reverse=True)),
        "average": {k: round(sum(s["overall"][k] for s in seasons.values()) / len(seasons), 4)
                    for k in METHODS},
    }
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    print(f"Wrote {OUT}: " + ", ".join(f"{k} {v:.1%}" for k, v in record["average"].items()))


if __name__ == "__main__":
    main()
