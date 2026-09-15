"""Consensus rankings from FantasyPros, via nflverse.

Expert consensus is the hardest baseline a projection faces and, blended in
lightly, the most useful anchor it can have. The model's value is in where it
disagrees with the market for a good reason; the market's value is that it
knows things the model cannot see, such as who is starting next week.

Rest-of-season rankings are scraped weekly, so a backtest can ask what the
consensus said on any date, and nothing later.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

PAGES = {"redraft-qb": "QB", "redraft-rb": "RB", "redraft-wr": "WR", "redraft-te": "TE"}


@functools.cache
def _history() -> pd.DataFrame:
    import nflreadpy as nfl

    ids = nfl.load_ff_playerids().to_pandas()[["fantasypros_id", "gsis_id"]].dropna()
    ids["fantasypros_id"] = ids["fantasypros_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    rk = nfl.load_ff_rankings("all").to_pandas()
    rk = rk[rk["page_type"].isin(PAGES)].copy()
    rk["scrape_date"] = pd.to_datetime(rk["scrape_date"])
    rk["fantasypros_id"] = rk["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    rk = rk.merge(ids.drop_duplicates("fantasypros_id"), on="fantasypros_id")
    return rk[["scrape_date", "page_type", "gsis_id", "ecr"]]


def ranks_before(cutoff) -> pd.Series:
    """Positional consensus rank per player, from the last scrape before cutoff."""
    rk = _history()
    rk = rk[rk["scrape_date"] < pd.Timestamp(cutoff)]
    if rk.empty:
        return pd.Series(dtype=float)
    latest = rk.groupby("page_type")["scrape_date"].transform("max")
    rk = rk[rk["scrape_date"] == latest].sort_values("ecr")
    rk["rank"] = rk.groupby("page_type").cumcount() + 1
    return rk.drop_duplicates("gsis_id").set_index("gsis_id")["rank"].astype(float)


def scraped_before(cutoff) -> pd.Timestamp | None:
    """When the consensus that ranks_before(cutoff) returns was taken. It is
    scraped weekly, so news since then is not in it."""
    rk = _history()
    rk = rk[rk["scrape_date"] < pd.Timestamp(cutoff)]
    return None if rk.empty else rk["scrape_date"].max()


def blend(model_ppg: pd.Series, position: pd.Series, rank: pd.Series,
          weight: float) -> pd.Series:
    """Shrink each projection toward what the consensus rank implies.

    A rank carries no points, so it borrows the model's own scale: the
    market's fifth quarterback is worth whatever the model's fifth-best
    quarterback projects. Players the market does not rank keep the model's
    number, rather than being dragged toward a rank nobody gave them.
    """
    out = model_ppg.astype(float).copy()
    if weight <= 0 or rank.dropna().empty:
        return out
    rank = rank.reindex(model_ppg.index)
    for pos in position.unique():
        idx = position.index[position == pos]
        ranked = rank[idx].dropna()
        if ranked.empty:
            continue
        scale = np.sort(model_ppg[idx].to_numpy(dtype=float))[::-1]
        order = ranked.rank(method="first").astype(int).to_numpy() - 1
        implied = pd.Series(scale[order], index=ranked.index)
        out[implied.index] = (1 - weight) * out[implied.index] + weight * implied
    return out
