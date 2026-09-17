"""Keep a copy of what Sleeper projected for a week, before it is played.

Sleeper serves past weeks as well as the coming one, but nothing promises that
a past week's numbers are the ones it showed at the time rather than something
revised since. A weekly report that claims to have beaten another site has to
be able to prove what that site actually said, so the numbers are captured
before the games and graded against the copy.

    python -m tools.sleeper_snapshot --week 3
    python -m tools.sleeper_snapshot --season 2025 --week 10 --out /tmp/w10.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

PROJECTIONS = "https://api.sleeper.app/v1/projections/nfl/regular/{season}/{week}"
OUT_DIR = Path("web/data/sleeper")
PAYLOAD = Path("web/data/projections.json")


def fetch(season: int, week: int, timeout: int = 60) -> dict:
    """Sleeper's projected points per player id, for everyone it projects."""
    url = PROJECTIONS.format(season=season, week=week)
    with urllib.request.urlopen(url, timeout=timeout) as res:
        raw = json.load(res)
    return {sid: float(stats["pts_ppr"]) for sid, stats in raw.items()
            if isinstance(stats, dict) and stats.get("pts_ppr") is not None}


def snapshot(season: int, week: int, payload: Path = PAYLOAD) -> dict:
    """Those projections, keyed by the ids the rest of the site uses.

    Only players the site itself projects are kept: the comparison is over the
    players both sides have an opinion about, which is the only fair way to
    score it.
    """
    theirs = fetch(season, week)
    players = json.loads(payload.read_text())["players"]
    ours = {str(p["sleeperId"]): p for p in players if p.get("sleeperId")}
    matched = {p["id"]: round(theirs[sid], 2) for sid, p in ours.items() if sid in theirs}
    return {
        "season": season,
        "week": week,
        "source": "sleeper",
        "captured": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "projected": matched,
        "matched": len(matched),
        "theirs": len(theirs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture Sleeper's projections for a week")
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season", type=int)
    ap.add_argument("--payload", type=Path, default=PAYLOAD)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    meta = json.loads(args.payload.read_text())["meta"]
    season = args.season or int(meta["season"])
    data = snapshot(season, args.week, args.payload)
    out = args.out or OUT_DIR / f"week-{args.week}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True))
    print(f"Wrote {out} — {data['matched']} of our players, from {data['theirs']} Sleeper projections")


if __name__ == "__main__":
    main()
