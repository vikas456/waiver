"""Static pages for search engines.

The comparison tool shows nothing until someone types names in, so a search
engine crawling the site finds little to index. These pages carry the same
weekly projections as plain HTML: rest-of-season rankings by position, and
the players fantasy managers are adding this week, ranked by the model. They
are rebuilt with the projections, so they are never older than the data.

    python -m pipeline.pages                    # writes into web/
    python -m pipeline.pages --out /tmp/pages   # anywhere else, as CI does
"""

from __future__ import annotations

import argparse
import html
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path

from .config import POSITIONS, REPLACEMENT_RANK_PER_TEAM

SITE = "https://fantasywaiverpicks.com"
LEAGUE_SIZE = 12
# How deep each ranking goes: roughly everyone worth a roster spot.
DEPTH = {"QB": 32, "RB": 60, "WR": 72, "TE": 32}
NAMES = {"QB": "quarterback", "RB": "running back", "WR": "wide receiver", "TE": "tight end"}
TRENDING = "https://api.sleeper.app/v1/players/nfl/trending/add?lookback_hours=48&limit=50"
ANALYTICS = re.compile(r"<!-- Cloudflare Web Analytics -->.*?<!-- End Cloudflare Web Analytics -->", re.S)

FOOTER = """<footer class="site-footer" aria-label="More from Waiver">
  <a href="/">Compare players</a>
  <a href="/waiver-wire/">Waiver pickups</a>
  <a href="/rankings/">Rankings</a>
  <a href="/rankings/qb/">QB</a>
  <a href="/rankings/rb/">RB</a>
  <a href="/rankings/wr/">WR</a>
  <a href="/rankings/te/">TE</a>
</footer>"""

e = html.escape


# ---------------------------------------------------------------------------
# Numbers, computed exactly as the site computes them
# ---------------------------------------------------------------------------

def expected_ppg(player: dict, weights: dict) -> float:
    """Points per scheduled game with missed games as zero. Mirrors meanPpg
    in web/app.js; keep the two in step."""
    weeks = player["weeks"]
    if not weeks:
        return 0.0
    total = sum(sum(v * weights.get(k, 0) for k, v in w["c"].items())
                * w.get("p", player["playProb"]) for w in weeks)
    return total / len(weeks)


def replacement_levels(players: list[dict], ppg: dict) -> dict:
    """Mirrors replacementLevels in web/app.js for a 12-team league."""
    by_pos: dict[str, list[float]] = {}
    for p in players:
        by_pos.setdefault(p["position"], []).append(ppg[p["id"]])
    levels = {}
    for pos, values in by_pos.items():
        values.sort(reverse=True)
        rank = max(1, round(REPLACEMENT_RANK_PER_TEAM.get(pos, 2) * LEAGUE_SIZE))
        band = values[max(0, rank - 3):min(len(values), rank + 2)]
        levels[pos] = sum(band) / len(band) if band else (values[-1] if values else 0.0)
    return levels


def trending_adds() -> list[tuple[str, int]]:
    """(gsis id, adds) for the players most added on Sleeper over the last
    two days, or nothing when either source is unavailable."""
    try:
        import nflreadpy as nfl

        with urllib.request.urlopen(TRENDING, timeout=20) as res:
            adds = json.load(res)
        ids = nfl.load_ff_playerids().to_pandas()[["sleeper_id", "gsis_id"]].dropna()
        sleeper = ids["sleeper_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        to_gsis = dict(zip(sleeper, ids["gsis_id"]))
        return [(to_gsis[a["player_id"]], a["count"]) for a in adds if a["player_id"] in to_gsis]
    except Exception as err:
        print(f"  trending adds unavailable ({type(err).__name__}); keeping the last waiver page")
        return []


# ---------------------------------------------------------------------------
# Markup
# ---------------------------------------------------------------------------

def _signed(v: float) -> str:
    return f"{'+' if v >= 0 else '−'}{abs(v):.1f}"


def _shell(meta: dict, path: str, title: str, description: str, body: str,
           analytics: str, data: dict | None = None) -> str:
    url = SITE + path
    generated = datetime.fromisoformat(meta["generated"])
    stamp = f"Season {meta['season']} · week {meta['fromWeek']} · updated {generated:%b} {generated.day}"
    ld = (f'<script type="application/ld+json">{json.dumps(data, ensure_ascii=False)}</script>\n'
          if data else "")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0e1211">
<title>{e(title)}</title>
<meta name="description" content="{e(description)}">
<link rel="canonical" href="{url}">
<meta property="og:type" content="website">
<meta property="og:url" content="{url}">
<meta property="og:title" content="{e(title)}">
<meta property="og:description" content="{e(description)}">
<meta property="og:image" content="{SITE}/og-image.png">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/styles.css">
{ld}</head>
<body>

<header class="masthead">
  <a class="wordmark" href="/">Waiver</a>
  <span class="meta">{e(stamp)}</span>
</header>

<main>
{body}
</main>

{FOOTER}
{analytics}
</body>
</html>
"""


def _position_nav(current: str | None) -> str:
    links = []
    for pos in POSITIONS:
        here = ' aria-current="page"' if pos == current else ""
        links.append(f'<a href="/rankings/{pos.lower()}/"{here}>{pos}</a>')
    return f'<nav class="board-nav" aria-label="Rankings by position">{"".join(links)}</nav>'


def _item_list(title: str, players: list[dict]) -> dict:
    return {
        "@context": "https://schema.org",
        "@type": "ItemList",
        "name": title,
        "itemListOrder": "https://schema.org/ItemListOrderDescending",
        "numberOfItems": len(players),
        "itemListElement": [
            {"@type": "ListItem", "position": i, "name": p["name"], "url": f"{SITE}/?p={p['id']}"}
            for i, p in enumerate(players, start=1)],
    }


def position_page(meta: dict, pos: str, ranked: list[dict], ppg: dict, analytics: str) -> str:
    name = NAMES[pos]
    weeks = f"weeks {meta['fromWeek']}–{meta['throughWeek']}"
    title = f"Rest-of-season {pos} rankings for week {meta['fromWeek']}, {meta['season']} — Waiver"
    description = (f"Fantasy football rest-of-season {name} rankings for {weeks} in full PPR, "
                   "from an AI model tested against expert rankings. Updated weekly.")
    rows = "\n".join(
        f'      <tr><td class="num">{i}</td><td>{e(p["name"])}</td><td>{e(p["team"])}</td>'
        f'<td class="num">{ppg[p["id"]]:.1f}</td><td><a href="/?p={e(p["id"])}">Compare</a></td></tr>'
        for i, p in enumerate(ranked, start=1))
    body = f"""  <section class="board">
    <h1>Rest-of-season {name} rankings</h1>
    <p class="lede">Every {name} worth a roster spot, ranked by the points per game
      Waiver projects for {weeks} in full PPR, with missed games counted as zero. To use
      your own scoring or league size, <a href="/">compare players in the tool</a>.</p>
    {_position_nav(pos)}
    <div class="board-table-wrap">
    <table class="board-table">
      <thead><tr><th scope="col" class="num">Rank</th><th scope="col">Player</th><th scope="col">Team</th><th scope="col" class="num">Points per game</th><th scope="col"><span class="visually-hidden">Compare</span></th></tr></thead>
      <tbody>
{rows}
      </tbody>
    </table>
    </div>
  </section>"""
    return _shell(meta, f"/rankings/{pos.lower()}/", title, description, body, analytics,
                  _item_list(title, ranked))


def hub_page(meta: dict, by_pos: dict[str, list[dict]], analytics: str) -> str:
    title = f"Rest-of-season fantasy football rankings, week {meta['fromWeek']} — Waiver"
    description = ("Rest-of-season fantasy football rankings for quarterbacks, running backs, "
                   "wide receivers and tight ends, from an AI model tested against expert rankings.")
    cards = "\n".join(
        f'      <li><a href="/rankings/{pos.lower()}/">{NAMES[pos].capitalize()} rankings</a>'
        f'<span>{e(", ".join(p["name"] for p in by_pos[pos][:3]))}</span></li>'
        for pos in POSITIONS if by_pos.get(pos))
    body = f"""  <section class="board">
    <h1>Rest-of-season fantasy football rankings</h1>
    <p class="lede">Projected points per game for weeks {meta['fromWeek']}–{meta['throughWeek']}
      in full PPR, rebuilt every week from the latest games, depth charts and injury reports.</p>
    <ul class="board-list">
{cards}
    </ul>
  </section>"""
    return _shell(meta, "/rankings/", title, description, body, analytics)


def waiver_page(meta: dict, rows: list[tuple[dict, int]], ppg: dict, levels: dict,
                analytics: str) -> str:
    title = f"Waiver wire pickups for week {meta['fromWeek']}, {meta['season']} — Waiver"
    description = (f"The most-added fantasy football players for week {meta['fromWeek']}, ranked by "
                   "how much each is projected to help your team for the rest of the season.")
    ranked = sorted(rows, key=lambda r: ppg[r[0]["id"]] - levels[r[0]["position"]], reverse=True)
    table = "\n".join(
        f'      <tr><td class="num">{i}</td><td>{e(p["name"])}</td><td>{e(p["position"])}</td>'
        f'<td>{e(p["team"])}</td><td class="num">{ppg[p["id"]]:.1f}</td>'
        f'<td class="num">{_signed(ppg[p["id"]] - levels[p["position"]])}</td>'
        f'<td class="num">{adds:,}</td><td><a href="/?p={e(p["id"])}">Compare</a></td></tr>'
        for i, (p, adds) in enumerate(ranked, start=1))
    top_two = ",".join(p["id"] for p, _ in ranked[:2])
    body = f"""  <section class="board">
    <h1>Waiver wire pickups for week {meta['fromWeek']}</h1>
    <p class="lede">The players fantasy managers added most on Sleeper over the last two
      days, ranked by how much Waiver expects each to help: projected points per game for
      the rest of the season, against the best free agent at the same position in a
      12-team PPR league.</p>
    <p><a class="share" href="/?p={e(top_two)}">Compare the top two</a></p>
    <div class="board-table-wrap">
    <table class="board-table">
      <thead><tr><th scope="col" class="num">Rank</th><th scope="col">Player</th><th scope="col">Pos</th><th scope="col">Team</th><th scope="col" class="num">Points per game</th><th scope="col" class="num">Over replacement</th><th scope="col" class="num">Adds, 48 hours</th><th scope="col"><span class="visually-hidden">Compare</span></th></tr></thead>
      <tbody>
{table}
      </tbody>
    </table>
    </div>
  </section>"""
    return _shell(meta, "/waiver-wire/", title, description, body, analytics,
                  _item_list(title, [p for p, _ in ranked]))


def sitemap(paths: list[str], lastmod: str) -> str:
    urls = "".join(f"  <url>\n    <loc>{SITE}{p}</loc>\n    <lastmod>{lastmod}</lastmod>\n"
                   f"    <changefreq>weekly</changefreq>\n  </url>\n" for p in paths)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{urls}</urlset>\n')


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(payload: dict, out: Path, adds: list[tuple[str, int]], analytics: str) -> list[str]:
    meta, players = payload["meta"], payload["players"]
    weights = meta["scoringFormats"]["ppr"]
    ppg = {p["id"]: expected_ppg(p, weights) for p in players}
    levels = replacement_levels(players, ppg)
    by_pos = {pos: sorted((p for p in players if p["position"] == pos),
                          key=lambda p: ppg[p["id"]], reverse=True)[:DEPTH[pos]]
              for pos in POSITIONS}

    pages = {"rankings/index.html": hub_page(meta, by_pos, analytics)}
    for pos in POSITIONS:
        pages[f"rankings/{pos.lower()}/index.html"] = position_page(meta, pos, by_pos[pos], ppg, analytics)
    by_id = {p["id"]: p for p in players}
    trending = [(by_id[gsis], n) for gsis, n in adds if gsis in by_id]
    if trending:
        pages["waiver-wire/index.html"] = waiver_page(meta, trending, ppg, levels, analytics)

    for rel, text in pages.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    # A waiver page from an earlier build stays listed if this week's adds
    # could not be fetched, so the sitemap never drops a live page.
    listed = ["/", "/waiver-wire/", "/rankings/"] + [f"/rankings/{p.lower()}/" for p in POSITIONS]
    if not (out / "waiver-wire/index.html").exists():
        listed.remove("/waiver-wire/")
    lastmod = datetime.fromisoformat(meta["generated"]).date().isoformat()
    (out / "sitemap.xml").write_text(sitemap(listed, lastmod))
    return sorted(pages)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the static pages search engines index")
    ap.add_argument("--payload", type=Path, default=Path("web/data/projections.json"))
    ap.add_argument("--out", type=Path, default=Path("web"))
    ap.add_argument("--no-trending", action="store_true",
                    help="skip Sleeper's trending adds, for offline runs and CI")
    args = ap.parse_args()

    payload = json.loads(args.payload.read_text())
    if payload["meta"].get("source") == "synthetic" and args.out == Path("web"):
        print("These projections are synthetic, so the live search pages are left alone. "
              "Pass --out to write them somewhere else.")
        return
    index = Path("web/index.html")
    match = ANALYTICS.search(index.read_text()) if index.exists() else None
    written = build(payload, args.out, [] if args.no_trending else trending_adds(),
                    match.group(0) if match else "")
    print(f"Wrote {len(written)} pages and a sitemap to {args.out}: {', '.join(written)}")


if __name__ == "__main__":
    main()
