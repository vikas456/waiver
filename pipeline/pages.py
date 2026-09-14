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
import itertools
import json
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

from . import sleeper
from .config import POSITIONS, REPLACEMENT_RANK_PER_TEAM

SITE = "https://fantasywaiverpicks.com"
LEAGUE_SIZE = 12
# How deep each ranking goes: roughly everyone worth a roster spot.
DEPTH = {"QB": 32, "RB": 60, "WR": 72, "TE": 32}
NAMES = {"QB": "quarterback", "RB": "running back", "WR": "wide receiver", "TE": "tight end"}
ANALYTICS = re.compile(r"<!-- Cloudflare Web Analytics -->.*?<!-- End Cloudflare Web Analytics -->", re.S)

FOOTER = """<footer class="site-footer" aria-label="More from Waiver">
  <a href="/">Compare players</a>
  <a href="/waiver-wire/">Waiver pickups</a>
  <a href="/rankings/">Rankings</a>
  <a href="/rankings/qb/">QB</a>
  <a href="/rankings/rb/">RB</a>
  <a href="/rankings/wr/">WR</a>
  <a href="/rankings/te/">TE</a>
  <a href="/terms/">Terms</a>
  <a href="/privacy/">Privacy</a>
</footer>"""
# Hand-written pages that belong in the sitemap alongside the generated ones.
STATIC_PATHS = ["/terms/", "/privacy/"]

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


def when_playing(player: dict, weights: dict) -> float:
    """Points per game in the games he plays."""
    weeks = player["weeks"]
    if not weeks:
        return 0.0
    return sum(sum(v * weights.get(k, 0) for k, v in w["c"].items()) for w in weeks) / len(weeks)


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
        return sleeper.trending_adds()
    except Exception as err:
        print(f"  trending adds unavailable ({type(err).__name__}); keeping the last waiver page")
        return []


# ---------------------------------------------------------------------------
# Markup
# ---------------------------------------------------------------------------

def _signed(v: float) -> str:
    return f"{'+' if v >= 0 else '−'}{abs(v):.1f}"


def _shell(meta: dict, path: str, title: str, description: str, body: str,
           analytics: str, data: list[dict] | None = None) -> str:
    url = SITE + path
    generated = datetime.fromisoformat(meta["generated"])
    stamp = f"Season {meta['season']} · week {meta['fromWeek']} · updated {generated:%b} {generated.day}"
    ld = "".join(f'<script type="application/ld+json">{json.dumps(d, ensure_ascii=False)}</script>\n'
                 for d in data or [])
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
<meta property="og:image:alt" content="Waiver: who should you pick up? Fantasy football waiver picks from an AI model.">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600&display=swap" rel="stylesheet" media="print" onload="this.media='all'">
<noscript><link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600&display=swap" rel="stylesheet"></noscript>
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


def slugs(players: list[dict]) -> dict[str, str]:
    """A readable, stable address per player, such as josh-allen. Two players
    who share a name are told apart by position, then team."""
    def plain(text: str) -> str:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
        # Apostrophes and full stops vanish rather than split a name, so the
        # address reads the way people type it: jamarr-chase, aj-brown.
        text = re.sub(r"['.]", "", text)
        return re.sub(r"[^a-z0-9]+", "-", text).strip("-")

    base = {p["id"]: plain(p["name"]) for p in players}
    counts: dict[str, int] = {}
    for s in base.values():
        counts[s] = counts.get(s, 0) + 1
    out = {}
    for p in players:
        s = base[p["id"]]
        if counts[s] > 1:
            s = f"{s}-{plain(p['position'])}-{plain(p['team'])}"
        out[p["id"]] = s
    return out


def _player_link(p: dict, slug: dict) -> str:
    return f'<a href="/players/{slug[p["id"]]}/">{e(p["name"])}</a>' if p["id"] in slug else e(p["name"])


# The shorthand fantasy apps use for injury designations. Mirrors SHORT_STATUS in web/app.js.
SHORT_STATUS = {"Questionable": "Q", "Doubtful": "D", "Out": "O", "Suspended": "SUS",
                "Inactive": "NA", "Did not report": "DNR"}


def _tag(p: dict) -> str:
    inj = p.get("injury")
    if not inj:
        return ""
    return f' <span class="tag" title="{e(inj["status"])}">{e(SHORT_STATUS.get(inj["status"], inj["status"]))}</span>'


def _item_list(title: str, players: list[dict], slug: dict) -> dict:
    return {
        "@context": "https://schema.org",
        "@type": "ItemList",
        "name": title,
        "itemListOrder": "https://schema.org/ItemListOrderDescending",
        "numberOfItems": len(players),
        "itemListElement": [
            {"@type": "ListItem", "position": i, "name": p["name"],
             "url": f"{SITE}/players/{slug[p['id']]}/" if p["id"] in slug else f"{SITE}/?p={p['id']}"}
            for i, p in enumerate(players, start=1)],
    }


def _breadcrumbs(*trail: tuple[str, str]) -> dict:
    """Home, then each (name, path) step, which search results show above the title."""
    steps = [("Home", "/"), *trail]
    return {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [{"@type": "ListItem", "position": i, "name": name, "item": SITE + path}
                            for i, (name, path) in enumerate(steps, start=1)],
    }


def position_page(meta: dict, pos: str, ranked: list[dict], ppg: dict, slug: dict,
                  analytics: str) -> str:
    name = NAMES[pos]
    weeks = f"weeks {meta['fromWeek']}–{meta['throughWeek']}"
    title = f"Rest-of-season {pos} rankings for week {meta['fromWeek']}, {meta['season']} — Waiver"
    description = (f"Fantasy football rest-of-season {name} rankings for {weeks} in full PPR, "
                   "from an AI model tested against expert rankings. Updated weekly.")
    rows = "\n".join(
        f'      <tr><td class="num">{i}</td><td>{_player_link(p, slug)}{_tag(p)}</td><td>{e(p["team"])}</td>'
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
                  [_item_list(title, ranked, slug),
                   _breadcrumbs(("Rankings", "/rankings/"), (f"{pos} rankings", f"/rankings/{pos.lower()}/"))])


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
    return _shell(meta, "/rankings/", title, description, body, analytics,
                  [_breadcrumbs(("Rankings", "/rankings/"))])


def _status_note(p: dict) -> str:
    """What is keeping a player off the field, in a sentence, or nothing."""
    inj, weeks = p.get("injury"), p["weeks"]
    if not inj or not weeks:
        return ""
    chance = lambda w: f"{round(w.get('p', p['playProb']) * 100)}%"
    detail = f" ({inj['detail'].lower()})" if inj.get("detail") else ""
    if inj["status"] in ("IR", "PUP"):
        where = "injured reserve" if inj["status"] == "IR" else "the physically unable to perform list"
        out = list(itertools.takewhile(lambda w: w.get("p", 1) == 0, weeks))
        rest = weeks[len(out):]
        text = f"On {where}{detail}."
        if out:
            text += f" He is out through at least week {out[-1]['w']}."
        if rest:
            text += (f" His chance of playing is {chance(rest[0])} in week {rest[0]['w']}"
                     + (f", rising to {chance(rest[-1])} by week {rest[-1]['w']}" if len(rest) > 1 else "")
                     + ", going by how long players placed on reserve have stayed out since 2022.")
    elif inj["status"] in ("Out", "Doubtful", "Questionable"):
        text = (f"Listed as {inj['status'].lower()}{detail} on the latest injury report, so his chance "
                f"of playing in week {weeks[0]['w']} is {chance(weeks[0])}.")
    else:
        text = f"{inj['status']}{detail}: his chance of playing in week {weeks[0]['w']} is {chance(weeks[0])}."
    return f'<p class="lede"><span class="pill bad">{e(inj["status"])}</span> {e(text)}</p>'


def player_page(meta: dict, p: dict, ppg: dict, rank: int, vor: float, weights: dict,
                analytics: str) -> str:
    pos, name = p["position"], p["name"]
    weeks = f"weeks {meta['fromWeek']}–{meta['throughWeek']}"
    inj = p.get("injury") or {}
    title = f"{name} fantasy outlook, week {meta['fromWeek']} — Waiver"
    reserve = {"IR": "is on injured reserve and ", "PUP": "is on the PUP list and "}.get(inj.get("status"), "")
    description = (f"{name} ({pos}, {p['team']}) {reserve}projects for {ppg[p['id']]:.1f} PPR points per game "
                   f"over {weeks}, {pos}{rank} in Waiver's rankings. Updated weekly.")
    s = p.get("stats", {})
    pct = lambda v: "—" if v is None else f"{round(v * 100)}%"
    figures = [("Expected points per game, PPR", f"{ppg[p['id']]:.1f}")]
    if inj:
        figures.append(("Points in games he plays", f"{when_playing(p, weights):.1f}"))
    figures.append(("Over replacement, 12 teams", _signed(vor)))
    figures.append(("Snap share", pct(s.get("snapShare"))))
    if pos == "RB":
        figures += [("Carry share", pct(s.get("carryShare"))), ("Goal-line share", pct(s.get("glShare")))]
    elif pos != "QB":
        figures += [("Target share", pct(s.get("targetShare"))), ("Route rate", pct(s.get("routeRate")))]
    figure_rows = "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in figures)

    def week_row(w: dict) -> str:
        pts = sum(v * weights.get(k, 0) for k, v in w["c"].items())
        prob = w.get("p", p["playProb"])
        return (f'      <tr><td class="num">{w["w"]}</td><td>{e(w["opp"])}</td>'
                f'<td class="num">{round(prob * 100)}%</td><td class="num">{pts:.1f}</td>'
                f'<td class="num">{pts * prob:.1f}</td></tr>')
    schedule = "\n".join(week_row(w) for w in p["weeks"])
    byes = p.get("byeWeeks") or []
    bye_note = f'<p class="footnote">Bye in week {", ".join(map(str, byes))}.</p>' if byes else ""
    photo = (f'<img src="{e(p["headshot"])}" alt="{e(name)}" width="64" height="64">'
             if p.get("headshot") else "")
    body = f"""  <article class="board">
    <div class="player-head">{photo}
      <div><h1>{e(name)} rest-of-season outlook</h1>
      <p class="meta">{e(pos)} · {e(p["team"])} · {e(pos)}{rank} in the rankings</p></div>
    </div>
    {_status_note(p)}
    <dl class="figures">
      {figure_rows}
    </dl>
    <p class="lede">Expected points count every game from week {meta['fromWeek']} to
      {meta['throughWeek']}, each weighted by his chance of playing it, so a game he misses
      counts as zero.</p>
    <div class="board-table-wrap">
    <table class="board-table">
      <thead><tr><th scope="col" class="num">Week</th><th scope="col">Opponent</th><th scope="col" class="num">Chance he plays</th><th scope="col" class="num">Points if he plays</th><th scope="col" class="num">Expected points</th></tr></thead>
      <tbody>
{schedule}
      </tbody>
    </table>
    </div>
    {bye_note}
    <p><a class="share" href="/?p={e(p["id"])}">Compare {e(name)} with another player</a></p>
    <p><a href="/rankings/{pos.lower()}/">See all {NAMES[pos]} rankings</a></p>
  </article>"""
    return _shell(meta, f"/players/{{slug}}/", title, description, body, analytics,
                  [_breadcrumbs(("Rankings", "/rankings/"), (f"{pos} rankings", f"/rankings/{pos.lower()}/"),
                                (name, "/players/{slug}/"))])


def waiver_page(meta: dict, rows: list[tuple[dict, int]], ppg: dict, levels: dict,
                slug: dict, analytics: str) -> str:
    title = f"Waiver wire pickups for week {meta['fromWeek']}, {meta['season']} — Waiver"
    description = (f"The most-added fantasy football players for week {meta['fromWeek']}, ranked by "
                   "how much each is projected to help your team for the rest of the season.")
    ranked = sorted(rows, key=lambda r: ppg[r[0]["id"]] - levels[r[0]["position"]], reverse=True)
    table = "\n".join(
        f'      <tr><td class="num">{i}</td><td>{_player_link(p, slug)}{_tag(p)}</td><td>{e(p["position"])}</td>'
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
                  [_item_list(title, [p for p, _ in ranked], slug),
                   _breadcrumbs(("Waiver pickups", "/waiver-wire/"))])


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

    by_id = {p["id"]: p for p in players}
    trending = [(by_id[gsis], n) for gsis, n in adds if gsis in by_id]

    # Everyone ranked or trending gets a page of their own: the searches a
    # new site can win are specific ones, such as a player's name.
    featured = {p["id"]: p for group in by_pos.values() for p in group}
    featured.update({p["id"]: p for p, _ in trending})
    # A player on reserve falls down the rankings, but people still search his
    # name, so anyone who would rank when healthy keeps his page.
    healthy = {p["id"]: when_playing(p, weights) for p in players}
    for pos in POSITIONS:
        pool = sorted((p for p in players if p["position"] == pos), key=lambda p: healthy[p["id"]], reverse=True)
        featured.update({p["id"]: p for p in pool[:DEPTH[pos]] if p.get("injury")})
    slug = slugs(list(featured.values()))
    pos_rank = {}
    for pos in POSITIONS:
        ordered = sorted((p for p in players if p["position"] == pos), key=lambda p: ppg[p["id"]], reverse=True)
        pos_rank.update({p["id"]: i for i, p in enumerate(ordered, start=1)})

    pages = {"rankings/index.html": hub_page(meta, by_pos, analytics)}
    for pos in POSITIONS:
        pages[f"rankings/{pos.lower()}/index.html"] = position_page(meta, pos, by_pos[pos], ppg, slug, analytics)
    if trending:
        pages["waiver-wire/index.html"] = waiver_page(meta, trending, ppg, levels, slug, analytics)
    for pid, p in featured.items():
        page = player_page(meta, p, ppg, pos_rank[pid], ppg[pid] - levels[p["position"]], weights, analytics)
        pages[f"players/{slug[pid]}/index.html"] = page.replace("{slug}", slug[pid])

    # Last week's player pages go, so a player who drops out of the rankings
    # does not leave a stale page behind.
    shutil.rmtree(out / "players", ignore_errors=True)
    for rel, text in pages.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    # The compare tool links each player who has a page to it.
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data/player-pages.json").write_text(json.dumps(slug, separators=(",", ":"), sort_keys=True))

    # A waiver page from an earlier build stays listed if this week's adds
    # could not be fetched, so the sitemap never drops a live page.
    listed = (["/", "/waiver-wire/", "/rankings/"] + [f"/rankings/{p.lower()}/" for p in POSITIONS]
              + [f"/players/{s}/" for s in sorted(slug.values())] + STATIC_PATHS)
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
    players = sum(1 for w in written if w.startswith("players/"))
    print(f"Wrote {len(written)} pages, {players} of them player pages, and a sitemap to {args.out}")


if __name__ == "__main__":
    main()
