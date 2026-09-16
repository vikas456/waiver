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

from . import disagree, sleeper
from .config import KDST_POSITIONS, PA_TIERS, POSITIONS, REPLACEMENT_RANK_PER_TEAM

SITE = "https://fantasywaiverpicks.com"
LEAGUE_SIZE = 12
# How deep each ranking goes: roughly everyone worth a roster spot.
DEPTH = {"QB": 32, "RB": 60, "WR": 72, "TE": 32, "K": 32, "DEF": 32}
# Rankings and navigation cover kickers and team defences too; the pages that
# grade the model against the experts do not, because those two are projected
# a different way (see pipeline/kdst.py).
ALL_POSITIONS = POSITIONS + KDST_POSITIONS
NAMES = {"QB": "quarterback", "RB": "running back", "WR": "wide receiver", "TE": "tight end",
         "K": "kicker", "DEF": "team defence"}
# What people actually type into a search box. "DEF rankings" is nobody's
# search; "D/ST rankings" is.
SEARCH_LABEL = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "kicker", "DEF": "D/ST"}
ANALYTICS = re.compile(r"<!-- Cloudflare Web Analytics -->.*?<!-- End Cloudflare Web Analytics -->", re.S)

FOOTER = """<footer class="site-footer" aria-label="More from Waiver">
  <a href="/">Compare players</a>
  <a href="/waiver-wire/">Waiver pickups</a>
  <a href="/ir-stash/">IR stash</a>
  <a href="/start-sit/">Start or sit</a>
  <a href="/rankings/">Rankings</a>
  <a href="/rankings/qb/">QB</a>
  <a href="/rankings/rb/">RB</a>
  <a href="/rankings/wr/">WR</a>
  <a href="/rankings/te/">TE</a>
  <a href="/model-vs-experts/">Model vs experts</a>
  <a href="/scorecard/">Scorecard</a>
  <a href="/terms/">Terms</a>
  <a href="/privacy/">Privacy</a>
</footer>"""
# The AdSense publisher id, as in web/ads.txt and web/ads.js.
ADSENSE_CLIENT = "ca-pub-1730230746444804"
# The one ad placement, after each page's main content, as in web/index.html.
# web/ads.js fills and reveals it once the AdSense ids are set.
AD_SLOT = """  <aside class="ad-slot" aria-label="Advertisement" hidden>
    <p class="ad-label">Advertisement</p>
    <div class="ad-unit"></div>
    <p class="ad-note">Waiver is free to use, and ads help cover what it costs to run.</p>
  </aside>"""
# Hand-written pages that belong in the sitemap alongside the generated ones.
STATIC_PATHS = ["/terms/", "/privacy/"]

e = html.escape


# ---------------------------------------------------------------------------
# Numbers, computed exactly as the site computes them
# ---------------------------------------------------------------------------

def week_points(week: dict, weights: dict) -> float:
    """One week's points: the stat line, plus the bands a defence's points
    allowed fall into. Mirrors weekPoints in web/app.js."""
    pts = sum(v * weights.get(k, 0) for k, v in week["c"].items())
    if week.get("pa"):
        pts += sum(p * value for p, (_, value) in zip(week["pa"], PA_TIERS))
    return pts


def expected_ppg(player: dict, weights: dict) -> float:
    """Points per scheduled game with missed games as zero. Mirrors meanPpg
    in web/app.js; keep the two in step."""
    weeks = player["weeks"]
    if not weeks:
        return 0.0
    total = sum(week_points(w, weights) * w.get("p", player["playProb"]) for w in weeks)
    return total / len(weeks)


def when_playing(player: dict, weights: dict) -> float:
    """Points per game in the games he plays."""
    weeks = player["weeks"]
    if not weeks:
        return 0.0
    return sum(week_points(w, weights) for w in weeks) / len(weeks)


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
<meta name="google-adsense-account" content="{ADSENSE_CLIENT}">
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
  <a class="wordmark" href="/" aria-label="Waiver">W<span class="ai">ai</span>ver</a>
  <span class="meta">{e(stamp)}</span>
</header>

<main>
{body}
{AD_SLOT}
</main>

{FOOTER}
{analytics}
<script src="/ads.js" defer></script>
</body>
</html>
"""


def _position_nav(current: str | None, positions: list[str] | None = None) -> str:
    links = []
    for pos in positions or ALL_POSITIONS:
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


SUFFIX = re.compile(r"^(jr|sr|ii|iii|iv|v)\.?$", re.I)


def _initials(name: str) -> str:
    """Mirrors initials and lastName in web/app.js: A.J. Brown is AB."""
    bits = [b for b in name.split(" ") if not SUFFIX.match(b)]
    last = bits[-1] if bits else name
    return re.sub(r"[^A-Za-z]", "", name[0] + last[0]).upper()


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
                  analytics: str, positions: list[str] | None = None) -> str:
    name = NAMES[pos]
    weeks = f"weeks {meta['fromWeek']}–{meta['throughWeek']}"
    title = (f"Rest-of-season {SEARCH_LABEL[pos]} rankings for week {meta['fromWeek']}, "
             f"{meta['season']} — Waiver")
    description = (f"Fantasy football rest-of-season {name} rankings for {weeks} in full PPR, "
                   "from an AI model tested against expert rankings. Updated weekly.")
    rows = "\n".join(
        f'      <tr><td class="num">{i}</td><td>{_player_link(p, slug)}{_tag(p)}</td><td>{e(p["team"])}</td>'
        f'<td class="num">{ppg[p["id"]]:.1f}</td><td><a href="/?p={e(p["id"])}">Compare</a></td></tr>'
        for i, p in enumerate(ranked, start=1))
    # These two positions deserve a warning rather than a ranking that looks
    # as authoritative as the others.
    caveat = ('\n    <p class="lede"><span class="pill warn">Close to a coin flip</span> '
              f'Tested on the 2024 and 2025 seasons, putting two {NAMES[pos]}s in the right order for '
              'the next four games came out at 58% for kickers and 57% for defences, against 86% at the '
              'positions where usage can be measured. There is no usage to measure here: both are '
              'projected from what the betting market expects of the game, their own season, and expert '
              'consensus. Use this to break a tie, not to plan around.</p>') if pos in KDST_POSITIONS else ""
    body = f"""  <section class="board">
    <h1>Rest-of-season {name} rankings</h1>
    <p class="lede">Every {name} worth a roster spot, ranked by the points per game
      Waiver projects for {weeks} in full PPR, with missed games counted as zero. To use
      your own scoring or league size, <a href="/">compare players in the tool</a>.</p>{caveat}
    {_position_nav(pos, positions)}
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
        for pos in ALL_POSITIONS if by_pos.get(pos))
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


def _kdst_player_page(meta: dict, p: dict, pos: str, rank: int, figure_rows: str,
                      weights: dict, analytics: str) -> str:
    """A kicker's or a defence's page.

    Shorter than a skill player's on purpose: there is no usage to show, and
    the honest thing to say is how little anyone knows week to week.
    """
    name, weeks = p["name"], f"weeks {meta['fromWeek']}-{meta['throughWeek']}"
    kicker = pos == "K"
    title = f"{name} fantasy outlook, week {meta['fromWeek']} — Waiver"
    description = (f"{name} projects {expected_ppg(p, weights):.1f} points per game over {weeks}, "
                   f"{pos}{rank} in Waiver's rankings. Updated weekly.")

    def row(w: dict) -> str:
        return (f'      <tr><td class="num">{w["w"]}</td><td>{e(w["opp"])}</td>'
                f'<td class="num">{week_points(w, weights):.1f}</td></tr>')
    schedule = "\n".join(row(w) for w in p["weeks"])
    badge = f'<span class="avatar" aria-hidden="true">{e(_initials(name) if kicker else p["team"])}</span>'
    # A defence is a team, not a person.
    what = ("how much his offence is expected to score, since extra points follow it and field goals "
            "come from drives that stall, nudged by his own season and blended with expert consensus"
            ) if kicker else (
            "how much the offence it faces is expected to score, which drives sacks, takeaways and the "
            "points it gives up, nudged by its own season and blended with expert consensus")
    body = f"""  <article class="board">
    <div class="player-head">{badge}
      <div><h1>{e(name)} outlook</h1>
      <p class="meta">{e(pos if kicker else "Team defence")} · {e(p["team"])} · {e(pos)}{rank} in the rankings</p></div>
    </div>
    {_status_note(p)}
    <dl class="figures">
      {figure_rows}
    </dl>
    <p class="lede">A {NAMES[pos]} is projected from {what}. Sportsbooks price only the coming week or
      two, so every week after that takes {"his" if kicker else "its"} team's typical game and they all
      read alike until the lines are posted.</p>
    <div class="board-table-wrap">
    <table class="board-table">
      <thead><tr><th scope="col" class="num">Week</th><th scope="col">Opponent</th><th scope="col" class="num">Projected points</th></tr></thead>
      <tbody>
{schedule}
      </tbody>
    </table>
    </div>
    <p><a class="share" href="/?p={e(p["id"])}">Compare {e(name)} with another {NAMES[pos]}</a></p>
    <p><a href="/rankings/{pos.lower()}/">See all {NAMES[pos]} rankings</a></p>
    <p class="footnote">Kickers and team defences are close to unpredictable: over the 2024 and 2025
      seasons, ordering two of them correctly for the next four games came out at 58% for kickers and
      57% for defences, against 86% for the positions where usage can be measured. Treat these
      rankings as a tie-breaker, not a plan.</p>
  </article>"""
    return _shell(meta, f"/players/{{slug}}/", title, description, body, analytics,
                  [_breadcrumbs(("Rankings", "/rankings/"), (f"{pos} rankings", f"/rankings/{pos.lower()}/"),
                                (name, "/players/{slug}/"))])


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
    if pos in KDST_POSITIONS:
        per = lambda key: sum(w["c"].get(key, 0) for w in p["weeks"]) / (len(p["weeks"]) or 1)
        if pos == "K":
            figures += [("Field goals a game", f"{per('fg_0_39') + per('fg_40_49') + per('fg_50'):.1f}"),
                        ("Extra points a game", f"{per('pat'):.1f}")]
        else:
            figures += [("Sacks a game", f"{per('sack'):.1f}"),
                        ("Takeaways a game", f"{per('interception_def') + per('fumble_recovery'):.1f}"),
                        ("Shutout odds", pct(sum((w.get("pa") or [0])[0] for w in p["weeks"])
                                             / (len(p["weeks"]) or 1)))]
        figure_rows = "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in figures)
        return _kdst_player_page(meta, p, pos, rank, figure_rows, weights, analytics)
    figures.append(("Snap share", pct(s.get("snapShare"))))
    if pos == "RB":
        figures += [("Carry share", pct(s.get("carryShare"))), ("Goal-line share", pct(s.get("glShare")))]
    elif pos != "QB":
        figures += [("Target share", pct(s.get("targetShare"))), ("Route rate", pct(s.get("routeRate")))]
    figure_rows = "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in figures)

    def week_row(w: dict) -> str:
        pts = week_points(w, weights)
        prob = w.get("p", p["playProb"])
        return (f'      <tr><td class="num">{w["w"]}</td><td>{e(w["opp"])}</td>'
                f'<td class="num">{round(prob * 100)}%</td><td class="num">{pts:.1f}</td>'
                f'<td class="num">{pts * prob:.1f}</td></tr>')
    schedule = "\n".join(week_row(w) for w in p["weeks"])
    byes = p.get("byeWeeks") or []
    bye_note = f'<p class="footnote">Bye in week {", ".join(map(str, byes))}.</p>' if byes else ""
    badge = f'<span class="avatar" aria-hidden="true">{e(_initials(name))}</span>'
    body = f"""  <article class="board">
    <div class="player-head">{badge}
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


# Lists a stash can hold a player through. Anyone else missing time is week to
# week, which the injury report covers.
STASH_STATUSES = {"IR": "IR", "PUP": "PUP"}
# The fantasy playoffs start here in most leagues, which is when a stash pays off.
PLAYOFF_WEEK = 15


def stash_rows(players: list[dict], weights: dict, levels: dict) -> list[dict]:
    """Players on reserve worth holding, best first.

    A week's chance of playing is his chance of being back from reserve times
    the chance any healthy player plays, so dividing by the second leaves his
    return odds. His value is what he adds over a replacement-level free agent
    in the games he is back for; a week in which he would not beat one is worth
    nothing, since he would sit.
    """
    rows = []
    for p in players:
        inj = p.get("injury") or {}
        if inj.get("status") not in STASH_STATUSES or not p["weeks"]:
            continue
        base = p["playProb"] or 1.0
        repl = levels[p["position"]]
        back = [(w["w"], min(1.0, w.get("p", base) / base)) for w in p["weeks"]]
        value = sum(w.get("p", base) * max(0.0, sum(v * weights.get(k, 0) for k, v in w["c"].items()) - repl)
                    for w in p["weeks"])
        by = min(PLAYOFF_WEEK, p["weeks"][-1]["w"])
        rows.append({
            "player": p, "value": value, "whenBack": when_playing(p, weights), "gap": when_playing(p, weights) - repl,
            "earliest": next((wk for wk, c in back if c > 0), None),
            "byWeek": by, "backBy": max((c for wk, c in back if wk <= by), default=0.0),
        })
    return sorted(rows, key=lambda r: r["value"], reverse=True)


# Close enough to a free agent's output, in points per game when back, that
# an injury ahead of him on the depth chart could make him worth a stash.
WATCH_GAP = 2.0


def split_stash(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Players worth a stash now, and ones a little short of it. A player
    with little chance of playing even when healthy, such as a backup
    quarterback, is neither."""
    worth = [r for r in rows if r["value"] >= 0.05]
    watch = sorted((r for r in rows if r["value"] < 0.05 and r["gap"] >= -WATCH_GAP and r["backBy"] >= 0.2),
                   key=lambda r: r["whenBack"], reverse=True)
    return worth, watch


def _stash_who(p: dict, slug: dict) -> str:
    """Name, list and injury in one cell, so the table stays narrow on a phone."""
    detail = (p["injury"].get("detail") or "").strip()
    return (f'{_player_link(p, slug)} <span class="tag">{e(STASH_STATUSES[p["injury"]["status"]])}</span>'
            + (f'<span class="sub">{e(detail)}</span>' if detail else ""))


def stash_page(meta: dict, rows: list[dict], levels: dict, slug: dict, analytics: str) -> str:
    title = f"IR stash rankings for week {meta['fromWeek']}, {meta['season']} — Waiver"
    description = ("Fantasy football players on injured reserve worth stashing, ranked by what they "
                   "should add once back, with their odds of returning. Updated weekly.")

    def row(i: int, r: dict) -> str:
        p = r["player"]
        earliest = f"Week {r['earliest']}" if r["earliest"] else "—"
        return (f'      <tr><td class="num">{i}</td><td>{_stash_who(p, slug)}</td>'
                f'<td>{e(p["position"])}</td><td class="num">{earliest}</td>'
                f'<td class="num">{round(r["backBy"] * 100)}%</td><td class="num">{r["whenBack"]:.1f}</td>'
                f'<td class="num">{r["value"]:.1f}</td></tr>')

    worth, watch = split_stash(rows)
    by = rows[0]["byWeek"] if rows else PLAYOFF_WEEK
    table = "\n".join(row(i, r) for i, r in enumerate(worth, start=1))
    content = f"""    <div class="board-table-wrap">
    <table class="board-table stash-table">
      <thead><tr><th scope="col" class="num">Rank</th><th scope="col">Player</th><th scope="col">Pos</th><th scope="col" class="num">Earliest return</th><th scope="col" class="num">Back by week {by}</th><th scope="col" class="num">Points per game when back</th><th scope="col" class="num">Stash value</th></tr></thead>
      <tbody>
{table}
      </tbody>
    </table>
    </div>""" if worth else ('    <p class="lede">No player on injured reserve or the PUP list projects to beat '
                             "a free agent once he is back, so there is no one worth stashing this week.</p>")
    if watch:
        def watch_row(r: dict) -> str:
            p = r["player"]
            earliest = f"Week {r['earliest']}" if r["earliest"] else "—"
            return (f'      <tr><td>{_stash_who(p, slug)}</td><td>{e(p["position"])}</td>'
                    f'<td class="num">{earliest}</td><td class="num">{round(r["backBy"] * 100)}%</td>'
                    f'<td class="num">{r["whenBack"]:.1f}</td><td class="num">{_signed(r["gap"])}</td></tr>')
        content += f"""
    <h2>Worth watching</h2>
    <p class="lede">Close to worth a stash, but projected just short of the best free agent at his
      position once he is back. An injury ahead of one of them on the depth chart could change that.</p>
    <div class="board-table-wrap">
    <table class="board-table stash-table">
      <thead><tr><th scope="col">Player</th><th scope="col">Pos</th><th scope="col" class="num">Earliest return</th><th scope="col" class="num">Back by week {by}</th><th scope="col" class="num">Points per game when back</th><th scope="col" class="num">Against a free agent</th></tr></thead>
      <tbody>
{chr(10).join(watch_row(r) for r in watch)}
      </tbody>
    </table>
    </div>"""
    body = f"""  <section class="board">
    <h1>IR stash rankings for week {meta['fromWeek']}</h1>
    <p class="lede">Players on injured reserve or the PUP list worth holding onto, ranked by what
      each should add once he is back. If your league has IR slots, holding one of these players
      costs you no bench spot.</p>
{content}
    <p class="footnote">Stash value is the points a player should add over the best free agent at
      his position in a 12-team PPR league, from week {meta['fromWeek']} to {meta['throughWeek']},
      counting each game by his chance of being back for it. A game he would not beat that free agent
      in counts as zero. Return odds come from every move to injured reserve during weeks 1–10 of
      2022–2025: no one came back before the four-game minimum was up, and about half never
      returned that season. Hamstring, ankle and knee injuries have their own odds. Reported
      timelines, such as "out four to six weeks", are not used yet, so a player with a known short
      absence may be back sooner than shown here.</p>
  </section>"""
    return _shell(meta, "/ir-stash/", title, description, body, analytics,
                  [_item_list(title, [r["player"] for r in worth], slug),
                   _breadcrumbs(("IR stash", "/ir-stash/"))])



# ---------------------------------------------------------------------------
# Start or sit, one week at a time
# ---------------------------------------------------------------------------

# Where the last startable player at each position sits in a week's ranking,
# in a twelve-team league: the same line the rest of the site calls replacement
# level, read here as "the worst player you would put in your lineup".
def startable_rank(pos: str) -> int:
    return max(1, round(REPLACEMENT_RANK_PER_TEAM.get(pos, 2) * LEAGUE_SIZE))


def week_entry(player: dict, weights: dict, week: int, dud_odds: dict) -> dict | None:
    """One player's week: points if he plays, his chance of playing, and the
    floor and ceiling the site draws, or nothing when he has no game."""
    wk = next((w for w in player["weeks"] if w["w"] == week), None)
    if wk is None:
        return None
    from scipy.stats import gamma

    if_plays = week_points(wk, weights)
    chance = wk.get("p", player["playProb"])
    mean = max(if_plays, 0.05)
    sd = max(wk.get("sd", 4.0), 0.5)
    # Mirrors dudChance and the mixture in web/app.js: some games he plays are
    # duds, so they are drawn apart from the rest of the curve.
    fit = (dud_odds or {}).get(player["position"])
    z = 0.0
    if fit:
        import math
        z = 1 / (1 + math.exp(-(fit["intercept"] + fit["slope"] * math.log(max(mean, 0.2)))))
        z = min(0.6, max(0.0, z))
    well = mean / (1 - z)
    shape, scale = (well / sd) ** 2, sd ** 2 / well
    q = lambda t: 0.0 if (t - z) / (1 - z) <= 0 else float(gamma.ppf((t - z) / (1 - z), shape, scale=scale))
    return {"player": player, "opp": wk["opp"], "chance": chance, "ifPlays": if_plays,
            "points": if_plays * chance, "floor": q(0.10), "ceiling": q(0.90)}


def start_sit_table(players: list[dict], weights: dict, week: int, dud_odds: dict) -> dict[str, list[dict]]:
    """Every player with a game that week, best first, by position."""
    table: dict[str, list[dict]] = {}
    for p in players:
        entry = week_entry(p, weights, week, dud_odds)
        if entry:
            table.setdefault(p["position"], []).append(entry)
    for pos, rows in table.items():
        rows.sort(key=lambda r: r["points"], reverse=True)
        for i, r in enumerate(rows, start=1):
            r["rank"] = i
    return table


def verdict(row: dict) -> tuple[str, str]:
    """Start him, a flex call, or sit him, with the pill to show it in."""
    line = startable_rank(row["player"]["position"])
    if row["rank"] <= line - 3:
        return "Start him", "good"
    if row["rank"] <= line + 3:
        return "Flex call", "neutral"
    return "Sit him", "bad"


def _verdict_text(row: dict, meta: dict) -> str:
    p, pos = row["player"], row["player"]["position"]
    rank, line = row["rank"], startable_rank(row["player"]["position"])
    name = p["name"]
    call, _ = verdict(row)
    week = meta["fromWeek"]
    where = (f"{pos}{rank} this week, inside the {pos}{line} who start in a twelve-team league"
             if rank <= line else
             f"{pos}{rank} this week, outside the {pos}{line} who start in a twelve-team league")
    if call == "Start him":
        lead = f"Start {name} in week {week}."
    elif call == "Flex call":
        lead = f"{name} is a flex call in week {week}."
    else:
        lead = f"Sit {name} in week {week} if you have another option."
    return (f"{lead} Waiver projects {row['points']:.1f} points against {row['opp']}, which makes him "
            f"{where}. A bad week for him looks like {row['floor']:.1f} points and a good one "
            f"{row['ceiling']:.1f}.")


def start_sit_page(meta: dict, row: dict, peers: list[dict], slug: dict, analytics: str) -> str:
    p = row["player"]
    pos, week = p["position"], meta["fromWeek"]
    call, tone = verdict(row)
    title = f"Start or sit {p['name']} in week {week}? — Waiver"
    description = (f"{p['name']} projects {row['points']:.1f} points against {row['opp']} in week {week}, "
                   f"{pos}{row['rank']} this week, with a floor of {row['floor']:.1f} and a ceiling of "
                   f"{row['ceiling']:.1f}.")
    inj = p.get("injury")
    note = ""
    if inj:
        note = (f'<p class="lede"><span class="pill bad">{e(inj["status"])}</span> '
                f'{e(f"Listed as {inj['status'].lower()}" + (f" ({inj['detail'].lower()})" if inj.get("detail") else "") + f", so his chance of playing is {round(row['chance'] * 100)}%. The projection counts that chance; the floor and ceiling are for a game he plays.")}</p>')
    figures = [("Projected points", f"{row['points']:.1f}"), ("Floor", f"{row['floor']:.1f}"),
               ("Ceiling", f"{row['ceiling']:.1f}"), ("Rank this week", f"{pos}{row['rank']}"),
               ("Opponent", e(row["opp"])), ("Chance he plays", f"{round(row['chance'] * 100)}%")]
    figure_rows = "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in figures)
    others = "\n".join(
        f'      <tr><td>{_player_link(r["player"], slug)}{_tag(r["player"])}</td>'
        f'<td class="num">{pos}{r["rank"]}</td><td>{e(r["opp"])}</td>'
        f'<td class="num">{r["points"]:.1f}</td><td class="num">{r["floor"]:.1f}\u2013{r["ceiling"]:.1f}</td>'
        f'<td><a href="/?p={e(p["id"])},{e(r["player"]["id"])}&amp;w={week}-{week}">Compare</a></td></tr>'
        for r in peers)
    body = f"""  <article class="board">
    <h1>Start or sit {e(p['name'])} in week {week}?</h1>
    <p class="lede"><span class="pill {tone}">{call}</span> {e(_verdict_text(row, meta))}</p>
    {note}
    <dl class="figures">
      {figure_rows}
    </dl>
    <h2>Others at his position this week</h2>
    <div class="board-table-wrap">
    <table class="board-table">
      <thead><tr><th scope="col">Player</th><th scope="col" class="num">Rank</th><th scope="col">Opponent</th><th scope="col" class="num">Projected</th><th scope="col" class="num">Floor\u2013ceiling</th><th scope="col"><span class="visually-hidden">Compare</span></th></tr></thead>
      <tbody>
{others}
      </tbody>
    </table>
    </div>
    <p><a class="share" href="/?p={e(p['id'])}&amp;w={week}-{week}">Compare {e(p['name'])} with your own players</a></p>
    <p><a href="/players/{slug[p['id']]}/">See his rest-of-season outlook</a> ·
      <a href="/start-sit/">All start or sit calls for week {week}</a></p>
    <p class="footnote">Projected points count his chance of playing. The floor and the ceiling are for a
      game he plays: he should score below the floor about one week in ten, and above the ceiling about
      one week in ten. Tested on single games in the 2024 and 2025 seasons. The line between starting and
      sitting is the {pos}{startable_rank(pos)} in a twelve-team league; deeper leagues start more.</p>
  </article>"""
    return _shell(meta, f"/start-sit/{{slug}}/", title, description, body, analytics,
                  [_breadcrumbs(("Start or sit", "/start-sit/"), (p["name"], f"/start-sit/{{slug}}/"))])


def start_sit_hub(meta: dict, table: dict[str, list[dict]], pages: set[str], slug: dict,
                  analytics: str) -> str:
    week = meta["fromWeek"]
    title = f"Start or sit: week {week} calls for every position, {meta['season']} — Waiver"
    description = (f"Who to start and who to sit in week {week}, with a projection, a floor and a "
                   "ceiling for every player, from an AI model tested against expert rankings.")
    sections = []
    for pos in ALL_POSITIONS:
        rows = [r for r in table.get(pos, []) if r["player"]["id"] in pages][:15]
        if not rows:
            continue
        line = startable_rank(pos)
        body = "\n".join(
            f'      <tr><td class="num">{r["rank"]}</td>'
            f'<td><a href="/start-sit/{slug[r["player"]["id"]]}/">{e(r["player"]["name"])}</a>{_tag(r["player"])}</td>'
            f'<td>{e(r["player"]["team"])}</td><td>{e(r["opp"])}</td>'
            f'<td class="num">{r["points"]:.1f}</td>'
            f'<td class="num">{r["floor"]:.1f}\u2013{r["ceiling"]:.1f}</td></tr>'
            for r in rows)
        sections.append(f"""    <h2 id="{pos.lower()}">{NAMES[pos].capitalize()}s</h2>
    <p class="lede">The {pos}{line} is where starting ends in a twelve-team league.</p>
    <div class="board-table-wrap">
    <table class="board-table">
      <thead><tr><th scope="col" class="num">Rank</th><th scope="col">Player</th><th scope="col">Team</th><th scope="col">Opponent</th><th scope="col" class="num">Projected</th><th scope="col" class="num">Floor\u2013ceiling</th></tr></thead>
      <tbody>
{body}
      </tbody>
    </table>
    </div>""")
    nav = "".join(f'<a href="#{pos.lower()}">{pos}</a>' for pos in POSITIONS if table.get(pos))
    body = f"""  <section class="board">
    <h1>Start or sit in week {week}</h1>
    <p class="lede">What Waiver projects for week {week} alone, with a floor and a ceiling for each
      player, so you can start the steady one when you are ahead and the volatile one when you need
      points. To weigh up two of your own players, <a href="/?w={week}-{week}">compare them in the tool</a>.</p>
    <nav class="board-nav" aria-label="Positions">{nav}</nav>
{chr(10).join(sections)}
    <p class="footnote">Projected points count each player's chance of playing. A floor is a bad week
      and a ceiling a good one: about one week in ten falls outside each. Rebuilt every week.</p>
  </section>"""
    return _shell(meta, "/start-sit/", title, description, body, analytics,
                  [_breadcrumbs(("Start or sit", "/start-sit/"))])


def _reason_text(p: dict, weights: dict, higher: bool, labels: dict) -> str:
    # A quarterback runs no routes; for him that group comes down to snaps.
    name = lambda g: "Snap share" if (g == "route_role" and p["position"] == "QB") else labels.get(g, g)
    parts = [f"{name(g)} {_signed(v)}" for g, v in disagree.reasons(p, weights, higher)]
    return " · ".join(parts) or "—"


def experts_page(meta: dict, table: dict, slug: dict, labels: dict, analytics: str) -> str:
    weights = meta["scoringFormats"]["ppr"]
    weeks = f"weeks {meta['fromWeek']}–{meta['throughWeek']}"
    title = f"Where the model disagrees with the experts, week {meta['fromWeek']} — Waiver"
    description = ("The players Waiver's model ranks furthest from FantasyPros' expert consensus, "
                   "by position, with what the model sees in each. Updated weekly.")
    sections = []
    for pos, sides in disagree.biggest(table, 5).items():
        blocks = []
        for side, heading in (("higher", "Waiver ranks him higher"), ("lower", "Waiver ranks him lower")):
            if not sides[side]:
                continue
            rows = "\n".join(
                f'      <tr><td>{_player_link(r["player"], slug)}{_tag(r["player"])}</td>'
                f'<td class="num">{pos}{r["modelRank"]}</td><td class="num">{pos}{r["consensusRank"]}</td>'
                f'<td class="num">{r["modelPpg"]:.1f}</td>'
                f'<td>{e(_reason_text(r["player"], weights, side == "higher", labels))}</td></tr>'
                for r in sides[side])
            blocks.append(f"""    <h3>{heading}</h3>
    <div class="board-table-wrap">
    <table class="board-table">
      <thead><tr><th scope="col">Player</th><th scope="col" class="num">Waiver</th><th scope="col" class="num">Experts</th><th scope="col" class="num">Points per game</th><th scope="col">What the model sees</th></tr></thead>
      <tbody>
{rows}
      </tbody>
    </table>
    </div>""")
        if blocks:
            sections.append(f'    <h2 id="{pos.lower()}">{NAMES[pos].capitalize()}s</h2>\n' + "\n".join(blocks))
    nav = "".join(f'<a href="#{pos.lower()}">{pos}</a>' for pos in POSITIONS if table.get(pos))
    content = ("\n".join(sections) if sections else
               '    <p class="lede">Expert consensus ranks are not available for this build, '
               "so there is nothing to compare yet.</p>")
    when = meta.get("consensusDate")
    as_of = f", taken {datetime.fromisoformat(when):%B} {datetime.fromisoformat(when).day}," if when else ""
    body = f"""  <section class="board">
    <h1>Where the model disagrees with the experts</h1>
    <p class="lede">The players Waiver ranks furthest from FantasyPros' expert consensus{as_of} for
      {weeks}. These are the calls that set the model apart, and the ones most likely to be
      wrong, so the <a href="/scorecard/">scorecard</a> grades them once the games are played.</p>
    <nav class="board-nav" aria-label="Positions">{nav}</nav>
{content}
    <p class="footnote">Both ranks are by expected points per game for the rest of the season,
      counting games a player is expected to miss. Waiver's rank is the model's own, before the
      site blends in the consensus. What the model sees lists its strongest factors, in points
      per game against a typical player at the position. The consensus is gathered once a week,
      so news since then, such as an injury, can show up here as a disagreement. Players with an
      injury designation are left off the side where Waiver ranks them higher, since the experts
      may know more about when they will be back.</p>
  </section>"""
    return _shell(meta, "/model-vs-experts/", title, description, body, analytics,
                  [_breadcrumbs(("Model vs experts", "/model-vs-experts/"))])


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _accuracy_table(first: str, rows: list[tuple[str, dict]]) -> str:
    body = "\n".join(
        f'      <tr><td>{e(label)}</td><td class="num">{_pct(r.get("site"))}</td>'
        f'<td class="num">{_pct(r.get("consensus"))}</td><td class="num">{_pct(r.get("last4"))}</td></tr>'
        for label, r in rows)
    return f"""    <div class="board-table-wrap">
    <table class="board-table">
      <thead><tr><th scope="col">{first}</th><th scope="col" class="num">Waiver</th><th scope="col" class="num">Experts</th><th scope="col" class="num">Last four games</th></tr></thead>
      <tbody>
{body}
      </tbody>
    </table>
    </div>"""


def scorecard_page(meta: dict, card: dict | None, record: dict | None, slug: dict,
                   analytics: str) -> str:
    title = f"Scorecard: how Waiver's picks have done, {meta['season']} — Waiver"
    description = ("Every week Waiver grades its own projections against real results and the "
                   "expert consensus, losing weeks included, beside its tested 2024 and 2025 record.")
    weeks = (card or {}).get("weeks", [])
    average = lambda rows: {k: sum(r[k] for r in rows) / len(rows) for k in ("site", "consensus", "last4")}

    if weeks:
        graded_weeks = [w["overall"] for w in weeks if w["overall"]]
        season_rows = ([(f"Week {w['week']}", w["overall"]) for w in weeks]
                       + ([("Season so far", average(graded_weeks))] if graded_weeks else []))
        season = (f'    <p class="lede">Each week\'s projections, graded on that week\'s games. A pair of players '
                  f'counts when their scores were at least two points apart, and a missed game counts as zero.</p>\n'
                  + _accuracy_table("Week", season_rows))
    else:
        season = (f'    <p class="lede">The first graded week arrives once week {meta["fromWeek"]} has been '
                  "played. Every week after that is added automatically, including the weeks the experts "
                  "do better.</p>")

    calls = (card or {}).get("calls", {})
    latest = weeks[-1]["calls"] if weeks else []
    if latest:
        rows = "\n".join(
            f'      <tr><td>{_player_link({"id": c["id"], "name": c["name"]}, slug)}</td>'
            f'<td class="num">{c["position"]}{c["modelRank"]}</td><td class="num">{c["position"]}{c["consensusRank"]}</td>'
            f'<td class="num">{c["position"]}{c["finish"]} · {c["points"]:.1f}</td>'
            f'<td><span class="pill {"good" if c["closer"] == "model" else "neutral"}">'
            f'{ {"model": "Waiver", "experts": "Experts", "even": "Even"}[c["closer"]] }</span></td></tr>'
            for c in latest)
        tally = sum(calls.values())
        graded = f"""    <p class="lede">Across the season so far, the player finished nearer Waiver's rank on
      {calls.get("model", 0)} of {tally} of these calls, and nearer the experts' on {calls.get("experts", 0)}.
      Week {weeks[-1]["week"]}'s:</p>
    <div class="board-table-wrap">
    <table class="board-table">
      <thead><tr><th scope="col">Player</th><th scope="col" class="num">Waiver</th><th scope="col" class="num">Experts</th><th scope="col" class="num">Finished</th><th scope="col">Nearer</th></tr></thead>
      <tbody>
{rows}
      </tbody>
    </table>
    </div>"""
    else:
        graded = ('    <p class="lede">Each week\'s biggest <a href="/model-vs-experts/">disagreements with the '
                  "experts</a> are graded here once the games are played: did the player finish nearer "
                  "Waiver's rank or the experts'?</p>")

    if record:
        avg = record["average"]
        seasons = record["seasons"]
        figures = "".join(f"<div><dt>{label}</dt><dd>{_pct(avg[k])}</dd></div>" for k, label in
                          (("site", "Waiver"), ("consensus", "Expert consensus"), ("last4", "Last four games")))
        by_pos = [(NAMES[pos].capitalize() + "s", average([s["byPosition"][pos] for s in seasons.values()]))
                  for pos in POSITIONS if all(pos in s["byPosition"] for s in seasons.values())]
        by_season = [(label, s["overall"]) for label, s in seasons.items()]
        tested = f"""    <p class="lede">Before the site launched, the model was replayed over every week of the
      {" and ".join(sorted(seasons))} seasons: retrained each week on only what was known at the time, then
      asked to order each position by points per game over the next {record.get("horizon", 4)} games.</p>
    <dl class="figures">{figures}</dl>
{_accuracy_table("Season", by_season)}
{_accuracy_table("Position", by_pos)}"""
    else:
        tested = '    <p class="lede">The tested record is not available in this build.</p>'

    body = f"""  <section class="board">
    <h1>How Waiver's picks have done</h1>
    <p class="lede">Waiver grades itself every week, in public, against the average of expert
      rankings and against going by recent points. Weeks it loses stay on the page.</p>
    <h2>This season</h2>
{season}
    <h2>Biggest calls against the experts</h2>
{graded}
    <h2>Tested record</h2>
{tested}
    <p class="footnote">Accuracy is the share of pairs of players at the same position that were put in
      the right order. Past accuracy is no guarantee of future results.</p>
  </section>"""
    return _shell(meta, "/scorecard/", title, description, body, analytics,
                  [_breadcrumbs(("Scorecard", "/scorecard/"))])


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


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
              for pos in ALL_POSITIONS}

    by_id = {p["id"]: p for p in players}
    trending = [(by_id[gsis], n) for gsis, n in adds if gsis in by_id]

    # Everyone ranked or trending gets a page of their own: the searches a
    # new site can win are specific ones, such as a player's name.
    featured = {p["id"]: p for group in by_pos.values() for p in group}
    featured.update({p["id"]: p for p, _ in trending})
    # A player on reserve falls down the rankings, but people still search his
    # name, so anyone who would rank when healthy keeps his page.
    healthy = {p["id"]: when_playing(p, weights) for p in players}
    for pos in ALL_POSITIONS:
        pool = sorted((p for p in players if p["position"] == pos), key=lambda p: healthy[p["id"]], reverse=True)
        featured.update({p["id"]: p for p in pool[:DEPTH[pos]] if p.get("injury")})
    slug = slugs(list(featured.values()))
    pos_rank = {}
    for pos in ALL_POSITIONS:
        ordered = sorted((p for p in players if p["position"] == pos), key=lambda p: ppg[p["id"]], reverse=True)
        pos_rank.update({p["id"]: i for i, p in enumerate(ordered, start=1)})

    # A position with nobody in it gets no page: early in a season, or in a
    # build where kickers and defences could not be loaded, an empty table is
    # worse than no page at all.
    ranked_positions = [pos for pos in ALL_POSITIONS if by_pos.get(pos)]
    pages = {"rankings/index.html": hub_page(meta, by_pos, analytics)}
    for pos in ranked_positions:
        pages[f"rankings/{pos.lower()}/index.html"] = position_page(
            meta, pos, by_pos[pos], ppg, slug, analytics, ranked_positions)
    if trending:
        pages["waiver-wire/index.html"] = waiver_page(meta, trending, ppg, levels, slug, analytics)
    players_by_pos = {p["id"]: p["position"] for p in players}
    stash = stash_rows(players, weights, levels)
    # A stash is searched by name as much as anyone, so each has a page too.
    for r in itertools.chain(*split_stash(stash)):
        featured.setdefault(r["player"]["id"], r["player"])
    slug = slugs(list(featured.values()))
    pages["ir-stash/index.html"] = stash_page(meta, stash, levels, slug, analytics)
    for pid, p in featured.items():
        page = player_page(meta, p, ppg, pos_rank[pid], ppg[pid] - levels[p["position"]], weights, analytics)
        pages[f"players/{slug[pid]}/index.html"] = page.replace("{slug}", slug[pid])
    # One page per player for the week ahead, which is how people search in
    # season, plus a hub that answers the position-wide version of it.
    week = meta["fromWeek"]
    calls = start_sit_table(players, weights, week, meta.get("dudOdds"))
    start_sit_slugs = []
    for pos, rows in calls.items():
        by_id = {r["player"]["id"]: i for i, r in enumerate(rows)}
        for pid in featured:
            i = by_id.get(pid)
            if i is None or players_by_pos.get(pid) != pos:
                continue
            near = [r for r in rows[max(0, i - 2):i + 3] if r["player"]["id"] != pid]
            page = start_sit_page(meta, rows[i], near, slug, analytics)
            pages[f"start-sit/{slug[pid]}/index.html"] = page.replace("{slug}", slug[pid])
            start_sit_slugs.append(slug[pid])
    pages["start-sit/index.html"] = start_sit_hub(meta, calls, set(featured), slug, analytics)

    # Where the model parts with the experts, and how its calls have gone.
    labels = meta.get("driverLabels", {}) | {"missed_games": "Missed games"}
    pages["model-vs-experts/index.html"] = experts_page(meta, disagree.disagreements(payload), slug,
                                                        labels, analytics)
    pages["scorecard/index.html"] = scorecard_page(meta, _read_json(out / "data/scorecard.json"),
                                                   _read_json(out / "data/track-record.json"), slug, analytics)

    # Last week's player pages go, so a player who drops out of the rankings
    # does not leave a stale page behind.
    shutil.rmtree(out / "players", ignore_errors=True)
    shutil.rmtree(out / "start-sit", ignore_errors=True)
    for rel, text in pages.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    # The compare tool links each player who has a page to it.
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data/player-pages.json").write_text(json.dumps(slug, separators=(",", ":"), sort_keys=True))

    # A waiver page from an earlier build stays listed if this week's adds
    # could not be fetched, so the sitemap never drops a live page.
    listed = (["/", "/waiver-wire/", "/rankings/"] + [f"/rankings/{p.lower()}/" for p in ranked_positions]
              + ["/ir-stash/", "/start-sit/", "/model-vs-experts/", "/scorecard/"]
              + [f"/start-sit/{s}/" for s in sorted(start_sit_slugs)]
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
