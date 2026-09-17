"""Tell Bing a week's pages have changed, instead of waiting to be crawled.

IndexNow is a ping: the site says which addresses changed, and Bing, Yandex
and Seznam fetch them when they get to it. Google does not take part, so this
speeds up one half of search and does nothing for the other.

It matters here because the pages that earn traffic are the ones published on
a Tuesday and searched for on a Thursday. Waiting a week to be crawled wastes
most of that.

The key is not a secret. It is published at the site root, which is how the
protocol proves the ping came from someone who can write to the site.

    python -m tools.indexnow --dry-run      # what would be sent
    python -m tools.indexnow                # send it
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from pipeline.config import INDEXNOW_KEY

ENDPOINT = "https://api.indexnow.org/IndexNow"
HOST = "fantasywaiverpicks.com"
SITE = f"https://{HOST}"
WEB = Path("web")

# The pages that genuinely change every build. Player pages change too, but
# there are nearly three hundred of them and they say much the same thing week
# to week; a ping is a claim that something is worth re-reading.
def changed(week: int, reports: list[int]) -> list[str]:
    paths = [
        "/", "/waiver-wire/", f"/waiver-wire/week-{week}/",
        "/start-sit/", f"/start-sit/week-{week}/",
        "/rankings/", "/rankings/qb/", "/rankings/rb/", "/rankings/wr/",
        "/rankings/te/", "/rankings/k/", "/rankings/def/",
        "/ir-stash/", "/scorecard/", "/model-vs-experts/", "/players/", "/reports/",
    ]
    paths += [f"/reports/week-{n}/" for n in sorted(reports)]
    # Only addresses that exist: a ping for a page that is not there yet is
    # worse than no ping at all.
    live = []
    for path in paths:
        target = WEB / path.strip("/") / "index.html" if path != "/" else WEB / "index.html"
        if target.exists():
            live.append(SITE + path)
    return live


def key_file() -> Path:
    """Where the key has to be readable, per the protocol."""
    return WEB / f"{INDEXNOW_KEY}.txt"


def write_key_file() -> Path:
    path = key_file()
    path.write_text(INDEXNOW_KEY + "\n")
    return path


def submit(urls: list[str], timeout: int = 30) -> int:
    """Send the list. Anything in the 200s means accepted; the crawl follows
    in its own time."""
    body = json.dumps({
        "host": HOST,
        "key": INDEXNOW_KEY,
        "keyLocation": f"{SITE}/{INDEXNOW_KEY}.txt",
        "urlList": urls,
    }).encode()
    request = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as res:
            return res.status
    except urllib.error.HTTPError as err:
        # 422 means the key or the host did not line up; worth seeing in the log.
        print(f"  IndexNow refused the ping: {err.code} {err.reason}")
        return err.code
    except Exception as err:
        print(f"  IndexNow unreachable ({type(err).__name__}); the pages will be crawled anyway")
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Ping IndexNow with this week's pages")
    ap.add_argument("--week", type=int, help="defaults to the week the payload was built for")
    ap.add_argument("--dry-run", action="store_true", help="print the list and send nothing")
    args = ap.parse_args()

    meta = json.loads((WEB / "data/projections.json").read_text())["meta"]
    week = args.week or int(meta["fromWeek"])
    reports = [int(p.stem.split("-")[1]) for p in (WEB / "data/reports").glob("week-*.json")]
    urls = changed(week, reports)

    path = write_key_file()
    if args.dry_run:
        print(f"key file: {path}")
        print(f"would send {len(urls)} urls:")
        for url in urls:
            print(f"  {url}")
        return

    status = submit(urls)
    print(f"IndexNow: {len(urls)} urls, HTTP {status}"
          + ("" if 200 <= status < 300 else " (not accepted)"))


if __name__ == "__main__":
    main()
