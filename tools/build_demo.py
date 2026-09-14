"""Bundle the site and a trimmed slice of projections into one HTML file.

Useful for sharing a working preview without hosting anything: the result
opens straight from a file system with no server and no network.

    python tools/build_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

WEB = Path("web")
# Roughly the pool a waiver page would realistically surface, rather than
# every rostered player, so the file stays small enough to email.
POOL = {"QB": 22, "RB": 42, "WR": 56, "TE": 26}


def main() -> None:
    data = json.loads((WEB / "data" / "projections.json").read_text())

    by_pos: dict[str, list] = {}
    for player in data["players"]:
        by_pos.setdefault(player["position"], []).append(player)

    keep = []
    for pos, limit in POOL.items():
        ranked = sorted(by_pos.get(pos, []),
                        key=lambda p: -(p["stats"].get("snapShare") or 0))
        keep.extend(ranked[:limit])

    payload = json.dumps({"meta": data["meta"], "players": keep},
                         separators=(",", ":"))

    html = (WEB / "index.html").read_text()
    css = (WEB / "styles.css").read_text()
    js = (WEB / "app.js").read_text()

    js = js.replace(
        """    const res = await fetch('data/projections.json');
    if (!res.ok) throw new Error(res.statusText);
    state.data = await res.json();""",
        "    state.data = JSON.parse(document.getElementById('payload').textContent);",
    )
    html = html.replace('<link rel="stylesheet" href="styles.css">',
                        f"<style>\n{css}\n</style>")
    html = html.replace(
        '<script src="app.js"></script>',
        f'<script type="application/json" id="payload">{payload}</script>\n'
        f"<script>\n{js}\n</script>",
    )
    html = html.replace("<title>Waiver — who should you pick up?</title>",
                        "<title>Waiver — demo</title>")

    out = WEB / "demo.html"
    out.write_text(html)
    print(f"Wrote {out} — {len(keep)} players, {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
