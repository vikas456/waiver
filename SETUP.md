# Setup

## 1. Run it locally on synthetic data first

This takes about a minute and proves the whole pipeline works before you wait
on four seasons of downloads.

```bash
pip install -r requirements.txt
python -m pipeline.project --source synthetic --from-week 11
cd web && python -m http.server 8000
```

Open `http://localhost:8000`. The players are invented, but every mechanism —
week ranges, scoring formats, value over replacement, the driver bars, the
reasoning — is the real thing running on real code.

## 2. Switch to real NFL data

One flag. No code changes.

```bash
python -m pipeline.project --source nflverse --from-week 11
```

The first run downloads about 130 MB, most of it play-by-play, and takes a few
minutes. Completed seasons are cached to `data/cache`, one file per season, so
later runs only fetch the season in progress, which is never cached because
it changes every week.

If a load fails or the numbers look wrong, check each loader on its own:

```bash
python -m pipeline.doctor                  # the last completed season
python -m pipeline.doctor --season 2026    # the season in progress
```

It reports which loader broke, with its traceback, and also catches the quiet
failures: missing columns, empty ones, duplicated rows, and scoring that
disagrees with nflverse's own PPR total.

`--from-week` should be the next week to be played. If it is Tuesday of week
11, pass 11. `python -m pipeline.current_week` works it out from the calendar.
From week 3, form comes from the current season alone. Weeks 1 and 2 start
each player from his last game of an earlier season, blend in any games
already played this season, and keep every player still with a team.

What gets pulled, all free and all from nflverse via
[nflreadpy](https://github.com/nflverse/nflreadpy):

| Source | What it gives |
|---|---|
| `load_pbp` | play-by-play, the basis for every usage share |
| `load_player_stats` | stat lines and the scoring components |
| `load_snap_counts` | snap share, the earliest signal of a role change |
| `load_schedules` | spreads and totals for game script |
| `load_rosters` | age, draft capital, experience for the priors |
| `load_players` | the id map that joins snap counts to stat lines |

## 3. Check the accuracy claims yourself

Do this before you trust a single ranking. For every week of the season it
retrains the model on what had been played by then, runs the same forecast the
site runs for the following four weeks, and scores it against what actually
happened.

```bash
python -m pipeline.backtest --source nflverse --season 2025
```

You get pairwise ranking accuracy and Spearman correlation for the model, for
blends of the model with FantasyPros' consensus, and for four baselines:
trailing four-game average, season average, raw volume, and the consensus
itself. The trailing average is a harder baseline than it looks, because it is
roughly what a sharp person does in their head; the consensus is harder still.
Weeks 1 and 2 are reported separately, since the model has least to go on there.
Add `--out results.csv` to keep the per-week, per-position scores.

**If the model does not beat the three simple baselines, it is not better than
what already exists, and the honest move is to say so and keep working rather
than ship it.** The blend the site ships should also at least match the
consensus. Expect roughly 3 to 8 points of pairwise accuracy over the trailing average if the
approach is working as intended. If you see less than 2, something is wrong —
start by checking that `--from-week` is right and that the cache is current.

Run in September 2026, the share of player pairs put in the right order was:

| Season | Weeks | Model | What the site ships | FantasyPros | Last four games | Season average |
|---|---|---|---|---|---|---|
| 2025 | 1–2 | 0.832 | 0.855 | 0.856 | 0.772 | 0.775 |
| 2025 | 3 on | 0.853 | 0.857 | 0.816 | 0.804 | 0.813 |
| 2024 | 1–2 | 0.800 | 0.826 | 0.829 | 0.786 | 0.780 |
| 2024 | 3 on | 0.850 | 0.854 | 0.835 | 0.809 | 0.817 |

The site ships an even blend with the consensus in weeks 1 and 2 and a 25%
blend after. Those weights were set on 2025; 2024 was held back as an
untouched check, and the edge over the consensus from week 3 was smaller
there. Quarterbacks remain the weakest position.

## 4. Deploy

The site is static. The model runs once a week in CI and commits a JSON file.
There is no server, no database and no hosting bill.

1. Push to GitHub.
2. Enable GitHub Pages, or connect the repo to Cloudflare Pages or Netlify,
   with `web` as the publish directory and no build command.
3. The workflow in `.github/workflows/weekly.yml` runs Tuesday, Friday and
   Sunday, rebuilds projections and commits them. Pages redeploys on push.

GitHub Actions gives 2,000 free minutes a month. This uses about 40.

To run it by hand: Actions tab, "Weekly projections", Run workflow, enter the
week.

## 5. Weekly rhythm

Tuesday is the real rebuild, after Monday night is in the data. Friday and
Sunday runs mostly exist to catch injury designations and moves to injured
reserve, which Sleeper reports within hours. To reflect news before the next
scheduled run, run the workflow by hand; an edit to
`web/data/projections.json` would be overwritten by the next build. Each
week's `p` there is his chance of playing, and the site counts a missed game
as a zero.

## 6. Things worth doing next

- **Return dates are base rates.** A player on injured reserve gets the return
  odds of past players with the same injury (hamstring, ankle, knee, or all
  others), re-measured with `python tools/return_curves.py`. Reported
  timelines ("out four to six weeks") are not read, and only the quarterback
  depth chart is used; either would sharpen availability.
- **Rookies with no snaps cannot be projected.** A prior built from draft
  capital and landing spot would let the site rank them in week one.
- **Coverage and cornerback matchups** are the thinnest part of the free data.
  The model approximates them from play-by-play. A paid feed would upgrade
  this, and it is the only place where paying for data would clearly pay off.

## Layout

```
pipeline/
  config.py       scoring formats, replacement levels, stabilisation rates
  ingest.py       nflverse loading and caching
  synth.py        offline stand-in data with the same shape
  features.py     shrinkage, rolling windows, change points, opponent adjustment
  train.py        per-position component models and the pairwise ranker
  simulate.py     Monte Carlo, injury hazard, value over replacement
  explain.py      SHAP grouping and the comparative reasoning
  project.py      builds web/data/projections.json
  backtest.py     walk-forward validation against baselines
  doctor.py       checks each nflverse loader against one season
  market.py       FantasyPros consensus ranks, and blending with them
  availability.py depth charts, injury reports and reserve lists, per week
  sleeper.py      Sleeper's injury statuses and trending adds
  pages.py        ranking and waiver pages for search engines
web/
  index.html, styles.css, app.js, data/projections.json
```
