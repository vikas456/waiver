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

The first run downloads roughly 2 GB of play-by-play across five seasons and
takes 15 to 40 minutes depending on your connection. Everything is cached to
`data/cache`, so later runs take two or three minutes.

`--from-week` should be the next week to be played. If it is Tuesday of week
11, pass 11. `python -m pipeline.current_week` works it out from the calendar.

What gets pulled, all free and all from nflverse:

| Source | What it gives |
|---|---|
| `import_pbp_data` | play-by-play, the basis for every usage share |
| `import_weekly_data` | stat lines and the scoring components |
| `import_snap_counts` | snap share, the earliest signal of a role change |
| `import_ftn_data` | charting data, 2022 onward |
| `import_schedules` | spreads and totals for game script |
| `import_seasonal_rosters` | age, draft capital, experience for the priors |

## 3. Check the accuracy claims yourself

Do this before you trust a single ranking. It retrains the model once per
week, predicts the following four weeks, and scores those predictions against
what actually happened.

```bash
python -m pipeline.backtest --source nflverse --season 2025 --start-week 6
```

You get pairwise ranking accuracy and Spearman correlation for the model and
for three baselines: trailing four-game average, season average, and raw
volume. The trailing average is a harder baseline than it looks, because it is
roughly what a sharp person does in their head.

**If the model does not beat all three, it is not better, and the honest move
is to say so and keep working rather than ship it.** I could not run this on
real data from my environment, so these numbers are yours to generate. Expect
roughly 3 to 8 points of pairwise accuracy over the trailing average if the
approach is working as intended. If you see less than 2, something is wrong —
start by checking that `--from-week` is right and that the cache is current.

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
Sunday runs mostly exist to catch injury designations. If you want to reflect
an injury immediately, edit `playProb` for that player in
`web/data/projections.json` and redeploy — setting it to 0 removes him from
consideration cleanly, since the simulation already handles missed games.

## 6. Things worth doing next

- **Injury designations are currently assumed healthy.** Wiring in a live
  status feed is the highest-value single upgrade, because an out designation
  matters more than any model refinement.
- **Market anchoring is stubbed.** `train.blend_with_market` is written and
  tested but not fed, because free ADP data needs a source decision. Blending
  the model 75/25 with consensus typically beats either alone, so this is the
  second thing to do.
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
web/
  index.html, styles.css, app.js, data/projections.json
```
