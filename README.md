# Waiver

Enter the free agents available in your league. Get a ranked answer for the
rest of the season, and the reasoning behind it.

## How the model works

**It predicts stat lines, not points.** Receptions, yards, carries and
touchdowns are each modelled separately, then scored afterwards. One set of
projections serves full PPR, half PPR, standard and tight end premium without
retraining, and the explanation can say something useful: the targets are
real, the touchdown rate is not.

**It measures role, not production.** Target share, air yards share, WOPR,
route participation, snap share, goal-line touch share and carry share carry
most of the weight. These lead production; yards and touchdowns lag it.

**It discounts luck.** Every player gets an expected touchdown total from his
goal-line and red-zone opportunities. The gap between that and his actual
touchdowns is the most common reason a hot free agent is a trap, and the model
treats it as noise rather than skill.

**It knows how fast each metric stabilises.** Target share firms up in about
four games. Touchdown rate essentially never stabilises inside a season. Each
feature is shrunk toward a prior in proportion to how much signal it actually
carries at that sample size, so two big games do not turn a bench player into
a must-add.

**It treats usage as zero-sum.** Targets and carries are modelled as a
composition within each team, so when a starter is out his vacated work is
redistributed rather than vanishing. This is exactly the situation a waiver
tool faces most often.

**It notices when a role changes.** Structural breaks — a new starting
quarterback, a fired coordinator, a committee back taking over — are detected
and the weeks since are weighted more heavily than the ones before.

**It adjusts per week, not per season.** Each remaining game is projected
against that specific opponent's profile, normalised for pace. A player can be
the right pick for the rest of the season and the wrong one for weeks 15 to
17, and the site says so when that happens.

**It simulates instead of guessing a number.** Each remaining week is
simulated thousands of times, with an availability model layered on top, so
the floor and ceiling shown are calibrated rather than decorative.

**It compares across positions honestly.** Rankings are driven by value over
replacement, computed from your league size. A tight end at 11 points can
outrank a receiver at 12.5, because the tight end you would otherwise start is
far worse than the receiver you would otherwise start.

## How the explanation works

The driver bars come from SHAP values on the fitted models, converted into
fantasy points and grouped into factors a person can reason about. They are
re-centred against the other players in that specific comparison, so each bar
reads as "this is what separates him from the others you asked about" rather
than "this is how he compares to a league average you never asked about".

The prose is generated from the same numbers. There is no separate narrative
layer that could drift from what the model actually did.

## What it does not do

It has no access to paid grading data, so coverage scheme and cornerback
matchups are approximated from play-by-play rather than charted directly.
Injury designations default to healthy until a status feed is wired in. And it
cannot tell you what a coach is thinking on Wednesday. It measures what has
already shown up in usage, which is earlier than the box score but not earlier
than the beat writer.

## Getting started

See [SETUP.md](SETUP.md). It runs on synthetic data in about a minute, then
switches to real nflverse data with one flag.

> The `web/data/projections.json` committed here is **synthetic**. The players
> are invented and the numbers mean nothing about football. The site labels
> itself as demo data whenever `meta.source` is `synthetic`, so this cannot be
> mistaken for the real thing by accident. Replace it by running
> `python -m pipeline.project --source nflverse --from-week <week>`.

## Repository

| Path | What it is |
|---|---|
| `pipeline/` | data, features, models, simulation, explanation |
| `web/` | the static site, deployed as-is |
| `tools/build_demo.py` | bundles the site and data into one shareable HTML file |
| `.github/workflows/weekly.yml` | rebuilds projections Tuesday, Friday and Sunday |
| `.github/workflows/pages.yml` | deploys `web/` to GitHub Pages on push |
| `.github/workflows/checks.yml` | runs the synthetic pipeline on every push |

Downloaded nflverse data and trained models are gitignored. Both are
rebuildable from one command, and neither belongs in history.

## Validating it

`python -m pipeline.backtest` retrains weekly, predicts forward, and scores
against the trailing four-game average, the season average and raw volume.
Run it before trusting a ranking. If it does not beat all three baselines out
of sample, it is not better than what already exists.

## Data

All data comes from [nflverse](https://github.com/nflverse), which is free and
open.
