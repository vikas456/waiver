/* Waiver — client-side ranking engine.
 *
 * The Python pipeline writes one static file of per-week stat-line
 * predictions. Everything a person can change in the interface — scoring
 * format, week range, league size — is applied here, in the
 * browser, against those same numbers. That keeps the site free to host and
 * makes every control instant.
 */

'use strict';

const STATS = ['pass_yd', 'pass_td', 'interception', 'rush_yd', 'rush_td',
  'reception', 'rec_yd', 'rec_td', 'fumble_lost', 'two_point'];

const REPLACEMENT_PER_TEAM = { QB: 1.5, RB: 2.5, WR: 3.0, TE: 1.2 };

const state = {
  data: null,
  pages: {},
  picked: [],
  fromWeek: null,
  toWeek: null,
  preset: 'rest',
  format: 'ppr',
  leagueSize: 12,
  tePremium: 0,
  // Replacement depth per team at each position; a connected league's lineup
  // moves it (see leagueReplacement).
  replacement: { ...REPLACEMENT_PER_TEAM },
  // A connected Sleeper league, or null (see applyLeague).
  league: null,
};

/* -- Scoring ------------------------------------------------------------- */

function scoreLine(components, weights, position, tePremium) {
  let pts = 0;
  for (const stat of STATS) pts += (components[stat] || 0) * (weights[stat] || 0);
  if (tePremium && position === 'TE') pts += (components.reception || 0) * tePremium;
  return pts;
}

/* -- Random draws -------------------------------------------------------- */

let spare = null;
function normal() {
  if (spare !== null) { const v = spare; spare = null; return v; }
  let u, v, s;
  do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; s = u * u + v * v; }
  while (s === 0 || s >= 1);
  const f = Math.sqrt(-2 * Math.log(s) / s);
  spare = v * f;
  return u * f;
}

// Marsaglia and Tsang. Fantasy outcomes are right-skewed and bounded below at
// roughly zero, so a gamma fits them far better than a normal, which would
// hand back negative point totals and a symmetric ceiling.
function gamma(shape, scale) {
  if (shape < 1) {
    return gamma(shape + 1, scale) * Math.pow(Math.random() || 1e-12, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = normal(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v * scale;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
  }
}

/* -- Projection ---------------------------------------------------------- */

function weeksInRange(player, from, to) {
  return player.weeks.filter(w => w.w >= from && w.w <= to);
}

// Chance of playing in one week, which the depth chart and injury report can
// lower. Older data files carry a single figure per player.
const baseChance = (player, wk) => wk.p ?? player.playProb;

// A single week is a start-or-sit decision: raw points for that week, rather
// than value over replacement for a stretch of them, with a floor and a
// ceiling beside the projection.
const startSit = () => state.fromWeek === state.toWeek;

// Lineups are set after the injury report, so for a single week a player with
// no designation is taken to play, unless the depth chart says he will not.
// The general chance of an absence nobody has reported yet only matters over
// a stretch of weeks.
function playChance(player, wk) {
  const p = baseChance(player, wk);
  if (!startSit() || player.injury) return p;
  return Math.min(1, p / (player.playProb || 1));
}

// Fast analytic mean, used to rank the whole player pool so replacement level
// has something to be measured against. Simulation is reserved for the few
// players actually being compared.
function meanPpg(player, from, to) {
  const weights = state.data.meta.scoringFormats[state.format];
  const weeks = weeksInRange(player, from, to);
  if (!weeks.length) return 0;
  let total = 0;
  for (const wk of weeks) {
    total += scoreLine(wk.c, weights, player.position, state.tePremium) * playChance(player, wk);
  }
  return total / weeks.length;
}

function simulate(player, from, to, draws = 3000) {
  const weights = state.data.meta.scoringFormats[state.format];
  const weeks = weeksInRange(player, from, to);
  if (!weeks.length) {
    return { ppg: 0, p10: 0, p90: 0, total: 0, games: 0, weeks: 0, draws: null };
  }
  // For one week the floor and ceiling describe the game if he plays; his
  // chance of playing is shown beside them and counted in the projection.
  // The draws are kept, unsorted, so two players can be set against each other.
  const single = from === to;
  if (single) draws = 6000;

  const means = weeks.map(wk =>
    Math.max(scoreLine(wk.c, weights, player.position, state.tePremium), 0.05));
  // Spread was calibrated in PPR, so it is rescaled when the format changes
  // the size of the numbers it is describing.
  const pprMeans = weeks.map(wk =>
    Math.max(scoreLine(wk.c, state.data.meta.scoringFormats.ppr, player.position, 0), 0.05));
  const sds = weeks.map((wk, i) => Math.max(wk.sd * (means[i] / pprMeans[i]), 0.5));

  const perGame = new Float64Array(draws);
  const totals = new Float64Array(draws);
  let gameSum = 0;

  for (let d = 0; d < draws; d++) {
    let total = 0, played = 0;
    for (let i = 0; i < weeks.length; i++) {
      if (!single && Math.random() >= playChance(player, weeks[i])) continue;
      const shape = (means[i] / sds[i]) ** 2;
      const scale = (sds[i] * sds[i]) / means[i];
      total += gamma(shape, scale);
      played++;
    }
    totals[d] = total;
    // Averaged over every scheduled game, so a missed game counts as the zero
    // it is in a lineup. Averaging only games played let a backup who rarely
    // plays look like a starter.
    perGame[d] = total / weeks.length;
    gameSum += played;
  }

  const sorted = Float64Array.from(perGame).sort();
  const at = q => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  let totalSum = 0;
  for (let d = 0; d < draws; d++) totalSum += totals[d];

  // Tested on single games in 2024 and 2025: a player projected for under 7
  // PPR points when he plays scores next to nothing in more than one game in
  // ten, which the gamma curve misses, so his floor is zero. From 7 points up,
  // about one game in ten fell below the floor and one in ten above the ceiling.
  const dud = single && pprMeans[0] < 7;
  return {
    ppg: sorted.reduce((a, b) => a + b, 0) / draws,
    p10: dud ? 0 : at(0.10),
    p90: at(0.90),
    total: totalSum / draws,
    games: gameSum / draws,
    weeks: weeks.length,
    draws: single ? perGame : null,
    chance: single ? playChance(player, weeks[0]) : null,
  };
}

// How often one player outscores another in a single week: both play and he
// scores more, or only he plays. A tie, such as both sitting out, counts half.
function beats(a, b) {
  if (!a.draws) return b.draws ? 0 : 0.5;
  if (!b.draws) return 1;
  let more = 0;
  const n = Math.min(a.draws.length, b.draws.length);
  for (let i = 0; i < n; i++) more += a.draws[i] > b.draws[i] ? 1 : 0;
  const both = more / n;
  const [pa, pb] = [a.chance, b.chance];
  return pa * pb * both + pa * (1 - pb) + 0.5 * (1 - pa) * (1 - pb);
}

// Draws are random, so without this every player's numbers would shift a
// little each time the list re-ranks as someone is added or removed.
const simCache = new Map();
function simulateCached(player, from, to) {
  const key = `${player.id}|${from}|${to}|${state.format}|${state.tePremium}`;
  if (!simCache.has(key)) simCache.set(key, simulate(player, from, to));
  return simCache.get(key);
}

/* -- Value over replacement ---------------------------------------------- */

function replacementLevels(from, to) {
  const byPos = {};
  for (const p of state.data.players) {
    (byPos[p.position] ||= []).push(meanPpg(p, from, to));
  }
  const levels = {};
  for (const [pos, values] of Object.entries(byPos)) {
    values.sort((a, b) => b - a);
    const rank = Math.max(1, Math.round((state.replacement[pos] || 2) * state.leagueSize));
    // Average a small band around the cut line so one outlier cannot move the
    // baseline for an entire position.
    const lo = Math.max(0, rank - 3);
    const hi = Math.min(values.length, rank + 2);
    const band = values.slice(lo, hi);
    levels[pos] = band.length
      ? band.reduce((a, b) => a + b, 0) / band.length
      : (values[values.length - 1] || 0);
  }
  return levels;
}

/* -- Drivers ------------------------------------------------------------- */

function driverTotals(player, from, to, scaleToFormat) {
  const weeks = weeksInRange(player, from, to);
  const out = {};
  for (const wk of weeks) {
    for (const [group, value] of Object.entries(wk.drivers || {})) {
      out[group] = (out[group] || 0) + value;
    }
  }
  const n = weeks.length || 1;
  for (const key of Object.keys(out)) out[key] = (out[key] / n) * scaleToFormat;
  return out;
}

// Each player is explained against one other: the top pick against the
// runner-up, everyone else against the top pick. A player's factors add up to
// his value over replacement, so the differences add up to the gap between
// the two, and the reasons given always point the way the ranking does.
function versus(row, other) {
  const rel = {};
  for (const g of new Set([...Object.keys(row.factors), ...Object.keys(other.factors)])) {
    rel[g] = (row.factors[g] || 0) - (other.factors[g] || 0);
  }
  // Between two players at one position the scarcity term holds only
  // scoring-format rounding, which is not a reason anyone would recognise.
  if (row.player.position === other.player.position) delete rel.position;
  return rel;
}

/* -- Ranking ------------------------------------------------------------- */

function rank(players, from, to) {
  // A start-or-sit call is about points in one lineup slot, so for a single
  // week nobody is measured against a replacement player.
  const single = from === to;
  const levels = single ? {} : replacementLevels(from, to);
  const pprWeights = state.data.meta.scoringFormats.ppr;
  const fmtWeights = state.data.meta.scoringFormats[state.format];

  const rows = players.map(p => {
    const sim = simulateCached(p, from, to);
    const weeks = weeksInRange(p, from, to);
    const n = weeks.length || 1;
    const pts = weeks.map(w => scoreLine(w.c, fmtWeights, p.position, state.tePremium));
    const pprMean = weeks.reduce((a, w) => a + scoreLine(w.c, pprWeights, p.position, 0), 0) / n;
    // Points per game if he plays every game, and the exact expectation once
    // the chance of missing each one is counted. Ranking on the expectation
    // rather than the simulation's estimate of it keeps a near tie from
    // flipping on random draws; the simulation still supplies the range.
    const ifPlays = pts.reduce((a, b) => a + b, 0) / n;
    const expected = weeks.reduce((a, w, i) => a + pts[i] * playChance(p, w), 0) / n;
    const scale = pprMean > 0 ? ifPlays / pprMean : 1;
    const replacement = levels[p.position] || 0;

    // Every point of his value over replacement belongs to a factor: what the
    // model sees in his play, the games he is expected to miss, and his
    // position's baseline against its replacement level.
    const drivers = driverTotals(p, from, to, scale);
    const driven = Object.values(drivers).reduce((a, b) => a + b, 0);
    const factors = { ...drivers, missed_games: expected - ifPlays,
      position: ifPlays - driven - replacement };

    return { player: p, ...sim, ppg: expected, ifPlays, opp: weeks.length ? weeks[0].opp : null,
      replacement, vorp: expected - replacement, factors };
  });

  // Two players can share a surname, and "against Moreau and Moreau" is the
  // kind of sentence that makes a reader stop trusting everything above it.
  const surnames = rows.map(r => lastName(r.player.name));
  rows.forEach((r, i) => {
    const clash = surnames.filter(n => n === surnames[i]).length > 1;
    r.displayName = clash ? r.player.name : surnames[i];
  });
  rows.sort((a, b) => b.vorp - a.vorp);
  rows.forEach((r, i) => {
    r.against = rows.length > 1 ? (i === 0 ? rows[1] : rows[0]) : null;
    r.rel = r.against ? versus(r, r.against) : {};
  });
  return rows;
}

/* -- Reasoning ----------------------------------------------------------- */

// Opportunity means different things by position: a back is judged on touches,
// a receiver on targets. Saying "targets" about a running back is the kind of
// small wrongness that makes a whole explanation look automated.
// Every reason is given against one other player, so the wording compares
// rather than grading him on his own.
const OPPORTUNITY_PHRASES = {
  RB: ['takes a bigger share of his backfield\u2019s touches',
    'takes a smaller share of his backfield\u2019s touches'],
  QB: ['carries more of his offence\u2019s volume', 'carries less of his offence\u2019s volume'],
};
// A quarterback does not run routes; for him this group comes down to snaps.
const ROLE_PHRASES = {
  QB: ['has been on the field for more of his team\u2019s snaps',
    'has been on the field for fewer of his team\u2019s snaps'],
};

const PHRASES = {
  opportunity: ['commands a bigger share of his offence\u2019s targets',
    'sees a smaller share of his offence\u2019s targets'],
  route_role: ['is on the field and running routes more often',
    'runs routes on fewer of his team\u2019s dropbacks'],
  goal_line: ['gets more of the work near the goal line', 'gets less of the work near the goal line'],
  efficiency: ['gets more out of each opportunity', 'gets less out of each opportunity'],
  td_regression: ['has scored closer to what his usage supports',
    'has scored more touchdowns than his usage supports, which tends to even out'],
  trend: ['has been gaining role over the past few weeks',
    'has been losing role over the past few weeks'],
  production: ['has been putting up bigger stat lines', 'has been putting up smaller stat lines'],
  market: ['is rated higher by expert consensus', 'is rated lower by expert consensus'],
  offense: ['plays in a more productive offence', 'plays in a less productive offence'],
  schedule: ['draws a friendlier set of remaining defences',
    'faces a harder set of remaining defences'],
  sample: ['has a longer track record to judge from', 'has a thinner sample to judge from'],
  prior: ['profiles better for his role', 'profiles worse for his role'],
  position: ['plays a position where a player like him is harder to replace',
    'plays a position where a player like him is easier to replace'],
};
// Older data files call the sample-size group "availability".
PHRASES.availability = PHRASES.sample;

// Names for the factors the data file's labels do not cover, or misname.
const DRIVER_NAMES = {
  missed_games: 'Missed games', position: 'Position scarcity',
  sample: 'Sample size', availability: 'Sample size',
};

// Ends a sentence on a name without doubling the full stop of a Jr. or Sr.
const stop = name => (name.endsWith('.') ? name : `${name}.`);

const ORDINAL = ['', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth'];

// Real names carry suffixes, and "Jr. ranks first" is not a sentence.
function lastName(name) {
  const bits = name.split(' ').filter(b => !/^(jr|sr|ii|iii|iv|v)\.?$/i.test(b));
  return bits[bits.length - 1] || name;
}

// For a single week, the position factor is the gap between what the two
// positions typically score rather than scarcity, and the schedule is one
// opponent.
const WEEK_PHRASES = {
  position: ['plays a higher-scoring position', 'plays a lower-scoring position'],
  schedule: ['has the easier matchup this week', 'has the harder matchup this week'],
};

// Missed games are described by why he is expected to miss them.
function missedPhrase(row, positive) {
  if (positive) {
    return startSit() ? 'is more likely to play this week'
      : 'is expected to be on the field for more of these games';
  }
  const inj = row.player.injury;
  if (inj && (inj.status === 'IR' || inj.status === 'PUP')) {
    return `is on ${inj.status === 'IR' ? 'injured reserve' : 'the PUP list'} and expected to miss games`;
  }
  if (inj) return `is listed as ${inj.status.toLowerCase()} and may ${startSit() ? 'sit out this week' : 'miss time'}`;
  const weeks = weeksInRange(row.player, state.fromWeek, state.toWeek);
  const chance = weeks.reduce((a, w) => a + playChance(row.player, w), 0) / (weeks.length || 1);
  return chance < 0.3 ? 'is not expected to start, so he rarely plays'
    : 'is less likely to be on the field each week';
}

function phraseFor(group, row, positive) {
  if (group === 'missed_games') return missedPhrase(row, positive);
  if (startSit() && WEEK_PHRASES[group]) return WEEK_PHRASES[group][positive ? 0 : 1];
  const pos = row.player.position;
  const pair = (group === 'opportunity' && OPPORTUNITY_PHRASES[pos])
    || (group === 'route_role' && ROLE_PHRASES[pos])
    || PHRASES[group] || ['rates better here', 'rates worse here'];
  return positive ? pair[0] : pair[1];
}

// Factors that clearly matter, strongest first. When nothing clears the bar
// but a reason must lead, the strongest one still does: the gap has to come
// from somewhere.
function strongest(entries, mustLead) {
  const clear = entries.filter(([, v]) => Math.abs(v) > 0.12);
  if (clear.length || !mustLead) return clear;
  return entries.filter(([, v]) => Math.abs(v) > 0.02).slice(0, 1);
}

function splitDrivers(rel) {
  const entries = Object.entries(rel);
  return {
    up: entries.filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]),
    down: entries.filter(([, v]) => v < 0).sort((a, b) => a[1] - b[1]),
  };
}

function reasoning(row, position) {
  const other = row.against;
  if (!other) return '';
  const name = row.displayName || lastName(row.player.name);
  const them = other.displayName || lastName(other.player.name);
  // With no game in the range his projection is zero, and no factor the model
  // weighs is the reason; say the real one.
  const idle = p => !weeksInRange(p, state.fromWeek, state.toWeek).length;
  const range = startSit() ? `in week ${state.fromWeek}` : 'in this range';
  if (idle(row.player)) return `${name} has no game ${range}.`;
  // Only the top pick is explained against a runner-up who might sit out;
  // anyone else trails the top pick, so the factors below say why.
  if (position === 1 && idle(other.player)) {
    return `${name} ranks first, ahead of ${them}, who has no game ${range}.`;
  }
  const { up, down } = splitDrivers(row.rel);
  const parts = [];
  const say = ([g, v]) => phraseFor(g, row, v > 0);
  // The reasons always point the way the ranking does: the top pick's case
  // is made from what puts him ahead, anyone else's from what keeps him
  // behind. Then the strongest counter-point, because a case that never
  // mentions the downside is not a case.
  const [forRank, againstRank] = position === 1 ? [up, down] : [down, up];
  const lead = strongest(forRank, true);
  const concede = strongest(againstRank, false);

  if (!lead.length) {
    parts.push(`${name} and ${them} finish level on every factor the model weighs, ` +
      'so the order between them could go either way.');
  } else {
    const where = position === 1 ? `first, ahead of ${them}`
      : `${ORDINAL[position] || `number ${position}`}, behind ${them}`;
    parts.push(`${name} ranks ${where}, mainly because he ${say(lead[0])}` +
      (lead[1] ? `, and ${say(lead[1])}.` : '.'));
    if (concede.length) {
      parts.push(position === 1
        ? `That holds even though he ${say(concede[0])}.`
        : `In his favour, he ${say(concede[0])}, but not by enough to close the gap.`);
    }
  }

  const s = row.player.stats;
  // The touchdown count only adds something when the reasons above did not
  // already make the same point.
  const said = new Set([...lead.slice(0, 2), ...concede.slice(0, 1)].map(([g]) => g));
  if (!said.has('td_regression') && s.actualTd != null && s.expectedTd != null) {
    const gap = s.actualTd - s.expectedTd;
    if (gap > 1.4) {
      parts.push(`He has ${Math.round(s.actualTd)} touchdowns against the ` +
        `${s.expectedTd.toFixed(1)} his opportunities support, the kind of gap that ` +
        `closes rather than continues.`);
    } else if (gap < -1.4) {
      parts.push('He has scored less than his opportunities deserve, so the ' +
        'correction ahead of him is more likely to be upward.');
    }
  }
  if (s.roleChange === 1) {
    parts.push('The model also detected a change in his role this season and ' +
      'weighted the weeks since more heavily than the ones before.');
  }
  return parts.join(' ');
}

// The answer leads, in one sentence. How firmly it is worded follows the size
// of the gap, so a near tie never reads as a sure thing.
function headline(rows) {
  if (startSit()) return startHeadline(rows);
  const [top, second] = rows;
  const gap = top.ppg - second.ppg;
  const vorpGap = top.vorp - second.vorp;
  let answer;

  if (vorpGap < 0.35) {
    answer = {
      tone: 'warn', label: 'Close call',
      title: `${top.player.name}, narrowly over ${stop(second.player.name)}`,
      sub: `They finish ${vorpGap.toFixed(1)} points of value apart. ` +
        'Take the one whose role you believe in.',
    };
  } else {
    answer = {
      tone: vorpGap < 1 ? 'neutral' : 'good',
      label: vorpGap < 1 ? 'Slight edge' : 'Clear pick',
      title: `Pick up ${stop(top.player.name)}`,
      // Fewer points but still first is the case value over replacement exists for.
      sub: gap < 0
        ? `He projects ${Math.abs(gap).toFixed(1)} fewer points a game than ${second.player.name}, ` +
          `but ${top.player.position} is the thinner position in your league, so he replaces a worse player.`
        : `About ${gap.toFixed(1)} more points a game than ${stop(second.player.name)}`,
    };
  }
  if (top.p90 - top.p10 > 10) {
    answer.sub += ' His range is wide, so take him if you need upside and the ' +
      'steadier option if you are protecting a lead.';
  }
  return answer;
}

// For one week: who to start, how often he outscores the next best, and when
// the floor or the ceiling argues for someone else.
function startHeadline(rows) {
  const [top, second] = rows;
  const week = `week ${state.fromWeek}`;
  if (!top.draws) {
    return { tone: 'warn', label: 'No game', title: `None of these players has a game in ${week}.`,
      sub: 'Pick another week, or add players who are playing.' };
  }
  let answer;
  if (!second.draws) {
    answer = { tone: 'good', label: 'Clear start', title: `Start ${stop(top.player.name)}`,
      sub: `${second.player.name} has no game in ${week}.` };
  } else {
    const odds = beats(top, second);
    const often = `${lastName(top.player.name)} outscores ${second.player.name} in ` +
      `${Math.round(odds * 100)}% of simulated ${week} games`;
    answer = odds < 0.55
      ? { tone: 'warn', label: 'Close call', title: `${top.player.name}, narrowly over ${stop(second.player.name)}`,
          sub: `${often}, close to a coin flip.` }
      : { tone: odds < 0.65 ? 'neutral' : 'good', label: odds < 0.65 ? 'Slight edge' : 'Clear start',
          title: `Start ${stop(top.player.name)}`,
          sub: `${often}, and projects ${(top.ppg - second.ppg).toFixed(1)} more points.` };
  }
  // The pick is the one expected to score most. A manager who needs a big
  // week, or only needs a steady one, may want the player at either end.
  const playing = rows.filter(r => r.draws);
  const ceiling = playing.reduce((a, r) => (r.p90 > a.p90 ? r : a));
  const floor = playing.reduce((a, r) => (r.p10 > a.p10 ? r : a));
  const extra = [];
  if (ceiling !== top && ceiling.p90 - top.p90 >= 1) {
    extra.push(`If you need a big week, ${ceiling.player.name} has the higher ceiling.`);
  }
  if (floor !== top && floor.p10 - top.p10 >= 1) {
    extra.push(`If you only need a steady week, ${floor.player.name} has the higher floor.`);
  }
  const inj = top.player.injury;
  if (inj && top.chance < 1) {
    extra.push(`${lastName(top.player.name)} is listed as ${inj.status.toLowerCase()}, so check he is active before kickoff.`);
  }
  if (extra.length) answer.sub += ` ${extra.join(' ')}`;
  return answer;
}

// A player can be right for the rest of the season and wrong for the fantasy
// playoffs alone, because the schedule adjustment is applied week by week.
// Saying so is something a fixed season-long ranking cannot do.
function flipNote(rows, from, to) {
  const meta = state.data.meta;
  const alt = { from: 15, to: Math.min(17, meta.throughWeek), label: 'weeks 15 to 17' };
  if (from >= 15 || to < 15 || alt.from > alt.to) return null;
  const altRows = rank(rows.map(r => r.player), alt.from, alt.to);
  if (!altRows.length || altRows[0].player.id === rows[0].player.id) return null;
  return `Over ${alt.label} alone, ${altRows[0].player.name} moves ahead of ` +
    `${rows[0].player.name} on schedule alone.`;
}

/* -- Rendering ----------------------------------------------------------- */

const $ = sel => document.querySelector(sel);
// Names come from the data file. Escaping them keeps a malformed name from
// ever being read as markup.
const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = v => (v == null ? '\u2014' : `${Math.round(v * 100)}%`);
const signed = v => `${v >= 0 ? '+' : '\u2212'}${Math.abs(v).toFixed(1)}`;

function confidencePill(row) {
  const spread = row.p90 - row.p10;
  if (row.player.stats.gamesPlayed < 4) return { cls: 'warn', text: 'Thin sample' };
  if (startSit()) return null;
  if (spread > 11) return { cls: 'warn', text: 'Volatile' };
  if (spread < 6.5) return { cls: 'good', text: 'Steady' };
  return null;
}

function renderDrivers(rel) {
  const labels = state.data.meta.driverLabels;
  const entries = Object.entries(rel)
    .filter(([, v]) => Math.abs(v) > 0.08)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 3);
  if (!entries.length) return '';
  const max = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.5);

  return '<div class="drivers">' + entries.map(([g, v]) => {
    const width = (Math.abs(v) / max) * 46;
    const bar = v > 0
      ? `<i style="left:50%;width:${width}%;background:var(--up)"></i>`
      : `<i style="right:50%;width:${width}%;background:var(--down)"></i>`;
    return `<div class="driver">
      <span>${esc(g === 'position' && startSit() ? 'Position' : DRIVER_NAMES[g] || labels[g] || g)}</span>
      <div class="track">${bar}</div>
      <b style="color:var(${v > 0 ? '--up' : '--down'})">${signed(v)}</b>
    </div>`;
  }).join('') + '</div>';
}

// The shorthand fantasy apps use. Mirrors SHORT_STATUS in pipeline/pages.py.
const SHORT_STATUS = {
  Questionable: 'Q', Doubtful: 'D', Out: 'O', Suspended: 'SUS',
  Inactive: 'NA', 'Did not report': 'DNR',
};

// What is keeping a player off the field, over the weeks being compared.
function injuryNote(player) {
  const inj = player.injury;
  const weeks = inj ? weeksInRange(player, state.fromWeek, state.toWeek) : [];
  if (!weeks.length) return '';
  const chance = wk => `${Math.round(playChance(player, wk) * 100)}%`;
  const what = { IR: 'On injured reserve', PUP: 'On the physically unable to perform list' }[inj.status]
    || `Listed as ${inj.status.toLowerCase()}`;
  const detail = inj.detail ? ` (${inj.detail.toLowerCase()})` : '';
  let out = 0;
  while (out < weeks.length && playChance(player, weeks[out]) === 0) out++;
  const rest = weeks.slice(out);
  let text;
  if (!rest.length) {
    text = `${what}${detail}, and out for every week in this range.`;
  } else if (out) {
    const last = rest[rest.length - 1];
    text = `${what}${detail}: out through week ${weeks[out - 1].w}, then a ${chance(rest[0])} ` +
      `chance he is back in week ${rest[0].w}` +
      (last !== rest[0] ? `, rising to ${chance(last)} by week ${last.w}.` : '.');
  } else {
    text = `${what}${detail}, so he has a ${chance(weeks[0])} chance of playing in week ${weeks[0].w}.`;
  }
  return `<p class="reason">${esc(text)} Games he misses count as zero.</p>`;
}

function renderStatLine(row) {
  const p = row.player;
  const s = p.stats;
  const pill = confidencePill(row);
  const items = pill ? [`<span class="pill ${pill.cls}">${pill.text}</span>`] : [];
  for (const mark of row.marks || []) items.push(`<span class="pill neutral">${mark}</span>`);
  if (startSit() && row.draws) items.push(`Floor ${row.p10.toFixed(1)}`, `Ceiling ${row.p90.toFixed(1)}`);
  items.push(`Snap share ${pct(s.snapShare)}`);
  if (p.position === 'RB') {
    items.push(`Carry share ${pct(s.carryShare)}`, `Goal-line share ${pct(s.glShare)}`);
  } else if (p.position !== 'QB') {
    items.push(`Target share ${pct(s.targetShare)}`, `Route rate ${pct(s.routeRate)}`);
  }
  if (s.actualTd != null && s.expectedTd != null) {
    items.push(`${Math.round(s.actualTd)} TD vs ${s.expectedTd.toFixed(1)} expected`);
  }
  return `<p class="statline">${items.map(i => `<span>${i}</span>`).join('')}</p>`;
}

function renderRanking(rows) {
  const compared = rows.length > 1;
  const single = startSit();
  // One scale for every bar, so ranges can be compared down the list.
  const scale = Math.max(...rows.map(r => r.p90), 1);
  const playing = rows.filter(r => r.draws);
  if (single && playing.length > 1) {
    const best = key => playing.reduce((a, r) => (r[key] > a[key] ? r : a));
    rows.forEach(r => { r.marks = []; });
    if (best('p10').p10 > 0) best('p10').marks.push('Highest floor');
    best('p90').marks.push('Highest ceiling');
  }
  $('#ranking').innerHTML = rows.map((row, i) => {
    const p = row.player;
    const tail = single
      ? (row.opp ? `vs ${esc(row.opp)}` : 'No game') +
        (row.draws && row.chance < 0.995 ? ` \u00b7 ${pct(row.chance)} to play` : '')
      : `<span class="${row.vorp >= 0 ? 'up' : 'down'}">${signed(row.vorp)}</span> over replacement`;
    const bar = single && row.draws
      ? `<span class="spread" role="img" aria-label="Floor ${row.p10.toFixed(1)}, ceiling ${row.p90.toFixed(1)} points">` +
        `<i style="left:${(row.p10 / scale) * 100}%;width:${((row.p90 - row.p10) / scale) * 100}%"></i>` +
        `<b style="left:${Math.min(row.ifPlays / scale, 1) * 100}%"></b></span>`
      : '';
    return `<li>
      <details class="row"${i === 0 ? ' open' : ''}>
        <summary class="row-head">
          <span class="rank">${i + 1}</span>
          ${avatar(p, 'lg')}
          <span class="row-main">
            <strong>${esc(p.name)}</strong>
            <span class="row-meta">${esc(p.position)} \u00b7 ${esc(p.team)} \u00b7
              ${p.injury ? `<span class="down" title="${esc(p.injury.status)}">${esc(SHORT_STATUS[p.injury.status] || p.injury.status)}</span> \u00b7` : ''}
              ${tail}</span>
            ${bar}
          </span>
          <span class="row-pts">
            <b>${row.ppg.toFixed(1)}</b>
            <span>${row.p10.toFixed(1)}\u2013${row.p90.toFixed(1)}</span>
          </span>
          <button class="remove" type="button" data-remove="${esc(p.id)}" aria-label="Remove ${esc(p.name)}">
            <svg viewBox="0 0 14 14" aria-hidden="true"><path d="M3 3 L11 11 M11 3 L3 11"/></svg>
          </button>
        </summary>
        <div class="row-body">
          ${compared ? renderDrivers(row.rel) : ''}
          ${renderStatLine(row)}
          ${injuryNote(p)}
          ${compared ? `<p class="reason">${esc(reasoning(row, i + 1))}</p>` : ''}
          ${state.pages[p.id] ? `<a class="outlook" href="/players/${esc(state.pages[p.id])}/">See ${esc(p.name)}’s full outlook</a>` : ''}
        </div>
      </details>
    </li>`;
  }).join('');
}

function rangeLabel() {
  if (state.preset === 'playoffs') return 'Fantasy playoffs';
  if (state.fromWeek === state.toWeek) return `Week ${state.fromWeek}`;
  return `Weeks ${state.fromWeek}\u2013${state.toWeek}`;
}

// There is no compare step: the list re-ranks, and the answer is rewritten,
// every time a player or a setting changes.
function render() {
  const n = state.picked.length;
  syncUrl();
  // With a league connected, its pickups take the introduction's place.
  $('#intro').hidden = n > 0 || !!state.league;
  $('#why').hidden = n > 0 || !!state.league;
  renderLeague();
  $('#hint').hidden = n !== 1;
  $('#answer').hidden = n < 2;
  $('#footnote').hidden = n === 0;
  if (!n) { $('#ranking').innerHTML = ''; return; }

  const rows = rank(state.picked, state.fromWeek, state.toWeek);
  renderRanking(rows);
  const span = state.fromWeek === state.toWeek
    ? `week ${state.fromWeek}` : `weeks ${state.fromWeek}\u2013${state.toWeek}`;
  $('#footnote').textContent = startSit()
    ? `Projected points in ${span}. A player on the injury report counts his chance of sitting ` +
      'out; anyone else is taken to play. Floor and ceiling are for a game he plays: tested on the ' +
      '2024 and 2025 seasons, about one game in ten fell below the floor and one in ten went above the ceiling.'
    : `Expected points per game over ${span}, counting the ` +
      'chance he misses a game. Value over replacement compares ' +
      'each player with the best free agent at his position in a league your size.';
  if (n < 2) return;

  const answer = headline(rows);
  $('#answerLabel').className = `pill ${answer.tone}`;
  $('#answerLabel').textContent = answer.label;
  $('#answerTitle').textContent = answer.title;
  $('#answerSub').textContent = answer.sub;
  const flip = flipNote(rows, state.fromWeek, state.toWeek);
  $('#flipNote').textContent = flip || '';
  $('#flipNote').hidden = !flip;
}

/* -- Sharing ------------------------------------------------------------- */

// The comparison lives in the address, so it can be sent to a league chat
// and opens exactly as it was, settings and all. Defaults are left out to
// keep shared links short.
function comparisonParams() {
  const q = new URLSearchParams();
  if (state.picked.length) q.set('p', state.picked.map(p => p.id).join(','));
  if (state.preset !== 'rest') q.set('w', `${state.fromWeek}-${state.toWeek}`);
  // A league's own scoring cannot travel in a link, so the nearest standard
  // format does.
  const format = state.format !== 'league' ? state.format
    : ({ 1: 'ppr', 0.5: 'half_ppr', 0: 'standard' }[state.data.meta.scoringFormats.league.reception] || 'ppr');
  if (format !== 'ppr') q.set('s', format);
  if (state.tePremium) q.set('te', String(state.tePremium));
  if (state.leagueSize !== 12) q.set('l', String(state.leagueSize));
  return q;
}

function syncUrl() {
  // Anything else on the address, such as an ad's tracking tags, is kept so
  // analytics can still see where a visit came from.
  const q = new URLSearchParams(location.search);
  for (const key of ['p', 'w', 's', 'te', 'l']) q.delete(key);
  for (const [key, value] of comparisonParams()) q.set(key, value);
  // Commas are legal in a query string, and "a,b" reads better in a group
  // chat than "a%2Cb".
  const s = q.toString().replace(/%2C/g, ',');
  history.replaceState(null, '', s ? `?${s}` : location.pathname);
}

function readUrl() {
  const q = new URLSearchParams(location.search);
  const meta = state.data.meta;
  if (meta.scoringFormats[q.get('s')]) state.format = q.get('s');
  const te = Number(q.get('te'));
  if ([0.5, 1].includes(te)) state.tePremium = te;
  const league = Number(q.get('l'));
  if ([8, 10, 12, 14, 16].includes(league)) state.leagueSize = league;

  // A link from an earlier week may start before this week's projections do.
  const [from, to] = (q.get('w') || '').split('-').map(Number);
  if (from && to) {
    const a = Math.max(from, meta.fromWeek);
    const b = Math.min(to, meta.throughWeek);
    if (a <= b) {
      state.fromWeek = a;
      state.toWeek = b;
      state.preset = a === meta.fromWeek && b === meta.throughWeek ? 'rest'
        : a === meta.fromWeek && b === a ? 'week'
        : a === 15 && b === Math.min(17, meta.throughWeek) ? 'playoffs' : 'custom';
    }
  }

  const ids = (q.get('p') || '').split(',').filter(Boolean);
  const byId = new Map(state.data.players.map(p => [p.id, p]));
  state.picked = ids.map(id => byId.get(id)).filter(Boolean).slice(0, 8);
  if (state.picked.length < ids.length) {
    showError('Some players from this link are not in this week’s projections.');
  }
}

async function share() {
  const label = $('#shareLabel');
  try {
    if (navigator.share) {
      await navigator.share({ title: 'Waiver', text: $('#answerTitle').textContent, url: location.href });
      return;
    }
    await navigator.clipboard.writeText(location.href);
    label.textContent = 'Link copied';
  } catch (err) {
    if (err && err.name === 'AbortError') return;
    label.textContent = 'Copy the address bar to share';
  }
  setTimeout(() => { label.textContent = 'Share this pick'; }, 2000);
}

/* -- Entry UI ------------------------------------------------------------ */

// Punctuation, accents, spaces and suffixes are ignored, so "cj stroud" finds
// C.J. Stroud and "devon achane" finds De'Von Achane.
function nameKey(s) {
  return s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
    .replace(/\b(jr|sr|ii|iii|iv|v)\b/g, '').replace(/[^a-z0-9]/g, '');
}

function search(query) {
  const q = nameKey(query);
  if (q.length < 2) return [];
  const picked = new Set(state.picked.map(p => p.id));
  const starts = p => (p.key.startsWith(q) || p.lastKey.startsWith(q) ? 0 : 1);
  return state.data.players
    .filter(p => !picked.has(p.id) && p.key.includes(q))
    .sort((a, b) => starts(a) - starts(b) || a.name.localeCompare(b.name))
    .slice(0, 8);
}

function initials(name) {
  return (name[0] + lastName(name)[0]).replace(/[^A-Za-z]/g, '').toUpperCase();
}

// Initials for everyone: free photos cover too few players to look even, and
// the league's own headshots are not licensed for a site with ads.
function avatar(p, size = '') {
  return `<span class="avatar ${size}" aria-hidden="true">${initials(p.name)}</span>`;
}

// In a connected league, whether a player can be picked up at all.
function leagueStatus(p) {
  const L = state.league;
  if (!L || !p.sleeperId) return '';
  if (L.mine.has(p.sleeperId)) return ' \u00b7 on your team';
  return L.rostered.has(p.sleeperId) ? ' \u00b7 on a team' : ' \u00b7 free agent';
}

function renderSuggestions(results) {
  const box = $('#suggestions');
  if (!results.length) { box.hidden = true; box.innerHTML = ''; return; }
  box.innerHTML = results.map((p, i) =>
    `<li role="option" data-add="${p.id}" ${i === 0 ? 'aria-selected="true"' : ''}>
      <span class="who">${avatar(p)}<span>${esc(p.name)}</span></span>
      <span class="tag">${esc(p.position)} \u00b7 ${esc(p.team)}${leagueStatus(p)}</span></li>`).join('');
  box.hidden = false;
  $('#search').setAttribute('aria-expanded', 'true');
}

function addPlayer(id) {
  const p = state.data.players.find(x => x.id === id);
  if (!p || state.picked.some(x => x.id === id)) return;
  if (state.picked.length >= 8) {
    showError('Eight players is the most the comparison can hold at once.');
    return;
  }
  state.picked.push(p);
  const input = $('#search');
  input.value = '';
  $('#suggestions').hidden = true;
  input.setAttribute('aria-expanded', 'false');
  showError(null);
  render();
  // Until there are two players the next step is always another search, so
  // the keyboard stays up. After that the answer is what people want to see.
  if (state.picked.length < 2) input.focus();
}

function showError(message) {
  const el = $('#entryError');
  el.textContent = message || '';
  el.hidden = !message;
}

/* -- Your Sleeper league ------------------------------------------------- */

// Sleeper's public API answers browsers directly, so a league loads with no
// sign-in and nothing passes through Waiver. Only the league and team chosen
// are remembered, in this browser.
const SLEEPER = 'https://api.sleeper.app/v1';
const LEAGUE_KEY = 'waiver.league';
const BETA_NOTE = '<p class="beta-note"><span class="beta">Beta</span> League mode is new, so check ' +
  'its picks against your league on Sleeper before acting on them.</p>';

// How each Sleeper lineup slot is usually filled, for working out how deep
// replacement level sits at each position in a particular league.
const SLOT_SHARE = {
  QB: { QB: 1 }, RB: { RB: 1 }, WR: { WR: 1 }, TE: { TE: 1 },
  FLEX: { RB: 0.45, WR: 0.45, TE: 0.1 }, WRRB_FLEX: { RB: 0.5, WR: 0.5 },
  REC_FLEX: { WR: 0.8, TE: 0.2 }, SUPER_FLEX: { QB: 0.9, RB: 0.05, WR: 0.05 },
};
const DEFAULT_LINEUP = ['QB', 'RB', 'RB', 'WR', 'WR', 'TE', 'FLEX'];

async function sleeperGet(path) {
  const res = await fetch(`${SLEEPER}${path}`);
  if (!res.ok) throw new Error(`Sleeper did not answer (${res.status}). Try again in a moment.`);
  return res.json();
}

function slotStarters(slots) {
  const out = { QB: 0, RB: 0, WR: 0, TE: 0 };
  for (const slot of slots) {
    for (const [pos, share] of Object.entries(SLOT_SHARE[slot] || {})) out[pos] += share;
  }
  return out;
}

// Replacement moves by however many more or fewer starters a league plays at
// a position than the usual one-quarterback lineup the defaults assume, so a
// superflex league makes quarterbacks scarce.
function leagueReplacement(slots) {
  const mine = slotStarters(slots);
  const usual = slotStarters(DEFAULT_LINEUP);
  return Object.fromEntries(Object.entries(REPLACEMENT_PER_TEAM)
    .map(([pos, per]) => [pos, Math.max(0.5, per + mine[pos] - usual[pos])]));
}

// Sleeper's scoring keys, mapped onto the stat lines the model predicts.
// Bonuses the model does not predict, such as points for first downs, are
// left out.
function leagueWeights(s) {
  const n = (key, fallback) => (typeof s[key] === 'number' ? s[key] : fallback);
  return {
    pass_yd: n('pass_yd', 0.04), pass_td: n('pass_td', 4), interception: n('pass_int', -2),
    rush_yd: n('rush_yd', 0.1), rush_td: n('rush_td', 6), reception: n('rec', 0),
    rec_yd: n('rec_yd', 0.1), rec_td: n('rec_td', 6), fumble_lost: n('fum_lost', -2),
    two_point: n('rec_2pt', n('rush_2pt', 2)),
  };
}

function receptionLabel(ppr) {
  return ppr === 1 ? 'full PPR' : ppr === 0.5 ? 'half PPR' : ppr === 0 ? 'no PPR' : `${ppr} per reception`;
}

function applyLeague(league, rosters, users, rosterId) {
  const mine = rosters.find(r => r.roster_id === rosterId);
  const owner = users.find(u => u.user_id === mine?.owner_id);
  // Players in reserve and taxi slots do not take a roster spot, so they are
  // never offered as the player to let go.
  const parked = new Set([...(mine?.reserve || []), ...(mine?.taxi || [])]);
  const weights = leagueWeights(league.scoring_settings || {});
  const budget = league.settings?.waiver_budget || 0;
  state.league = {
    id: league.league_id,
    name: league.name || 'Your league',
    teams: league.total_rosters || rosters.length,
    rosterId,
    team: owner?.metadata?.team_name || owner?.display_name || `Team ${rosterId}`,
    rostered: new Set(rosters.flatMap(r => r.players || [])),
    mine: new Set((mine?.players || []).filter(id => !parked.has(id))),
    slots: league.roster_positions || [],
    faab: budget ? { left: Math.max(0, budget - (mine?.settings?.waiver_budget_used || 0)) } : null,
  };
  state.data.meta.scoringFormats.league = { ...weights, label: `League scoring, ${receptionLabel(weights.reception)}` };
  state.format = 'league';
  state.tePremium = league.scoring_settings?.bonus_rec_te || 0;
  state.leagueSize = state.league.teams;
  state.replacement = leagueReplacement(state.league.slots);
  simCache.clear();
  try { localStorage.setItem(LEAGUE_KEY, JSON.stringify({ id: league.league_id, rosterId })); } catch { /* private mode */ }
  syncSettings();
  render();
}

function disconnectLeague() {
  state.league = null;
  delete state.data.meta.scoringFormats.league;
  state.format = 'ppr';
  state.tePremium = 0;
  state.leagueSize = 12;
  state.replacement = { ...REPLACEMENT_PER_TEAM };
  simCache.clear();
  try { localStorage.removeItem(LEAGUE_KEY); } catch { /* private mode */ }
  syncSettings();
  render();
}

function connectMessage(html) {
  const box = $('#connectResult');
  if (box) box.innerHTML = html;
  else showError(html.replace(/<[^>]+>/g, ''));
}

// Loads a league and settles on a team: the one given, the one the user owns,
// or, failing both, asks which is theirs.
async function chooseTeam(leagueId, userId, rosterId = null) {
  const [league, rosters, users] = await Promise.all([
    sleeperGet(`/league/${encodeURIComponent(leagueId)}`),
    sleeperGet(`/league/${encodeURIComponent(leagueId)}/rosters`),
    sleeperGet(`/league/${encodeURIComponent(leagueId)}/users`),
  ]);
  if (!league || !Array.isArray(rosters)) {
    connectMessage('<p class="error">Sleeper has no league with that ID.</p>');
    return;
  }
  const owns = r => r.owner_id === userId || (r.co_owners || []).includes(userId);
  const mine = rosterId != null ? rosters.find(r => r.roster_id === rosterId) : rosters.find(owns);
  if (mine) {
    applyLeague(league, rosters, users || [], mine.roster_id);
    closeSheet();
    return;
  }
  const teamName = r => {
    const u = (users || []).find(x => x.user_id === r.owner_id);
    return u?.metadata?.team_name || u?.display_name || `Team ${r.roster_id}`;
  };
  connectMessage('<p class="label">Which team is yours?</p>' + rosters.map(r =>
    option(false, esc(teamName(r)), '',
      `data-sleeper-league="${esc(leagueId)}" data-sleeper-roster="${r.roster_id}"`)).join(''));
}

async function findLeagues(input) {
  const text = input.trim();
  if (!text) { connectMessage('<p class="error">Enter a Sleeper username or league ID.</p>'); return; }
  connectMessage('<p class="hint">Looking that up on Sleeper…</p>');
  try {
    // League IDs are long numbers; anything else is taken as a username.
    if (/^\d{12,}$/.test(text)) { await chooseTeam(text, null); return; }
    const user = await sleeperGet(`/user/${encodeURIComponent(text)}`);
    if (!user) {
      connectMessage('<p class="error">Sleeper has no user by that name. Check the spelling, or paste a league ID instead.</p>');
      return;
    }
    const season = state.data.meta.season;
    const leagues = (await sleeperGet(`/user/${user.user_id}/leagues/nfl/${season}`)) || [];
    if (!leagues.length) {
      connectMessage(`<p class="error">${esc(user.display_name)} has no ${season} leagues on Sleeper.</p>`);
      return;
    }
    if (leagues.length === 1) { await chooseTeam(leagues[0].league_id, user.user_id); return; }
    connectMessage('<p class="label">Pick a league</p>' + leagues.map(l =>
      option(false, esc(l.name), `${l.total_rosters} teams`,
        `data-sleeper-league="${esc(l.league_id)}" data-sleeper-user="${esc(user.user_id)}"`)).join(''));
  } catch (err) {
    connectMessage(`<p class="error">${esc(err.message)}</p>`);
  }
}

async function reloadLeague(saved) {
  try { await chooseTeam(saved.id, null, saved.rosterId); } catch (err) {
    showError(`Your Sleeper league could not be loaded: ${err.message}`);
  }
}

// A rule of thumb rather than a model: a share of the budget that is left,
// rising with how much a player adds over the one he would replace.
function faabBid(gain, left) {
  if (!left) return '$0';
  const [lo, hi] = gain >= 3 ? [0.2, 0.3] : gain >= 2 ? [0.1, 0.2] : gain >= 1 ? [0.05, 0.1] : [0.01, 0.04];
  const a = Math.max(1, Math.round(left * lo));
  const b = Math.max(a, Math.round(left * hi));
  return a === b ? `$${a}` : `$${a}–${b}`;
}

// Who can fill each lineup slot, in the order slots are filled: dedicated
// slots first, then the flexible ones from the narrowest to the widest.
const SLOT_ELIGIBLE = {
  QB: ['QB'], RB: ['RB'], WR: ['WR'], TE: ['TE'],
  REC_FLEX: ['WR', 'TE'], WRRB_FLEX: ['RB', 'WR'], FLEX: ['RB', 'WR', 'TE'],
  SUPER_FLEX: ['QB', 'RB', 'WR', 'TE'],
};

// Expected points per game from the best starting lineup a roster can field
// in these slots. Depth beyond the lineup is worth nothing here, which is
// what stops a third quarterback looking like an upgrade.
function lineupPoints(roster, slots) {
  const pool = [...roster].sort((a, b) => b.pts - a.pts);
  const used = new Set();
  let total = 0;
  for (const [slot, eligible] of Object.entries(SLOT_ELIGIBLE)) {
    for (let i = slots.filter(s => s === slot).length; i > 0; i--) {
      const pick = pool.find(m => !used.has(m) && eligible.includes(m.p.position));
      if (pick) { used.add(pick); total += pick.pts; }
    }
  }
  return total;
}

// The roster moves worth making, best first. Each is judged by how much it
// lifts the best starting lineup, and made on the roster the moves before it
// leave behind, so every pickup replaces a different player and none empties
// a slot the lineup needs.
function leagueAdvice() {
  const L = state.league;
  const levels = replacementLevels(state.fromWeek, state.toWeek);
  const entry = p => {
    const pts = meanPpg(p, state.fromWeek, state.toWeek);
    return { p, pts, v: pts - (levels[p.position] || 0) };
  };
  const bySleeper = new Map(state.data.players.filter(p => p.sleeperId).map(p => [p.sleeperId, p]));
  let roster = [...L.mine].map(id => bySleeper.get(id)).filter(Boolean).map(entry);
  const start = roster.length;
  const free = state.data.players.filter(p => p.sleeperId && !L.rostered.has(p.sleeperId))
    .map(entry).sort((a, b) => b.v - a.v);
  const slots = L.slots.length ? L.slots : DEFAULT_LINEUP;
  const pickups = [];
  const taken = new Set();
  for (let k = 0; k < 5; k++) {
    const base = lineupPoints(roster, slots);
    let best = null;
    for (const add of free.slice(0, 80)) {
      if (taken.has(add)) continue;
      for (const drop of roster) {
        if (drop.added) continue;
        const gain = lineupPoints(roster.filter(m => m !== drop).concat(add), slots) - base;
        // Between equal gains, the weaker player is the one to let go.
        if (!best || gain > best.gain + 1e-9
            || (Math.abs(gain - best.gain) <= 1e-9 && add === best.add && drop.v < best.drop.v)) {
          best = { add, drop, gain };
        }
      }
    }
    if (!best || best.gain < 0.3) break;
    pickups.push(best);
    taken.add(best.add);
    roster = roster.filter(m => m !== best.drop).concat({ ...best.add, added: true });
  }
  return { pickups, free, roster: start };
}

function renderLeague() {
  const box = $('#league');
  const L = state.league;
  box.hidden = !L || state.picked.length > 0;
  if (box.hidden) return;
  const { pickups, free, roster } = leagueAdvice();
  const rows = pickups.map(({ add, drop, gain }) => `<li class="pick">
      <div class="pick-who">${avatar(add.p)}<div><strong>${esc(add.p.name)}</strong>
        <small>${esc(add.p.position)} · ${esc(add.p.team)}${add.p.injury
          ? ` · ${esc(SHORT_STATUS[add.p.injury.status] || add.p.injury.status)}` : ''}</small></div></div>
      <div class="pick-gain"><b>+${gain.toFixed(1)}</b><small>replacing ${esc(drop.p.name)}</small></div>
      <button class="chip" type="button" data-compare="${esc(add.p.id)},${esc(drop.p.id)}">Compare</button>
      ${L.faab ? `<p class="pick-bid">FAAB: about ${faabBid(gain, L.faab.left)} of your $${L.faab.left}</p>` : ''}
    </li>`).join('');
  const best = ['QB', 'RB', 'WR', 'TE'].map(pos => {
    const top = free.filter(f => f.p.position === pos).slice(0, 3);
    return `<p><span class="label">${pos}</span>${top.length
      ? top.map(f => `${esc(f.p.name)} <small>${signed(f.v)}</small>`).join(' · ')
      : 'No one worth a spot'}</p>`;
  }).join('');
  const empty = roster
    ? 'No free agent in your league lifts your starting lineup by enough to be worth a move right now.'
    : 'Your roster has no players the model projects yet, so there is nothing to compare against.';
  box.innerHTML = `<div class="league-head">
      <h2 id="leagueTitle">Top pickups for ${esc(L.team)} <span class="beta">Beta</span></h2>
      <span>${esc(L.name)} · ${L.teams} teams · ${esc(state.data.meta.scoringFormats.league.label.replace('League scoring, ', ''))} · ${rangeLabel().toLowerCase()}</span>
    </div>
    ${rows ? `<ol class="picks">${rows}</ol>` : `<p class="league-note">${empty}</p>`}
    <p class="league-note">Each pickup is paired with the player on your roster he would replace,
      and the gain is how much the move lifts your best starting lineup, in points per game over
      ${rangeLabel().toLowerCase()}. Moves are listed in the order to make them.${L.faab
        ? ' FAAB amounts are a rule of thumb scaled by that gain, not a model prediction.' : ''}</p>
    <details class="league-more"><summary>Best free agents at each position</summary>${best}</details>`;
}

/* -- Settings sheet ------------------------------------------------------ */

const CHECK = '<svg viewBox="0 0 20 20"><path d="M4 10.5 L8 14.5 L16 5.5"/></svg>';

function option(checked, title, sub, attrs) {
  return `<button class="opt" type="button" role="radio" aria-checked="${checked}" ${attrs}>
    <span><strong>${title}</strong>${sub ? `<small>${sub}</small>` : ''}</span>
    <span class="check">${checked ? CHECK : ''}</span></button>`;
}

function openSheet(kind) {
  const meta = state.data.meta;
  const body = $('#sheetBody');
  let title = '';

  if (kind === 'weeks') {
    title = 'Weeks to project';
    const last = meta.throughWeek;
    const opts = [
      ['rest', 'Rest of season', `Weeks ${meta.fromWeek}\u2013${last}`],
      ['playoffs', 'Fantasy playoffs', `Weeks 15\u2013${Math.min(17, last)}`],
      ['week', 'This week: start or sit', `Week ${meta.fromWeek}, with a floor and a ceiling`],
    ].map(([key, label, sub]) =>
      option(state.preset === key, label, sub, `data-preset="${key}"`)).join('');

    const weekOptions = (selected) => {
      let out = '';
      for (let w = meta.fromWeek; w <= last; w++) {
        out += `<option value="${w}"${w === selected ? ' selected' : ''}>Week ${w}</option>`;
      }
      return out;
    };
    body.innerHTML = opts + `<div class="custom">
      <p class="label">Custom range</p>
      <div class="range">
        <select id="fromSel" aria-label="First week">${weekOptions(state.fromWeek)}</select>
        <span>to</span>
        <select id="toSel" aria-label="Last week">${weekOptions(state.toWeek)}</select>
      </div>
      <button class="primary" type="button" data-apply="custom" style="margin-top:0">Apply range</button>
    </div>`;
  }

  if (kind === 'format') {
    title = 'Scoring format';
    body.innerHTML = Object.entries(meta.scoringFormats).map(([key, f]) =>
      option(state.format === key, f.label,
        f.reception === 1 ? '1 point per reception'
          : f.reception === 0.5 ? '0.5 points per reception' : 'No points per reception',
        `data-format="${key}"`)).join('')
      + `<div class="custom">
        <p class="label">Tight end premium</p>
        ${[0, 0.5, 1].map(v => option(state.tePremium === v,
          v === 0 ? 'None' : `+${v} per reception`, '', `data-te="${v}"`)).join('')}
      </div>`;
  }

  if (kind === 'league') {
    title = 'League size';
    body.innerHTML = [8, 10, 12, 14, 16].map(n =>
      option(state.leagueSize === n, `${n} teams`,
        `Replacement is about the ${Math.round(state.replacement.RB * n)}th running back`,
        `data-league="${n}"`)).join('');
  }

  if (kind === 'sleeper') {
    const L = state.league;
    title = L ? 'Your Sleeper league' : 'Connect a Sleeper league';
    body.innerHTML = L
      ? `<div class="connect">
          ${BETA_NOTE}
          <p><strong>${esc(L.name)}</strong><br><small>${esc(L.team)} · ${L.teams} teams</small></p>
          <p class="hint">Scoring, league size and lineup come from Sleeper, and rosters were
            loaded when this page opened.</p>
          <button class="opt" type="button" data-refresh="1">Reload rosters from Sleeper</button>
          <button class="opt" type="button" data-disconnect="1">Disconnect this league</button>
        </div>`
      : `<form class="connect" id="connectForm">
          ${BETA_NOTE}
          <label class="label" for="sleeperInput">Sleeper username or league ID</label>
          <input id="sleeperInput" type="text" autocomplete="off" spellcheck="false"
                 autocapitalize="off" placeholder="Your Sleeper username">
          <button class="primary" type="submit" style="margin-top:0">Find my leagues</button>
          <p class="hint">Your browser reads the league straight from Sleeper. Nothing goes to
            Waiver, and there is nothing to sign in to.</p>
          <div id="connectResult"></div>
        </form>`;
  }

  $('#sheetTitle').textContent = title;
  $('#scrim').hidden = false;
  ($('#sleeperInput') || $('#sheetClose')).focus();
}

function closeSheet() { $('#scrim').hidden = true; }

function setPreset(key) {
  const meta = state.data.meta;
  state.preset = key;
  if (key === 'rest') { state.fromWeek = meta.fromWeek; state.toWeek = meta.throughWeek; }
  if (key === 'playoffs') { state.fromWeek = 15; state.toWeek = Math.min(17, meta.throughWeek); }
  if (key === 'week') { state.fromWeek = meta.fromWeek; state.toWeek = meta.fromWeek; }
  syncSettings();
}

function syncSettings() {
  const single = startSit();
  for (const b of document.querySelectorAll('.mode')) {
    b.setAttribute('aria-pressed', String((b.dataset.mode === 'week') === single));
  }
  $('#introTitle').textContent = single ? 'Who should you start?' : 'Who should you pick up?';
  $('#introLede').textContent = single
    ? `Add two or more of your players. Waiver projects week ${state.fromWeek} for each, with a ` +
      'floor and a ceiling, and how often each outscores the other.'
    : 'Add two or more fantasy football free agents. Waiver ranks them on the usage and efficiency ' +
      'that predicts the rest of the season, not the points they already scored.';
  $('#weekValue').textContent = rangeLabel();
  $('#formatValue').textContent = (state.format === 'league' ? 'League scoring'
    : state.data.meta.scoringFormats[state.format].label)
    + (state.tePremium ? ` +${state.tePremium} TE` : '');
  $('#leagueValue').textContent = `${state.leagueSize} teams`;
  $('#sleeperValue').textContent = state.league ? state.league.name : 'Connect Sleeper';
}

/* -- Wiring -------------------------------------------------------------- */

function wire() {
  const input = $('#search');
  input.addEventListener('input', () => renderSuggestions(search(input.value)));
  input.addEventListener('keydown', e => {
    const box = $('#suggestions');
    if (e.key === 'Enter' && !box.hidden) {
      e.preventDefault();
      const sel = box.querySelector('[aria-selected="true"]') || box.firstElementChild;
      if (sel) addPlayer(sel.dataset.add);
    }
    if (e.key === 'Escape') { box.hidden = true; }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      if (box.hidden) return;
      e.preventDefault();
      const items = [...box.children];
      const cur = items.findIndex(li => li.getAttribute('aria-selected') === 'true');
      const next = e.key === 'ArrowDown'
        ? Math.min(items.length - 1, cur + 1) : Math.max(0, cur - 1);
      items.forEach(li => li.removeAttribute('aria-selected'));
      items[next].setAttribute('aria-selected', 'true');
    }
  });

  $('#suggestions').addEventListener('click', e => {
    const li = e.target.closest('[data-add]');
    if (li) addPlayer(li.dataset.add);
  });

  $('#ranking').addEventListener('click', e => {
    const btn = e.target.closest('[data-remove]');
    if (!btn) return;
    // The button sits inside the row's summary, so stop it toggling the row.
    e.preventDefault();
    state.picked = state.picked.filter(p => p.id !== btn.dataset.remove);
    showError(null);
    render();
  });

  document.addEventListener('click', e => {
    if (!e.target.closest('.search')) {
      $('#suggestions').hidden = true;
      $('#search').setAttribute('aria-expanded', 'false');
    }
  });

  $('#weekSetting').addEventListener('click', () => openSheet('weeks'));
  $('#formatSetting').addEventListener('click', () => openSheet('format'));
  $('#leagueSetting').addEventListener('click', () => openSheet('league'));
  $('#sleeperSetting').addEventListener('click', () => openSheet('sleeper'));
  $('#share').addEventListener('click', share);
  $('.modes').addEventListener('click', e => {
    const btn = e.target.closest('[data-mode]');
    if (!btn) return;
    setPreset(btn.dataset.mode);
    render();
  });

  // A pickup's Compare button puts him beside the player he would replace, so
  // the explanation below says exactly why the move is worth making.
  $('#league').addEventListener('click', e => {
    const btn = e.target.closest('[data-compare]');
    if (!btn) return;
    const byId = new Map(state.data.players.map(p => [p.id, p]));
    state.picked = btn.dataset.compare.split(',').map(id => byId.get(id)).filter(Boolean);
    showError(null);
    render();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  $('#sheetBody').addEventListener('submit', e => {
    e.preventDefault();
    if (e.target.id === 'connectForm') findLeagues($('#sleeperInput').value);
  });
  $('#sheetClose').addEventListener('click', closeSheet);
  $('#scrim').addEventListener('click', e => { if (e.target === $('#scrim')) closeSheet(); });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !$('#scrim').hidden) closeSheet();
  });

  $('#sheetBody').addEventListener('click', e => {
    const btn = e.target.closest('button');
    if (!btn) return;
    const d = btn.dataset;
    if (d.preset) { setPreset(d.preset); closeSheet(); }
    else if (d.format) { state.format = d.format; syncSettings(); closeSheet(); }
    else if (d.te !== undefined) { state.tePremium = Number(d.te); syncSettings(); closeSheet(); }
    else if (d.league) { state.leagueSize = Number(d.league); syncSettings(); closeSheet(); }
    else if (d.sleeperLeague) {
      chooseTeam(d.sleeperLeague, d.sleeperUser || null, d.sleeperRoster ? Number(d.sleeperRoster) : null)
        .catch(err => connectMessage(`<p class="error">${esc(err.message)}</p>`));
      return;
    }
    else if (d.refresh) { closeSheet(); reloadLeague({ id: state.league.id, rosterId: state.league.rosterId }); return; }
    else if (d.disconnect) { closeSheet(); disconnectLeague(); return; }
    else if (d.apply === 'custom') {
      const from = Number($('#fromSel').value);
      const to = Number($('#toSel').value);
      if (to < from) {
        $('#sheetBody').querySelector('.custom').insertAdjacentHTML('beforeend',
          '<p class="error">The last week has to come after the first.</p>');
        return;
      }
      state.fromWeek = from; state.toWeek = to; state.preset = 'custom';
      syncSettings(); closeSheet();
    }
    render();
  });
}

/* -- Boot ---------------------------------------------------------------- */

async function boot() {
  // Which players have a page of their own. Fetched alongside the projections
  // but never waited on: without it the results simply carry no links.
  const pages = fetch('data/player-pages.json')
    .then(res => (res.ok ? res.json() : {}))
    .catch(() => ({}));
  try {
    const res = await fetch('data/projections.json');
    if (!res.ok) throw new Error(res.statusText);
    state.data = await res.json();
  } catch (err) {
    $('#dataStamp').textContent = 'Data unavailable';
    $('#panel').insertAdjacentHTML('beforeend',
      '<p class="error">Projections could not be loaded. Run the weekly build ' +
      'to generate data/projections.json, then reload.</p>');
    return;
  }

  for (const p of state.data.players) {
    p.key = nameKey(p.name);
    p.lastKey = nameKey(lastName(p.name));
  }

  const meta = state.data.meta;
  state.fromWeek = meta.fromWeek;
  state.toWeek = meta.throughWeek;
  const stamp = new Date(meta.generated);
  if (meta.source === 'synthetic') {
    $('#dataStamp').textContent = 'Demo data \u00b7 not real players';
    $('#dataStamp').classList.add('warn-stamp');
  } else {
    const form = meta.earlySeason
      ? 'early season, form partly from last season'
      : `through week ${meta.fromWeek - 1}`;
    $('#dataStamp').textContent =
      `Season ${meta.season} \u00b7 ${form} \u00b7 updated ` +
      stamp.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  }

  readUrl();
  syncSettings();
  render();
  wire();

  // A league connected on an earlier visit reloads fresh from Sleeper.
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(LEAGUE_KEY) || 'null'); } catch { /* private mode */ }
  if (saved && saved.id) reloadLeague(saved);

  state.pages = await pages;
  if (state.picked.length) render();
}

boot();
