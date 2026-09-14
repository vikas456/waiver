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
const playChance = (player, wk) => wk.p ?? player.playProb;

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
    return { ppg: 0, p10: 0, p90: 0, total: 0, games: 0, weeks: 0 };
  }

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
      if (Math.random() >= playChance(player, weeks[i])) continue;
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

  return {
    ppg: sorted.reduce((a, b) => a + b, 0) / draws,
    p10: at(0.10),
    p90: at(0.90),
    total: totalSum / draws,
    games: gameSum / draws,
    weeks: weeks.length,
  };
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
    const rank = Math.max(1, Math.round((REPLACEMENT_PER_TEAM[pos] || 2) * state.leagueSize));
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

// Contributions are re-centred against the other players in this comparison.
// Users want to know why B beat A, not why B sits above a league average they
// never asked about.
function relativeDrivers(all) {
  const groups = new Set();
  all.forEach(d => Object.keys(d).forEach(g => groups.add(g)));
  const means = {};
  for (const g of groups) {
    means[g] = all.reduce((sum, d) => sum + (d[g] || 0), 0) / all.length;
  }
  return all.map(d => {
    const rel = {};
    for (const g of groups) rel[g] = (d[g] || 0) - means[g];
    return rel;
  });
}

/* -- Ranking ------------------------------------------------------------- */

function rank(players, from, to) {
  const levels = replacementLevels(from, to);
  const pprWeights = state.data.meta.scoringFormats.ppr;
  const fmtWeights = state.data.meta.scoringFormats[state.format];

  const rows = players.map(p => {
    const sim = simulateCached(p, from, to);
    const weeks = weeksInRange(p, from, to);
    const pprMean = weeks.length
      ? weeks.reduce((a, w) => a + scoreLine(w.c, pprWeights, p.position, 0), 0) / weeks.length
      : 1;
    const fmtMean = weeks.length
      ? weeks.reduce((a, w) => a + scoreLine(w.c, fmtWeights, p.position, state.tePremium), 0) / weeks.length
      : 1;
    const scale = pprMean > 0 ? fmtMean / pprMean : 1;

    const replacement = levels[p.position] || 0;
    const vorp = sim.ppg - replacement;

    const byes = (p.byeWeeks || []).filter(w => w >= from && w <= to);
    return { player: p, ...sim, replacement, vorp, drivers: driverTotals(p, from, to, scale), byes };
  });

  const rel = relativeDrivers(rows.map(r => r.drivers));
  rows.forEach((r, i) => { r.rel = rel[i]; });

  // Two players can share a surname, and "against Moreau and Moreau" is the
  // kind of sentence that makes a reader stop trusting everything above it.
  const surnames = rows.map(r => lastName(r.player.name));
  rows.forEach((r, i) => {
    const clash = surnames.filter(n => n === surnames[i]).length > 1;
    r.displayName = clash ? r.player.name : surnames[i];
  });
  rows.sort((a, b) => b.vorp - a.vorp);
  return rows;
}

/* -- Reasoning ----------------------------------------------------------- */

function listOf(items) {
  if (items.length <= 1) return String(items[0] ?? '');
  return items.slice(0, -1).join(', ') + ' and ' + items[items.length - 1];
}

// Opportunity means different things by position: a back is judged on touches,
// a receiver on targets. Saying "targets" about a running back is the kind of
// small wrongness that makes a whole explanation look automated.
const OPPORTUNITY_PHRASES = {
  RB: ['takes a bigger share of his backfield\u2019s touches',
    'splits his backfield\u2019s touches with too many other backs'],
  QB: ['carries more of his offence\u2019s volume', 'handles less volume than the others'],
};

const PHRASES = {
  opportunity: ['commands a bigger share of his offence\u2019s targets',
    'sees a smaller share of his offence\u2019s targets'],
  route_role: ['is on the field and running routes far more often',
    'runs routes on too few dropbacks to hold a steady floor'],
  goal_line: ['owns the work near the goal line', 'rarely gets the ball inside the five'],
  efficiency: ['is producing more than his opportunities alone would suggest',
    'has been inefficient with the touches he does get'],
  td_regression: ['has scored about what his usage supports',
    'has scored far more than his usage supports'],
  trend: ['has been gaining role over the past few weeks',
    'has been losing role over the past few weeks'],
  production: ['has been putting up bigger stat lines', 'has been putting up smaller stat lines'],
  market: ['is rated higher by expert consensus', 'is rated lower by expert consensus'],
  offense: ['plays in a faster, more productive offence',
    'is stuck in an offence that does not generate enough volume'],
  schedule: ['draws a friendlier set of remaining defences',
    'faces a harder set of remaining defences'],
  availability: ['has a longer track record to judge from', 'has a thin sample to judge from'],
  prior: ['profiles well for his role', 'profiles poorly for his role'],
};

const ORDINAL = ['', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth'];

// Real names carry suffixes, and "Jr. ranks first" is not a sentence.
function lastName(name) {
  const bits = name.split(' ').filter(b => !/^(jr|sr|ii|iii|iv|v)\.?$/i.test(b));
  return bits[bits.length - 1] || name;
}

function phraseFor(group, position) {
  if (group === 'opportunity' && OPPORTUNITY_PHRASES[position]) {
    return OPPORTUNITY_PHRASES[position];
  }
  return PHRASES[group] || ['rates well here', 'rates poorly here'];
}

function splitDrivers(rel) {
  const entries = Object.entries(rel).filter(([, v]) => Math.abs(v) > 0.12);
  return {
    up: entries.filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]),
    down: entries.filter(([, v]) => v < 0).sort((a, b) => a[1] - b[1]),
  };
}

function reasoning(row, position, others) {
  const rel = row.rel;
  const { up, down } = splitDrivers(rel);
  const name = row.displayName || lastName(row.player.name);
  const parts = [];
  const say = ([g, v]) => {
    const phrase = phraseFor(g, row.player.position);
    return v > 0 ? phrase[0] : phrase[1];
  };

  if (!up.length && !down.length) {
    parts.push(`${name} lands close to the middle of this group on every factor the model weighs.`);
  } else if (position === 1) {
    // Lead with what earned the ranking, then concede the strongest argument
    // against it. A case that never mentions the downside is not a case.
    const lead = up.length ? up : down;
    parts.push(`${name} ranks first because he ${say(lead[0])}` +
      (lead[1] ? `, and ${say(lead[1])}.` : '.'));
    if (up.length && down.length) {
      parts.push(`That holds even though he ${say(down[0])}.`);
    }
  } else {
    const lead = down.length ? down : up;
    parts.push(`${name} ${say(lead[0])}` +
      (lead[1] ? `, and ${say(lead[1])},` : ',') +
      ` which is what drops him to ${ORDINAL[position] || position}.`);
    if (down.length && up.length) {
      parts.push(`In his favour, he ${say(up[0])}, but not by enough to close the gap.`);
    }
  }

  const s = row.player.stats;
  if (s.actualTd != null && s.expectedTd != null) {
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
  if (row.byes.length) {
    const word = row.byes.length > 1 ? 'weeks' : 'week';
    parts.push(`He is on bye in ${word} ${listOf(row.byes)}, which the per-game ` +
      'figure above already excludes, so his total over the range is lower than ' +
      'the average suggests.');
  }
  if (others.length) {
    parts.push(`Against ${listOf(others)}, that is the difference.`);
  }
  return parts.join(' ');
}

// The answer leads, in one sentence. How firmly it is worded follows the size
// of the gap, so a near tie never reads as a sure thing.
function headline(rows) {
  const [top, second] = rows;
  const gap = top.ppg - second.ppg;
  const vorpGap = top.vorp - second.vorp;
  let answer;

  if (vorpGap < 0.35) {
    answer = {
      tone: 'warn', label: 'Close call',
      title: `${top.player.name}, narrowly over ${second.player.name}.`,
      sub: `They finish ${vorpGap.toFixed(1)} points of value apart. ` +
        'Take the one whose role you believe in.',
    };
  } else {
    answer = {
      tone: vorpGap < 1 ? 'neutral' : 'good',
      label: vorpGap < 1 ? 'Slight edge' : 'Clear pick',
      title: `Pick up ${top.player.name}.`,
      // Fewer points but still first is the case value over replacement exists for.
      sub: gap < 0
        ? `He projects ${Math.abs(gap).toFixed(1)} fewer points a game than ${second.player.name}, ` +
          `but ${top.player.position} is the thinner position in your league, so he replaces a worse player.`
        : `About ${gap.toFixed(1)} more points a game than ${second.player.name}.`,
    };
  }
  if (top.p90 - top.p10 > 10) {
    answer.sub += ' His range is wide, so take him if you need upside and the ' +
      'steadier option if you are protecting a lead.';
  }
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
      <span>${labels[g] || g}</span>
      <div class="track">${bar}</div>
      <b style="color:var(${v > 0 ? '--up' : '--down'})">${signed(v)}</b>
    </div>`;
  }).join('') + '</div>';
}

function renderStatLine(row) {
  const p = row.player;
  const s = p.stats;
  const pill = confidencePill(row);
  const items = pill ? [`<span class="pill ${pill.cls}">${pill.text}</span>`] : [];
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
  $('#ranking').innerHTML = rows.map((row, i) => {
    const p = row.player;
    const others = rows.filter(r => r !== row).map(r => r.displayName);
    return `<li>
      <details class="row"${i === 0 ? ' open' : ''}>
        <summary class="row-head">
          <span class="rank">${i + 1}</span>
          ${avatar(p, 'lg')}
          <span class="row-main">
            <strong>${esc(p.name)}</strong>
            <span class="row-meta">${esc(p.position)} \u00b7 ${esc(p.team)} \u00b7
              <span class="${row.vorp >= 0 ? 'up' : 'down'}">${signed(row.vorp)}</span> over replacement</span>
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
          ${compared ? `<p class="reason">${esc(reasoning(row, i + 1, others))}</p>` : ''}
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
  $('#intro').hidden = n > 0;
  $('#why').hidden = n > 0;
  $('#hint').hidden = n !== 1;
  $('#answer').hidden = n < 2;
  $('#footnote').hidden = n === 0;
  if (!n) { $('#ranking').innerHTML = ''; return; }

  const rows = rank(state.picked, state.fromWeek, state.toWeek);
  renderRanking(rows);
  const span = state.fromWeek === state.toWeek
    ? `week ${state.fromWeek}` : `weeks ${state.fromWeek}\u2013${state.toWeek}`;
  $('#footnote').textContent = `Expected points per game over ${span}, counting the ` +
    'chance he misses a game. Byes are left out. Value over replacement compares ' +
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
  if (state.format !== 'ppr') q.set('s', state.format);
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

// A photo that fails to load falls back to initials rather than a broken image.
function avatar(p, size = '') {
  const letters = initials(p.name);
  if (!p.headshot) return `<span class="avatar ${size}">${letters}</span>`;
  return `<span class="avatar ${size}" data-initials="${letters}"><img src="${esc(p.headshot)}" alt=""
    loading="lazy" onerror="this.parentElement.textContent = this.parentElement.dataset.initials"></span>`;
}

function renderSuggestions(results) {
  const box = $('#suggestions');
  if (!results.length) { box.hidden = true; box.innerHTML = ''; return; }
  box.innerHTML = results.map((p, i) =>
    `<li role="option" data-add="${p.id}" ${i === 0 ? 'aria-selected="true"' : ''}>
      <span class="who">${avatar(p)}<span>${esc(p.name)}</span></span>
      <span class="tag">${esc(p.position)} \u00b7 ${esc(p.team)}</span></li>`).join('');
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
      ['week', 'This week only', `Week ${meta.fromWeek}`],
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
        `Replacement is about the ${Math.round(2.5 * n)}th running back`,
        `data-league="${n}"`)).join('');
  }

  $('#sheetTitle').textContent = title;
  $('#scrim').hidden = false;
  $('#sheetClose').focus();
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
  $('#weekValue').textContent = rangeLabel();
  $('#formatValue').textContent = state.data.meta.scoringFormats[state.format].label
    + (state.tePremium ? ` +${state.tePremium} TE` : '');
  $('#leagueValue').textContent = `${state.leagueSize} teams`;
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
  $('#share').addEventListener('click', share);
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

  state.pages = await pages;
  if (state.picked.length) render();
}

boot();
