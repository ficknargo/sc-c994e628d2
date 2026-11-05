import test from 'node:test';
import assert from 'node:assert/strict';
import {scoreMove, emptyBoard, applyPlacements, replay, rackValue, PREMIUM, gameStats, endGameAdjustments, consecutiveScoreless} from '../js/scoring.js';

const row = (r, c0, word, blanks = []) => [...word].map((l, i) => ({r, c: c0 + i, letter: l, blank: blanks.includes(i)})).filter(p => p.letter !== '_');
const col = (c, r0, word, blanks = []) => [...word].map((l, i) => ({r: r0 + i, c, letter: l, blank: blanks.includes(i)})).filter(p => p.letter !== '_');
const boardWith = (...ps) => ps.reduce((b, p) => applyPlacements(b, p), emptyBoard());

test('premium layout sanity', () => {
  assert.equal(PREMIUM[7][7], 'DW'); assert.equal(PREMIUM[0][0], 'TW'); assert.equal(PREMIUM[1][5], 'TL'); assert.equal(PREMIUM[0][3], 'DL');
  const count = t => PREMIUM.flat().filter(x => x === t).length;
  assert.deepEqual([count('TW'), count('DW'), count('TL'), count('DL')], [8, 17, 12, 24]);
});

test('first move on centre DW: CAT = 10', () => {
  const r = scoreMove(emptyBoard(), row(7, 7, 'CAT'));
  assert.ok(r.valid); assert.equal(r.total, 10); assert.deepEqual(r.words.map(w => w.word), ['CAT']);
});

test('first move must cover centre', () => {
  assert.match(scoreMove(emptyBoard(), row(3, 3, 'CAT')).error, /centre/);
});

test('premium only counts when first covered: CAT + S = 6', () => {
  const b = boardWith(row(7, 7, 'CAT'));
  const r = scoreMove(b, [{r: 7, c: 10, letter: 'S'}]);
  assert.equal(r.total, 6); assert.equal(r.words[0].word, 'CATS');
});

test('single tile forming two words with DL counted in both', () => {
  const b = boardWith(row(7, 7, 'CAT'), [{r: 8, c: 9, letter: 'O'}]);
  const r = scoreMove(b, [{r: 8, c: 8, letter: 'A'}]); // (8,8) is DL
  assert.deepEqual(r.words.map(w => [w.word, w.score]), [['AO', 3], ['AA', 3]]);
  assert.equal(r.total, 6);
});

test('parallel play with cross-words', () => {
  const b = boardWith(row(7, 7, 'CAT'));
  const r = scoreMove(b, row(8, 8, 'OX')); // O on DL (8,8)
  assert.deepEqual(r.words.map(w => [w.word, w.score]), [['OX', 10], ['AO', 3], ['TX', 9]]);
  assert.equal(r.total, 22);
});

test('triple-triple bingo through existing tile: QUIETEST = 212', () => {
  const b = boardWith([{r: 0, c: 4, letter: 'T'}]);
  const r = scoreMove(b, row(0, 0, 'QUIE_EST'));
  assert.ok(r.valid, r.error); assert.equal(r.bingo, true);
  assert.equal(r.words[0].word, 'QUIETEST'); assert.equal(r.words[0].score, 162); assert.equal(r.total, 212);
});

test('blank scores 0 even on premium: QUIETEST with blank Q = 122', () => {
  const b = boardWith([{r: 0, c: 4, letter: 'T'}]);
  const r = scoreMove(b, row(0, 0, 'QUIE_EST', [0]));
  assert.equal(r.total, 72 + 50);
});

test('double-double bingo with cross word: RETAINS = 80', () => {
  const b = boardWith(row(7, 5, 'T'));
  const r = scoreMove(b, col(4, 4, 'RETAINS'));
  assert.ok(r.valid, r.error);
  assert.deepEqual(r.words.map(w => [w.word, w.score]), [['RETAINS', 28], ['AT', 2]]);
  assert.equal(r.total, 80);
});

test('TL letter inside a DW word', () => {
  // existing O at (5,6); play Z at (5,5 TL) ... vertical word ZOO? use row 4: (4,4) DW, (5,5) not in row. Row 1: TL at (1,5), DW at (1,1)
  const b = boardWith([{r: 1, c: 3, letter: 'E'}]);
  const r = scoreMove(b, [{r: 1, c: 1, letter: 'Z'}, {r: 1, c: 2, letter: 'O'}, {r: 1, c: 4, letter: 'A'}, {r: 1, c: 5, letter: 'X'}]);
  // Z10 (DW) + O1 + E1 + A1 + X8*3=24 -> 37 * 2 = 74
  assert.equal(r.words[0].word, 'ZOEAX'); assert.equal(r.total, 74);
});

test('pass scores zero', () => { assert.equal(scoreMove(emptyBoard(), []).total, 0); });

test('validation errors', () => {
  const b = boardWith(row(7, 7, 'CAT'));
  assert.match(scoreMove(b, [{r: 6, c: 6, letter: 'A'}, {r: 8, c: 8, letter: 'B'}]).error, /single row/);
  assert.match(scoreMove(b, [{r: 8, c: 7, letter: 'A'}, {r: 8, c: 10, letter: 'B'}]).error, /Gap/);
  assert.match(scoreMove(b, row(2, 2, 'DOG')).error, /connected/);
  assert.match(scoreMove(b, [{r: 7, c: 7, letter: 'A'}]).error, /occupied/);
  assert.match(scoreMove(b, [{r: 8, c: 7, letter: ''}]).error, /Missing/);
});

test('gap filled by existing tile is fine: C_T around A', () => {
  const r = scoreMove(boardWith([{r: 7, c: 7, letter: 'A'}]), [{r: 7, c: 6, letter: 'C'}, {r: 7, c: 8, letter: 'T'}]);
  assert.equal(r.total, 5);
});

test('replay: scores, rotation, adjust, end-game penalty', () => {
  const h = [
    {type: 'play', player: 0, placements: row(7, 7, 'CAT')},   // 10
    {type: 'play', player: 1, placements: row(8, 8, 'OX')},    // 22
    {type: 'pass', player: 0},
    {type: 'play', player: 1, placements: [{r: 7, c: 10, letter: 'S'}]}, // CATS 6 + XS? (8,10) empty -> 6
    {type: 'adjust', player: 0, delta: 5},
    {type: 'penalty', racks: ['QE', ''], outPlayer: 1},
  ];
  const g = replay(2, h);
  assert.deepEqual(g.scores, [10 + 5 - 11, 22 + 6 + 11]);
  assert.equal(g.current, 0);
  assert.equal(g.board[7][10].letter, 'S');
  assert.equal(rackValue('Q?z'), 20);
});

test('game stats + awards', () => {
  const h = [
    {type: 'play', player: 0, placements: row(7, 7, 'CAT')},          // 10, bonus 1 (DW centre)
    {type: 'play', player: 1, placements: row(8, 8, 'OX')},           // 22: OX(10, bonus1) AO 3 TX 9
    {type: 'play', player: 0, placements: [{r: 7, c: 10, letter: 'S'}]}, // CATS 6
    {type: 'pass', player: 1},
    {type: 'play', player: 0, placements: col(4, 4, 'RETAINS').map(p => p)}, // invalid here (not connected) -> 0
  ];
  const s = gameStats(2, h);
  assert.deepEqual(s.scores, [16, 22]);
  assert.equal(s.players[0].turns, 3); assert.equal(s.players[0].plays, 2);
  assert.equal(s.players[0].avgPerTurn, 5.3); assert.equal(s.players[1].avgPerTurn, 11);
  assert.equal(s.players[0].longestWord.word, 'CATS');
  assert.deepEqual(s.awards.longestWord.players, [0]);
  assert.deepEqual(s.awards.bestTurn, {players: [1], value: 22, detail: 'OX, AO, TX'});
  assert.deepEqual(s.awards.bestWord, {players: [0, 1], value: 10, detail: 'CAT / OX'});
  assert.equal(s.awards.mostBonusTotal.value, 1);
  assert.equal(s.awards.mostBingos, null);
  // bingo + double-double counts 2 bonus squares in one word
  const s2 = gameStats(2, [{type: 'play', player: 0, placements: row(7, 1, 'RETAINS')}]); // (7,3) DL, (7,7) DW
  assert.equal(s2.players[0].bingos, 1); assert.equal(s2.players[0].bestBonusWord.bonus, 2);
  assert.equal(s2.awards.mostBonusInWord.detail, 'RETAINS');
});

test('end game: rack values from letters or numbers', () => {
  assert.equal(rackValue('QE?'), 11); assert.equal(rackValue('12'), 12); assert.equal(rackValue('q 3'), 13); assert.equal(rackValue(''), 0);
});

test('end game: someone went out gets others\' leftovers (official rule)', () => {
  const {deltas} = endGameAdjustments(['', 'QE', 'ZA?'], 0);  // 11 + 11
  assert.deepEqual(deltas, [22, -11, -11]);
  // whatever is typed for the player who went out is ignored
  assert.deepEqual(endGameAdjustments(['X', '5'], 0).deltas, [5, -5]);
});

test('end game: nobody went out -> everyone just subtracts', () => {
  assert.deepEqual(endGameAdjustments(['QE', '7', 'A'], -1).deltas, [-11, -7, -1]);
  const g = replay(2, [{type: 'play', player: 0, placements: row(7, 7, 'CAT')}, {type: 'penalty', racks: ['E', '10'], outPlayer: -1}]);
  assert.deepEqual(g.scores, [9, -10]);
});

test('consecutive scoreless turns (auto-end at 2x players)', () => {
  const pass = p => ({type: 'pass', player: p});
  const h = [{type: 'play', player: 0, placements: row(7, 7, 'CAT')}, pass(1), pass(2), {type: 'adjust', player: 0, delta: 3}, pass(0), pass(1), pass(2)];
  assert.equal(consecutiveScoreless(3, h), 5);
  h.push(pass(0)); assert.equal(consecutiveScoreless(3, h), 6); // 6 >= 2*3 -> prompt
  // an invalid (0-point) play counts as scoreless; a scoring play resets
  assert.equal(consecutiveScoreless(2, [pass(0), {type: 'play', player: 1, placements: row(2, 2, 'DOG')}]), 2);
  assert.equal(consecutiveScoreless(2, [pass(0), {type: 'play', player: 1, placements: row(7, 7, 'CAT')}]), 0);
});
