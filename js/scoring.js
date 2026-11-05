// Pure Scrabble scoring logic (no DOM). Works in browser (ES module) and Node.
export const SIZE = 15;
export const LETTER_VALUES = {A:1,B:3,C:3,D:2,E:1,F:4,G:2,H:4,I:1,J:8,K:5,L:1,M:3,N:1,O:1,P:3,Q:10,R:1,S:1,T:1,U:1,V:4,W:4,X:8,Y:4,Z:10};

const TW = ['0,0','0,7','0,14','7,0','7,14','14,0','14,7','14,14'];
const DW = ['1,1','2,2','3,3','4,4','10,10','11,11','12,12','13,13','1,13','2,12','3,11','4,10','10,4','11,3','12,2','13,1','7,7'];
const TL = ['1,5','1,9','5,1','5,5','5,9','5,13','9,1','9,5','9,9','9,13','13,5','13,9'];
const DL = ['0,3','0,11','2,6','2,8','3,0','3,7','3,14','6,2','6,6','6,8','6,12','7,3','7,11','8,2','8,6','8,8','8,12','11,0','11,7','11,14','12,6','12,8','14,3','14,11'];
export const PREMIUM = (() => {
  const m = Array.from({length: SIZE}, () => Array(SIZE).fill(''));
  const set = (list, v) => list.forEach(k => { const [r, c] = k.split(',').map(Number); m[r][c] = v; });
  set(TW, 'TW'); set(DW, 'DW'); set(TL, 'TL'); set(DL, 'DL');
  return m;
})();

export function emptyBoard() { return Array.from({length: SIZE}, () => Array(SIZE).fill(null)); }
export function cloneBoard(b) { return b.map(row => row.map(c => (c ? {...c} : null))); }
export function tileValue(t) { return t.blank ? 0 : (LETTER_VALUES[t.letter] || 0); }
/** Leftover rack value: letters (A-Z, ? = blank 0) and/or numbers, summed. "QE?" -> 11, "12" -> 12, "Q 3" -> 13. */
export function rackValue(str) {
  str = String(str ?? '').toUpperCase();
  const nums = (str.match(/\d+/g) || []).reduce((a, n) => a + parseInt(n, 10), 0);
  return nums + [...str.replace(/\d+/g, '')].reduce((s, ch) => s + (LETTER_VALUES[ch] || 0), 0);
}

/** Official end-game adjustment: everyone loses own leftovers; the player who went out (or -1) gains everyone else's. */
export function endGameAdjustments(racks, outPlayer) {
  const vals = racks.map((r, i) => (i === outPlayer ? 0 : rackValue(r)));
  const deltas = vals.map(v => -v);
  if (outPlayer >= 0) deltas[outPlayer] = vals.reduce((a, b) => a + b, 0);
  return {vals, deltas};
}

/** Number of consecutive scoreless turns (passes/exchanges or 0-point plays) at the end of history; adjustments ignored. */
export function consecutiveScoreless(nPlayers, history) {
  const {turns} = replay(nPlayers, history);
  let n = 0;
  for (let i = turns.length - 1; i >= 0; i--) {
    const t = turns[i];
    if (t.type === 'adjust') continue;
    if (t.type === 'pass' || (t.type === 'play' && !(t.result.valid && t.result.total > 0))) n++;
    else break;
  }
  return n;
}
function isEmptyBoard(b) { return b.every(r => r.every(c => !c)); }
const inb = (r, c) => r >= 0 && c >= 0 && r < SIZE && c < SIZE;

/**
 * Score a move.
 * board: 15x15 array of null | {letter, blank}
 * placements: [{r, c, letter, blank}]
 * returns {valid, error, words:[{word, score, cells}], bingo, total}
 */
export function scoreMove(board, placements) {
  const res = {valid: false, error: null, words: [], bingo: false, total: 0};
  if (!placements.length) { res.valid = true; return res; }
  const seen = new Set();
  for (const p of placements) {
    if (!inb(p.r, p.c)) { res.error = 'Tile off board'; return res; }
    const k = p.r + ',' + p.c;
    if (seen.has(k)) { res.error = 'Two tiles on one square'; return res; }
    seen.add(k);
    if (board[p.r][p.c]) { res.error = `Square ${cellName(p.r, p.c)} already occupied`; return res; }
    if (!p.letter || !/^[A-Z]$/.test(p.letter)) { res.error = `Missing letter at ${cellName(p.r, p.c)}`; return res; }
  }
  const nb = cloneBoard(board);
  const isNew = Array.from({length: SIZE}, () => Array(SIZE).fill(false));
  for (const p of placements) { nb[p.r][p.c] = {letter: p.letter, blank: !!p.blank}; isNew[p.r][p.c] = true; }

  const rows = new Set(placements.map(p => p.r)), cols = new Set(placements.map(p => p.c));
  let dir; // [dr, dc] of main word
  if (rows.size === 1 && cols.size === 1) dir = null;
  else if (rows.size === 1) dir = [0, 1];
  else if (cols.size === 1) dir = [1, 0];
  else { res.error = 'Tiles not in a single row or column'; return res; }

  if (dir) { // contiguity (gaps may be filled by existing tiles)
    const fixed = dir[0] === 0 ? 'r' : 'c', vary = dir[0] === 0 ? 'c' : 'r';
    const vals = placements.map(p => p[vary]);
    const lo = Math.min(...vals), hi = Math.max(...vals), f = placements[0][fixed];
    for (let i = lo; i <= hi; i++) {
      const [r, c] = fixed === 'r' ? [f, i] : [i, f];
      if (!nb[r][c]) { res.error = 'Gap in the played word'; return res; }
    }
  }

  const wordAt = (r, c, dr, dc) => {
    while (inb(r - dr, c - dc) && nb[r - dr][c - dc]) { r -= dr; c -= dc; }
    const cells = [];
    while (inb(r, c) && nb[r][c]) { cells.push([r, c]); r += dr; c += dc; }
    return cells;
  };
  const scoreCells = cells => {
    let sum = 0, wm = 1;
    for (const [r, c] of cells) {
      let v = tileValue(nb[r][c]);
      if (isNew[r][c]) {
        const pm = PREMIUM[r][c];
        if (pm === 'DL') v *= 2; else if (pm === 'TL') v *= 3;
        else if (pm === 'DW') wm *= 2; else if (pm === 'TW') wm *= 3;
      }
      sum += v;
    }
    return sum * wm;
  };
  const addWord = cells => {
    if (cells.length < 2) return;
    const bonus = cells.filter(([r, c]) => isNew[r][c] && PREMIUM[r][c]).length;
    res.words.push({word: cells.map(([r, c]) => nb[r][c].letter).join(''), score: scoreCells(cells), cells, start: cellName(...cells[0]), bonus});
  };
  const p0 = placements[0];
  if (dir) {
    addWord(wordAt(p0.r, p0.c, dir[0], dir[1]));
    const [cr, cc] = [dir[1], dir[0]];
    for (const p of placements) addWord(wordAt(p.r, p.c, cr, cc));
  } else {
    addWord(wordAt(p0.r, p0.c, 0, 1));
    addWord(wordAt(p0.r, p0.c, 1, 0));
  }
  if (!res.words.length) { res.error = 'No word of 2+ letters formed'; return res; }

  if (isEmptyBoard(board)) {
    if (!seen.has('7,7')) { res.error = 'First word must cover the centre star (H8)'; return res; }
  } else {
    const touchesOld = res.words.some(w => w.cells.some(([r, c]) => !isNew[r][c]));
    if (!touchesOld) { res.error = 'Word not connected to existing tiles'; return res; }
  }
  res.bingo = placements.length === 7;
  res.bonusSquares = placements.filter(p => PREMIUM[p.r][p.c]).length;
  res.total = res.words.reduce((s, w) => s + w.score, 0) + (res.bingo ? 50 : 0);
  res.valid = true;
  return res;
}

export function cellName(r, c) { return String.fromCharCode(65 + c) + (r + 1); }

export function applyPlacements(board, placements) {
  const nb = cloneBoard(board);
  for (const p of placements) nb[p.r][p.c] = {letter: p.letter, blank: !!p.blank};
  return nb;
}

/**
 * Replay a full game history -> {board, scores, current, turns:[{...entry, result}]}
 * entry types: {type:'play', player, placements} | {type:'pass', player} | {type:'adjust', player, delta, note}
 *              | {type:'penalty', racks:[string per player], outPlayer: index|-1}
 */
export function replay(nPlayers, history) {
  let board = emptyBoard();
  const scores = Array(nPlayers).fill(0);
  const turns = [];
  let current = 0;
  for (const e of history) {
    if (e.type === 'play') {
      const result = scoreMove(board, e.placements);
      if (result.valid) { board = applyPlacements(board, e.placements); scores[e.player] += result.total; }
      else if (e.placements.every(p => !board[p.r]?.[p.c])) board = applyPlacements(board, e.placements.filter(p => /^[A-Z]$/.test(p.letter || '')));
      turns.push({...e, result});
      current = (e.player + 1) % nPlayers;
    } else if (e.type === 'pass') {
      turns.push({...e}); current = (e.player + 1) % nPlayers;
    } else if (e.type === 'adjust') {
      scores[e.player] += e.delta; turns.push({...e});
    } else if (e.type === 'penalty') {
      const {vals, deltas} = endGameAdjustments(e.racks, e.outPlayer);
      deltas.forEach((d, i) => { scores[i] += d; });
      turns.push({...e, vals, deltas});
    }
  }
  return {board, scores, current, turns};
}

/**
 * Per-player stats + game awards, derived from history (every turn is replayed).
 * players[i]: {turns, plays, passes, playPoints, avgPerTurn, bingos, bonusSquares, longestWord, bestTurn, bestWord, bestBonusWord}
 * awards: {longestWord, bestTurn, bestWord, mostBonusInWord, mostBonusTotal, mostBingos, bestAverage} -> {players:[idx], value, detail}
 */
export function gameStats(nPlayers, history) {
  const g = replay(nPlayers, history);
  const P = Array.from({length: nPlayers}, () => ({turns: 0, plays: 0, passes: 0, playPoints: 0, avgPerTurn: 0, bingos: 0, bonusSquares: 0,
    longestWord: null, bestTurn: null, bestWord: null, bestBonusWord: null}));
  g.turns.forEach((t, i) => {
    const p = P[t.player];
    if (t.type === 'pass') { p.turns++; p.passes++; return; }
    if (t.type !== 'play') return;
    p.turns++;
    if (!t.result.valid) return;
    p.plays++; p.playPoints += t.result.total;
    if (t.result.bingo) p.bingos++;
    p.bonusSquares += t.result.bonusSquares;
    const main = t.result.words.reduce((a, w) => (w.score > a.score ? w : a));
    if (!p.bestTurn || t.result.total > p.bestTurn.score) p.bestTurn = {score: t.result.total, turn: i + 1, words: t.result.words.map(w => w.word), bingo: t.result.bingo, main: main.word};
    for (const w of t.result.words) {
      if (!p.longestWord || w.word.length > p.longestWord.word.length) p.longestWord = {word: w.word, length: w.word.length, turn: i + 1};
      if (!p.bestWord || w.score > p.bestWord.score) p.bestWord = {word: w.word, score: w.score, turn: i + 1};
      if (w.bonus && (!p.bestBonusWord || w.bonus > p.bestBonusWord.bonus)) p.bestBonusWord = {word: w.word, bonus: w.bonus, turn: i + 1};
    }
  });
  P.forEach(p => { p.avgPerTurn = p.turns ? Math.round(p.playPoints / p.turns * 10) / 10 : 0; });
  const award = (val, detail) => {
    const vals = P.map(val); const best = Math.max(...vals);
    if (!(best > 0)) return null;
    const players = vals.map((v, i) => (v === best ? i : -1)).filter(i => i >= 0);
    return {players, value: best, detail: players.map(i => detail(P[i])).join(' / ')};
  };
  const awards = {
    longestWord: award(p => p.longestWord?.length || 0, p => p.longestWord.word),
    bestTurn: award(p => p.bestTurn?.score || 0, p => p.bestTurn.words.join(', ')),
    bestWord: award(p => p.bestWord?.score || 0, p => p.bestWord.word),
    mostBonusInWord: award(p => p.bestBonusWord?.bonus || 0, p => p.bestBonusWord.word),
    mostBonusTotal: award(p => p.bonusSquares, () => ''),
    mostBingos: award(p => p.bingos, () => ''),
    bestAverage: award(p => p.avgPerTurn, () => ''),
  };
  return {scores: g.scores, players: P, awards};
}

/** Rotate every placement in a history clockwise by k quarter-turns (scores unchanged: premium layout is 4-fold symmetric). */
export function rotateHistory(history, k) {
  k = ((k % 4) + 4) % 4;
  const rot = (r, c) => { for (let i = 0; i < k; i++) [r, c] = [c, SIZE - 1 - r]; return [r, c]; };
  return history.map(e => (e.type === 'play' ? {...e, placements: e.placements.map(p => { const [r, c] = rot(p.r, p.c); return {...p, r, c}; })} : e));
}
