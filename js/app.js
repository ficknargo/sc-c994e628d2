import {SIZE, PREMIUM, scoreMove, replay, cellName, rackValue, gameStats, endGameAdjustments, consecutiveScoreless, rotateHistory} from './scoring.js';
import {warp, cellFeatures, diffScores, pickNewCells, homography, applyH, packFeatures, unpackFeatures, rotCell, normalizeCorners, findRotation} from './geom.js';
import {grabFrame, readCells, readBoard} from './vision.js';

const $ = s => document.querySelector(s);
const GKEY = 'scrabblecam.game.v1', SKEY = 'scrabblecam.settings.v1';
const TEST = new URLSearchParams(location.search).has('test');

let game = load(GKEY);           // {players:[], history:[], corners:[{x,y}] normalized | null, baseline: packed | null}
let settings = load(SKEY) || {};
let baseline = game?.baseline ? unpackFeatures(game.baseline) : null;
let aligning = false, busy = false, stream = null;

function load(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } }
function saveGame() {
  game.baseline = baseline ? packFeatures(baseline) : null;
  // keep the "baseline before" snapshot only on the last play (for undo)
  game.history.forEach((e, i) => { if (i < game.history.length - 1) delete e.baselineBefore; });
  localStorage.setItem(GKEY, JSON.stringify(game));
}
function state() { return replay(game.players.length, game.history); }
function toast(msg, ms = 2600) { const t = $('#toast'); t.textContent = msg; t.classList.remove('hidden'); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.add('hidden'), ms); }
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

/* ---------------- Setup ---------------- */
let names = game?.players?.slice() || ['Player 1', 'Player 2'];
function renderNames() {
  $('#names').innerHTML = names.map((n, i) => `<input data-i="${i}" value="${esc(n)}" placeholder="Player ${i + 1}" autocomplete="off">`).join('');
  $('#names').querySelectorAll('input').forEach(inp => inp.oninput = () => { names[+inp.dataset.i] = inp.value; });
  $('#resumeGame').classList.toggle('hidden', !game);
  $('#count').querySelectorAll('button').forEach(b => b.classList.toggle('primary', +b.dataset.n === names.length));
}
$('#count').onclick = e => {
  const n = +e.target.dataset.n; if (!n) return;
  while (names.length < n) names.push('Player ' + (names.length + 1));
  names.length = n; renderNames();
};
$('#startGame').onclick = () => {
  if (game && game.history.length && !confirm('Start a new game? The current one is discarded.')) return;
  const corners = game?.corners || null;
  game = {players: names.map((n, i) => n.trim() || 'Player ' + (i + 1)), history: [], corners, baseline: null};
  baseline = null; saveGame(); enterGame(true);
};
$('#resumeGame').onclick = () => enterGame(false);

/* ---------------- Camera + overlay ---------------- */
const video = $('#video'), overlay = $('#overlay'), testSrc = $('#testSrc');
function source() { return TEST ? testSrc : video; }
function srcSize() { const s = source(); return {w: s.videoWidth || s.width, h: s.videoHeight || s.height}; }

async function startCamera() {
  if (TEST) { layoutCam(); return; }
  if (stream) return;
  const msg = $('#camMsg');
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: {ideal: 'environment'}, width: {ideal: 1920}, height: {ideal: 1440}}});
    video.srcObject = stream; await video.play().catch(() => {});
    msg.classList.add('hidden');
  } catch (e) {
    msg.textContent = 'Camera unavailable: ' + e.message + '. Allow camera access (Safari: aA › Website Settings › Camera) and reload.';
    msg.classList.remove('hidden');
  }
}
video.addEventListener('loadedmetadata', layoutCam);
window.addEventListener('resize', layoutCam);
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && stream && video.paused) video.play().catch(() => {}); });

async function setFrame(url) { // test hook: use an image instead of the camera
  const img = new Image(); img.src = url; await img.decode();
  testSrc.width = img.naturalWidth; testSrc.height = img.naturalHeight;
  testSrc.getContext('2d').drawImage(img, 0, 0); testSrc.classList.remove('hidden'); layoutCam();
}

function layoutCam() {
  const {w, h} = srcSize(); if (!w || !game) return;
  const box = $('#camBox'), maxH = window.innerHeight * 0.58, availW = $('#camWrap').clientWidth;
  let bw = availW, bh = bw * h / w; if (bh > maxH) { bh = maxH; bw = bh * w / h; }
  box.style.width = bw + 'px'; box.style.height = bh + 'px';
  const dpr = devicePixelRatio || 1; overlay.width = bw * dpr; overlay.height = bh * dpr;
  if (!game.corners) defaultCorners();
  drawOverlay();
}
function defaultCorners() {
  const {w, h} = srcSize(), side = Math.min(w, h) * 0.8, x0 = (w - side) / 2, y0 = (h - side) / 2;
  game.corners = [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]].map(([x, y]) => ({x: x / w, y: y / h}));
}
function cornersPx(w, h) { return game.corners.map(p => ({x: p.x * w, y: p.y * h})); }
// Orientation: canonical board cell = rotCell(view cell, game.rot); view = rotCell(canon, 4 - rot)
const K = () => game.rot || 0;
const toCanon = (r, c) => rotCell(r, c, K());
const toView = (r, c) => rotCell(r, c, 4 - K());

function drawOverlay() {
  const ctx = overlay.getContext('2d'), W = overlay.width, H = overlay.height;
  ctx.clearRect(0, 0, W, H); if (!game?.corners) return;
  const q = cornersPx(W, H), Hm = homography([{x: 0, y: 0}, {x: 15, y: 0}, {x: 15, y: 15}, {x: 0, y: 15}], q);
  ctx.strokeStyle = aligning ? 'rgba(255,47,208,.95)' : 'rgba(255,214,64,.85)';
  for (let i = 0; i <= 15; i++) {
    const seg = (a, b) => { const p = applyH(Hm, a.x, a.y), r = applyH(Hm, b.x, b.y); ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(r.x, r.y); ctx.stroke(); };
    ctx.lineWidth = Math.max(1, W / (i % 15 ? 450 : 150));
    seg({x: i, y: 0}, {x: i, y: 15}); seg({x: 0, y: i}, {x: 15, y: i});
  }
  // mark tiles already scored so misalignment is obvious
  const st = state(); ctx.fillStyle = 'rgba(126,224,129,.35)';
  for (let R = 0; R < 15; R++) for (let C = 0; C < 15; C++) if (st.board[R][C]) {
    const [r, c] = toView(R, C);
    const pts = [[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]].map(([x, y]) => applyH(Hm, x, y));
    ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.fill();
  }
  if (aligning) q.forEach(p => { ctx.beginPath(); ctx.arc(p.x, p.y, 16 * (devicePixelRatio || 1), 0, 7); ctx.fillStyle = 'rgba(255,47,208,.5)'; ctx.fill(); ctx.stroke(); });
}

let drag = -1;
overlay.addEventListener('pointerdown', e => {
  if (!aligning) return;
  const r = overlay.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  let best = -1, bd = 1e9;
  game.corners.forEach((p, i) => { const d = Math.hypot((p.x - x) * r.width, (p.y - y) * r.height); if (d < bd) { bd = d; best = i; } });
  if (bd < 80) { drag = best; overlay.setPointerCapture(e.pointerId); }
});
overlay.addEventListener('pointermove', e => {
  if (drag < 0) return;
  const r = overlay.getBoundingClientRect();
  game.corners[drag] = {x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))};
  drawOverlay();
});
overlay.addEventListener('pointerup', () => { drag = -1; });

$('#alignBtn').onclick = () => {
  aligning = !aligning;
  $('#alignBtn').textContent = aligning ? 'Done aligning' : 'Align corners';
  $('#status').textContent = aligning ? 'Drag the 4 dots onto the outer corners of the 15×15 grid' : '';
  if (!aligning) { realign(); saveGame(); }
  drawOverlay();
};

/* ---------------- Capture + recognition ---------------- */
function capture() {
  const frame = grabFrame(source());
  const corners = cornersPx(frame.width, frame.height);
  const feats = cellFeatures(warp(frame, corners, 600));
  return {frame, corners, feats};
}
function realign() {
  // corners may have been dragged in any order: make them clockwise from top-left (no mirror)
  const {w, h} = srcSize();
  if (w) game.corners = normalizeCorners(cornersPx(w, h)).map(p => ({x: p.x / w, y: p.y / h}));
  const hasTiles = state().board.flat().some(Boolean);
  if (baseline && hasTiles && w) {
    // phone may now be on another side: pick the rotation that best matches the stored board
    const feats = capture().feats, {d} = findRotation(baseline, feats);
    if (d) { game.rot = (K() - d + 4) % 4; toast('Board orientation updated (' + d * 90 + '°)'); }
    baseline = feats; $('#status').textContent = 'Board synced ✓';
  } else syncBaseline();
  drawOverlay();
}
function syncBaseline() { // the board as seen now == app's board
  try { if (srcSize().w) { baseline = capture().feats; $('#status').textContent = 'Board synced ✓'; } } catch (e) { console.warn(e); }
}

$('#scoreBtn').onclick = async () => {
  if (busy) return;
  if (!srcSize().w) return toast('Camera not ready');
  if (aligning) $('#alignBtn').click();
  busy = true; const btn = $('#scoreBtn'); btn.disabled = true; btn.textContent = 'Reading board…';
  try { await scoreTurn(); } catch (e) { console.error(e); toast('Error: ' + e.message, 5000); }
  finally { busy = false; btn.disabled = false; btn.textContent = 'Score turn'; }
};

async function scoreTurn() {
  const cap = capture(), st = state(), boardEmpty = !st.board.flat().some(Boolean);
  const occupied = Array.from({length: 225}, (_, i) => { const [r, c] = toCanon(Math.floor(i / 15), i % 15); return !!st.board[r][c]; }); // view-indexed
  const hasKey = !!settings.apiKey;
  let cells = []; // view coords
  if (baseline) cells = pickNewCells(diffScores(baseline, cap.feats), occupied).cells;
  const canonP = (r, c, extra) => { const [R, C] = rotCell(...toCanon(r, c), extra || 0); return {r: R, c: C}; };
  let placements = cells.map(c => ({...canonP(c.r, c.c), letter: '', blank: false}));
  let note = '';
  if (hasKey) {
    const cfg = {apiKey: settings.apiKey, model: settings.model || 'gpt-4.1-mini', baseUrl: settings.baseUrl};
    if (cells.length) {
      if (TEST) window.__lastCells = cells.map(c => [c.r, c.c]);
      const {letters, extra} = await readCells(cfg, cap.frame, cap.corners, cells, K());
      const ex = boardEmpty ? extra : 0;  // first word decides which way is "up"; later turns keep the stored orientation
      placements = cells.map((c, i) => ({...canonP(c.r, c.c, ex), letter: letters[i] === '?' ? '' : letters[i], blank: letters[i] === '?', _drop: !letters[i]})).filter(p => !p._drop).map(({_drop, ...p}) => p);
      if (ex) game.rot = (K() + ex) % 4;
    } else {
      const known = st.board.map(row => row.map(t => t ? t.letter : '.').join(''));
      const {rows, extra} = await readBoard(cfg, cap.frame, cap.corners, known, K());
      const ex = boardEmpty ? extra : 0;
      placements = [];
      rows.forEach((row, r0) => [...row].forEach((ch, c0) => {
        const [r, c] = rotCell(r0, c0, ex);
        if (!st.board[r][c] && /[A-Z?]/.test(ch)) placements.push({r, c, letter: ch === '?' ? '' : ch, blank: ch === '?'});
      }));
      if (ex) game.rot = (K() + ex) % 4;
    }
  } else note = 'No API key: type the letter for each highlighted tile. Words read the wrong way? Tap ⟳ Rotate.';
  if (!placements.length) {
    toast(baseline || hasKey ? 'No new tiles found. Check alignment, or tap squares to add them.' : 'Tap the squares where tiles were placed.', 4000);
  }
  const needsInput = !placements.length || placements.some(p => !p.letter);
  const res = scoreMove(st.board, placements);
  const entry = {type: 'play', player: st.current, placements};
  if (!needsInput && res.valid) return commitPlay(entry, cap.feats, res);
  if (!note) note = placements.some(p => p.blank && !p.letter) ? 'Blank tile: pick the letter it stands for (Blank stays on).' : (res.error || 'Check the tiles.');
  openEditor({entry, index: -1, note, onSave: e => commitPlay(e, cap.feats, scoreMove(state().board, e.placements))});
}

function commitPlay(entry, feats, res) {
  entry.summary = {total: res.total, bingo: res.bingo, bonusSquares: res.bonusSquares, words: res.words.map(w => ({word: w.word, score: w.score, bonus: w.bonus}))};
  entry.baselineBefore = baseline ? packFeatures(baseline) : null;
  game.history.push(entry); baseline = feats; saveGame(); render(); checkAutoEnd();
  flash(game.players[entry.player], res);
}

function flash(name, res) {
  const f = $('#flash');
  f.innerHTML = `<div>${esc(name)}</div><div class="pts">+${res.total}</div><div class="w">${res.words.map(w => `${w.word} ${w.score}`).join(' · ')}${res.bingo ? ' · BINGO +50' : ''}</div>`;
  f.classList.remove('hidden'); clearTimeout(flash.t); flash.t = setTimeout(() => f.classList.add('hidden'), 3500);
  f.onclick = () => f.classList.add('hidden');
}

/* ---------------- Game screen ---------------- */
function enterGame(fresh) {
  $('#setup').classList.add('hidden'); $('#game').classList.remove('hidden');
  const needAlign = fresh && !game.corners;
  render(); startCamera().then(() => { layoutCam(); if (fresh) setTimeout(() => { if (!state().board.flat().some(Boolean)) syncBaseline(); }, 1200); });
  if (needAlign) setTimeout(() => { if (!aligning) $('#alignBtn').click(); }, 400);
}
function render() {
  const st = state();
  $('#turnLabel').textContent = game.players[st.current] + "'s turn";
  $('#scores').innerHTML = game.players.map((n, i) => `<div class="pl ${i === st.current ? 'cur' : ''}" data-i="${i}"><div class="n">${esc(n)}</div><div class="s">${st.scores[i]}</div></div>`).join('');
  $('#scores').querySelectorAll('.pl').forEach(el => el.onclick = () => adjust(+el.dataset.i));
  drawOverlay();
}
function adjust(i) {
  const v = prompt(`Adjust ${game.players[i]}'s score by (e.g. -5 or 12):`);
  const d = parseInt(v, 10); if (!v || isNaN(d)) return;
  game.history.push({type: 'adjust', player: i, delta: d}); saveGame(); render();
}
$('#undoBtn').onclick = () => {
  const e = game.history[game.history.length - 1]; if (!e) return toast('Nothing to undo');
  if (!confirm('Undo: ' + describe(e, state().turns.at(-1)) + '?')) return;
  game.history.pop();
  if (e.type === 'play') baseline = e.baselineBefore ? unpackFeatures(e.baselineBefore) : null;
  saveGame(); render(); toast('Undone. Leave/restore tiles, then Score turn again.');
};
$('#passBtn').onclick = () => { const st = state(); if (!confirm(game.players[st.current] + ' passes / exchanges?')) return; game.history.push({type: 'pass', player: st.current}); saveGame(); render(); checkAutoEnd(); };

function describe(e, t) {
  const n = game.players[e.player] ?? '';
  if (e.type === 'play') return t?.result?.valid ? `${n}: ${t.result.words.map(w => w.word).join(', ')} +${t.result.total}` : `${n}: ⚠ ${t?.result?.error || 'invalid'}`;
  if (e.type === 'pass') return `${n}: pass`;
  if (e.type === 'adjust') return `${n}: adjust ${e.delta > 0 ? '+' : ''}${e.delta}`;
  if (e.type === 'penalty') return `End of game: ${game.players.map((p, i) => `${p} ${(t?.deltas?.[i] ?? 0) >= 0 ? '+' : ''}${t?.deltas?.[i] ?? ''}`).join(', ')}`;
  return e.type;
}

/* ---------------- History ---------------- */
$('#historyBtn').onclick = () => {
  const st = state();
  $('#histList').innerHTML = st.turns.map((t, i) => `<li data-i="${i}"><span>${i + 1}. ${esc(describe(game.history[i], t))}</span><b>›</b></li>`).reverse().join('') || '<li>No turns yet</li>';
  $('#histList').querySelectorAll('li[data-i]').forEach(li => li.onclick = () => editHistory(+li.dataset.i));
  $('#history').classList.remove('hidden');
};
function editHistory(i) {
  const e = game.history[i];
  if (e.type === 'play') return openEditor({entry: JSON.parse(JSON.stringify(e)), index: i, note: 'Tap a square, then a letter. Tap an empty square to add a tile.', onSave: ne => { game.history[i] = {...game.history[i], player: ne.player, placements: ne.placements}; saveGame(); render(); $('#historyBtn').click(); }});
  if (e.type === 'adjust') { const v = prompt('Adjustment (blank = delete):', e.delta); if (v === null) return; if (v.trim() === '') game.history.splice(i, 1); else e.delta = parseInt(v, 10) || 0; }
  else if (confirm('Delete this entry?')) game.history.splice(i, 1);
  else return;
  saveGame(); render(); $('#historyBtn').click();
}
document.querySelectorAll('.closeSheet').forEach(b => b.onclick = () => b.closest('.sheet').classList.add('hidden'));

/* ---------------- Editor ---------------- */
let ed = null;
function openEditor(opts) {
  ed = {...opts, sel: null};
  const before = replay(game.players.length, opts.index >= 0 ? game.history.slice(0, opts.index) : game.history);
  ed.board = before.board;
  $('#edTitle').textContent = opts.index >= 0 ? `Edit turn ${opts.index + 1}` : 'New turn';
  $('#edPlayer').innerHTML = game.players.map((p, i) => `<option value="${i}" ${i === opts.entry.player ? 'selected' : ''}>${esc(p)}</option>`).join('');
  $('#edHint').textContent = opts.note || '';
  $('#edDelete').classList.toggle('hidden', opts.index < 0);
  ed.sel = opts.entry.placements.find(p => !p.letter) || null;
  if (ed.sel) ed.sel = {r: ed.sel.r, c: ed.sel.c};
  $('#editor').classList.remove('hidden');
  renderEditor();
}
function edTile(r, c) { return ed.entry.placements.find(p => p.r === r && p.c === c); }
function renderEditor() {
  let html = '';
  for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
    const old = ed.board[r][c], nw = edTile(r, c), sel = ed.sel && ed.sel.r === r && ed.sel.c === c;
    const cls = ['cell', old ? 'old' : nw ? 'new' : PREMIUM[r][c], (old || nw)?.blank ? 'blank' : '', sel ? 'sel' : '', nw && !nw.letter ? 'ask' : ''].join(' ');
    html += `<div class="${cls}" data-r="${r}" data-c="${c}">${old ? old.letter : nw ? nw.letter : ''}</div>`;
  }
  $('#edGrid').innerHTML = html;
  const res = scoreMove(ed.board, ed.entry.placements);
  $('#edResult').innerHTML = res.valid && ed.entry.placements.length
    ? `${res.words.map(w => `<b>${w.word}</b> ${w.score}`).join(' · ')}${res.bingo ? ' · BINGO +50' : ''} = <b>${res.total}</b>`
    : `<span class="err">${esc(res.error || 'Tap squares to add the tiles played')}</span>`;
  const t = ed.sel && edTile(ed.sel.r, ed.sel.c);
  $('#keys').innerHTML = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'].map(l => `<button data-k="${l}" class="${t?.letter === l ? 'on' : ''}">${l}</button>`).join('') +
    `<button data-k="blank" class="${t?.blank ? 'on' : ''}">Blank</button><button data-k="del">⌫</button>`;
  $('#keys').style.visibility = ed.sel ? 'visible' : 'hidden';
}
$('#edGrid').onclick = e => {
  const el = e.target.closest('.cell'); if (!el) return;
  const r = +el.dataset.r, c = +el.dataset.c;
  if (ed.board[r][c]) return toast('That tile was played on an earlier turn');
  ed.sel = {r, c}; renderEditor();
};
$('#keys').onclick = e => {
  const k = e.target.dataset?.k; if (!k || !ed.sel) return;
  const {r, c} = ed.sel; let t = edTile(r, c);
  if (k === 'del') { ed.entry.placements = ed.entry.placements.filter(p => p !== t); }
  else if (k === 'blank') { if (!t) { t = {r, c, letter: '', blank: true}; ed.entry.placements.push(t); } else t.blank = !t.blank; }
  else {
    if (!t) { t = {r, c, letter: k, blank: false}; ed.entry.placements.push(t); } else t.letter = k;
    const next = ed.entry.placements.find(p => !p.letter); if (next) ed.sel = {r: next.r, c: next.c};
  }
  renderEditor();
};
$('#edRotate').onclick = () => {
  // relabel the whole board a quarter-turn clockwise (scores unchanged), and keep the camera mapping consistent
  game.history = rotateHistory(game.history, 1); game.rot = (K() + 1) % 4;
  ed.entry.placements = rotateHistory([ed.entry], 1)[0].placements;
  if (ed.sel) { const [r, c] = rotCell(ed.sel.r, ed.sel.c, 1); ed.sel = {r, c}; }
  ed.board = replay(game.players.length, ed.index >= 0 ? game.history.slice(0, ed.index) : game.history).board;
  saveGame(); render(); renderEditor();
};
$('#edPlayer').onchange = e => { ed.entry.player = +e.target.value; };
$('#edCancel').onclick = () => { $('#editor').classList.add('hidden'); ed = null; };
$('#edDelete').onclick = () => { if (!confirm('Delete this turn?')) return; game.history.splice(ed.index, 1); saveGame(); render(); $('#editor').classList.add('hidden'); $('#historyBtn').click(); };
$('#edSave').onclick = () => {
  const res = scoreMove(ed.board, ed.entry.placements);
  if (!res.valid && !confirm((res.error || 'Invalid') + ' — save anyway (scores 0)?')) return;
  if (ed.entry.placements.some(p => !p.letter)) return toast('Every tile needs a letter (blanks too)');
  $('#editor').classList.add('hidden'); const o = ed; ed = null; o.onSave(o.entry);
};

/* ---------------- End game ---------------- */
function checkAutoEnd() {
  const n = game.players.length, k = consecutiveScoreless(n, game.history);
  if (k >= 2 * n && !game.history.some(e => e.type === 'penalty') && confirm(`${k} scoreless turns in a row — end the game?`)) openEnd(-1);
}
function openEnd(defaultOut) {
  $('#penaltyBox').classList.toggle('hidden', game.history.some(e => e.type === 'penalty'));
  $('#racks').innerHTML = game.players.map((p, i) => `<label>${esc(p)} <input data-i="${i}" placeholder="tiles (QE?) or points" autocapitalize="characters"><span class="rv" data-i="${i}">0</span></label>`).join('');
  $('#outPlayer').innerHTML = game.players.map((p, i) => `<option value="${i}">${esc(p)}</option>`).join('') + '<option value="-1">Nobody — all passed</option>';
  $('#outPlayer').value = String(defaultOut ?? 0);
  updateRacks(); showFinal(); $('#endgame').classList.remove('hidden');
}
function updateRacks() {
  const out = +$('#outPlayer').value, inputs = [...$('#racks').querySelectorAll('input')];
  inputs.forEach((inp, i) => { inp.disabled = i === out; if (i === out) inp.value = ''; });
  const {deltas} = endGameAdjustments(inputs.map(i => i.value), out);
  $('#racks').querySelectorAll('.rv').forEach((el, i) => { el.textContent = (deltas[i] >= 0 ? '+' : '') + deltas[i]; });
}
$('#racks').oninput = updateRacks;
$('#outPlayer').onchange = updateRacks;
$('#endBtn').onclick = () => openEnd(0);
$('#applyPenalty').onclick = () => {
  const racks = [...$('#racks').querySelectorAll('input')].map(i => i.value.toUpperCase().trim());
  const out = +$('#outPlayer').value;
  game.history.push({type: 'penalty', racks, outPlayer: out}); saveGame(); render(); showFinal(); $('#penaltyBox').classList.add('hidden');
  const {deltas} = endGameAdjustments(racks, out);
  toast(deltas.map((d, i) => `${game.players[i]} ${d >= 0 ? '+' : ''}${d}`).join(', '));
};
function showFinal() {
  const st = gameStats(game.players.length, game.history), name = i => esc(game.players[i]);
  const order = game.players.map((p, i) => i).sort((a, b) => st.scores[b] - st.scores[a]);
  const A = st.awards, who = a => a.players.map(name).join(' & ');
  const aw = [
    ['🏆 Winner', {players: [order[0]], value: st.scores[order[0]], detail: ''}, v => v + ' pts'],
    ['📏 Longest word', A.longestWord, (v, a) => `${a.detail} (${v})`],
    ['💥 Best turn', A.bestTurn, (v, a) => `${v} pts — ${a.detail}`],
    ['⭐ Best single word', A.bestWord, (v, a) => `${a.detail} ${v}`],
    ['🎯 Most bonus squares in one word', A.mostBonusInWord, (v, a) => `${a.detail} (${v})`],
    ['🟥 Most bonus squares total', A.mostBonusTotal, v => v],
    ['🎰 Most bingos', A.mostBingos, v => v],
    ['📈 Best average / turn', A.bestAverage, v => v],
  ].filter(([, a]) => a).map(([t, a, f]) => `<div class="aw"><span>${t}</span><b>${who(a)}</b><small>${esc(f(a.value, a))}</small></div>`).join('');
  const rows = order.map(i => { const p = st.players[i]; return `<tr><td>${name(i)}</td><td><b>${st.scores[i]}</b></td><td>${p.avgPerTurn}</td><td>${p.bingos}</td><td>${p.bonusSquares}</td><td>${p.longestWord ? esc(p.longestWord.word) : '–'}</td><td>${p.bestTurn ? p.bestTurn.score : '–'}</td><td>${p.bestWord ? esc(p.bestWord.word) + ' ' + p.bestWord.score : '–'}</td></tr>`; }).join('');
  $('#final').innerHTML = `<h3>Awards</h3>${aw}<h3>Players</h3><div class="tw"><table><tr><th></th><th>Pts</th><th>Avg</th><th>Bingo</th><th>Bonus sq</th><th>Longest</th><th>Best turn</th><th>Best word</th></tr>${rows}</table></div>`;
}

/* ---------------- Settings ---------------- */
document.querySelectorAll('.openSettings').forEach(b => b.onclick = () => {
  $('#apiKey').value = settings.apiKey || ''; $('#model').value = settings.model || ''; $('#baseUrl').value = settings.baseUrl || '';
  $('#newGame').classList.toggle('hidden', !game); $('#resetCorners').classList.toggle('hidden', !game);
  $('#settings').classList.remove('hidden');
});
$('#saveSettings').onclick = () => {
  settings = {apiKey: $('#apiKey').value.trim(), model: $('#model').value.trim(), baseUrl: $('#baseUrl').value.trim()};
  localStorage.setItem(SKEY, JSON.stringify(settings)); $('#settings').classList.add('hidden'); toast('Saved');
};
$('#resetCorners').onclick = () => { defaultCorners(); saveGame(); drawOverlay(); $('#settings').classList.add('hidden'); if (!aligning) $('#alignBtn').click(); };
$('#newGame').onclick = () => {
  if (!confirm('Discard this game and go to setup?')) return;
  names = game.players.slice(); game.history = []; localStorage.setItem(GKEY, JSON.stringify({...game, history: [], baseline: null}));
  game = load(GKEY); baseline = null;
  $('#settings').classList.add('hidden'); $('#game').classList.add('hidden'); $('#setup').classList.remove('hidden'); renderNames();
};

renderNames();
if (!game) $('#resumeGame').classList.add('hidden');
if (TEST) window.scrabbleCam = {setFrame, state, get game() { return game; }};
