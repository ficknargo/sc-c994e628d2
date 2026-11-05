// Browser-side vision helpers: canvas conversions + OpenAI-compatible vision LLM calls.
import {warp, rotateImage, N} from './geom.js';

export function toCanvas(img) {
  const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
  c.getContext('2d').putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  return c;
}

/** Grab a frame (video or image/canvas) into ImageData, longest side <= maxDim. */
export function grabFrame(source, maxDim = 1600) {
  const w0 = source.videoWidth || source.naturalWidth || source.width, h0 = source.videoHeight || source.naturalHeight || source.height;
  const s = Math.min(1, maxDim / Math.max(w0, h0)), w = Math.round(w0 * s), h = Math.round(h0 * s);
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d', {willReadFrequently: true}); ctx.drawImage(source, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

/** Contact sheet of cell crops (each centred on one candidate cell, green box marks the cell). */
export function cropSheet(frame, corners, cells, k = 0) {
  const S = 1050, cs = S / N, big = toCanvas(warp(frame, corners, S));
  const tile = 170, head = 30, cols = Math.min(4, cells.length), rows = Math.ceil(cells.length / cols);
  const c = document.createElement('canvas'); c.width = cols * tile; c.height = rows * (tile + head);
  const ctx = c.getContext('2d'); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, c.width, c.height);
  const ext = 1.6; // crop 1.6 cells wide
  cells.forEach((cell, i) => {
    const x = (i % cols) * tile, y = Math.floor(i / cols) * (tile + head);
    ctx.fillStyle = '#000'; ctx.font = 'bold 24px sans-serif'; ctx.fillText('#' + (i + 1), x + 6, y + 24);
    const sx = (cell.c + 0.5 - ext / 2) * cs, sy = (cell.r + 0.5 - ext / 2) * cs;
    // rotate each crop by k quarter-turns so it shows the board's canonical orientation
    const cx = x + tile / 2, cy = y + head + (tile - 8) / 2 + 0;
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(k * Math.PI / 2);
    ctx.drawImage(big, sx, sy, cs * ext, cs * ext, -(tile - 8) / 2, -(tile - 8) / 2, tile - 8, tile - 8); ctx.restore();
    const inner = (tile - 8) / ext;
    ctx.strokeStyle = '#00c853'; ctx.lineWidth = 3;
    ctx.strokeRect(x + 4 + (tile - 8 - inner) / 2, y + head + (tile - 8 - inner) / 2, inner, inner);
  });
  return c;
}

/** Whole board with A-O / 1-15 labels and grid lines. */
export function labelledBoard(frame, corners, k = 0) {
  const S = 1050, cs = S / N, m = 44, big = toCanvas(rotateImage(warp(frame, corners, S), k));
  const c = document.createElement('canvas'); c.width = c.height = S + m;
  const ctx = c.getContext('2d'); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(big, m, m);
  ctx.strokeStyle = 'rgba(255,0,255,.55)'; ctx.lineWidth = 1;
  for (let i = 0; i <= N; i++) {
    ctx.beginPath(); ctx.moveTo(m + i * cs, m); ctx.lineTo(m + i * cs, m + S); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(m, m + i * cs); ctx.lineTo(m + S, m + i * cs); ctx.stroke();
  }
  ctx.fillStyle = '#000'; ctx.font = 'bold 22px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let i = 0; i < N; i++) { ctx.fillText(String.fromCharCode(65 + i), m + (i + .5) * cs, m / 2); ctx.fillText(String(i + 1), m / 2, m + (i + .5) * cs); }
  return c;
}

async function chat({apiKey, model, baseUrl}, system, text, canvas, schemaName, schema) {
  const url = canvas.toDataURL('image/jpeg', 0.9);
  const resp = await fetch((baseUrl || 'https://api.openai.com/v1').replace(/\/$/, '') + '/chat/completions', {
    method: 'POST',
    headers: {'Content-Type': 'application/json', Authorization: 'Bearer ' + apiKey},
    body: JSON.stringify({
      model, temperature: 0,
      messages: [{role: 'system', content: system}, {role: 'user', content: [{type: 'text', text}, {type: 'image_url', image_url: {url, detail: 'high'}}]}],
      response_format: {type: 'json_schema', json_schema: {name: schemaName, strict: true, schema}},
    }),
  });
  if (!resp.ok) throw new Error('Vision API ' + resp.status + ': ' + (await resp.text()).slice(0, 200));
  const j = await resp.json();
  return JSON.parse(j.choices[0].message.content);
}

const UP = ['up', 'right', 'down', 'left'];
/** Clockwise quarter-turns needed to make letters upright, from their "up" directions (majority vote). */
export function majorityExtra(ups) {
  const n = {up: 0, right: 0, down: 0, left: 0}; ups.forEach(u => { if (u in n) n[u]++; });
  const best = Object.keys(n).reduce((a, b) => (n[b] > n[a] ? b : a), 'up');
  return {up: 0, left: 1, down: 2, right: 3}[best];
}
const SYS = 'You read Scrabble tiles from photos. Answer only with the requested JSON. Letters are uppercase A-Z. A tile with no printed letter is a blank tile: report "?".';

/** Read the letter in each candidate cell. Returns array of 'A'-'Z' | '?' (blank) | '' (no tile). */
export async function readCells(settings, frame, corners, cells) {
  const sheet = cropSheet(frame, corners, cells);
  const schema = {type: 'object', additionalProperties: false, required: ['tiles'], properties: {tiles: {type: 'array', items: {
    type: 'object', additionalProperties: false, required: ['n', 'letter'],
    properties: {n: {type: 'integer'}, letter: {type: 'string', enum: [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ', '?', '-']}, up: {type: 'string', enum: UP}}}}}};
  const text = `The image has ${cells.length} numbered crops (#1..#${cells.length}) from a Scrabble board. In each crop, look ONLY at the square outlined in green. ` +
    'Return its letter (ignore the small point-value number); the letter may be rotated, identify it anyway. Also return "up": which side of the crop the TOP of the letter points to (up/right/down/left).  If the tile is blank (no letter) return "?". If the green square holds no tile (empty board square, possibly printed with words like TRIPLE WORD) return "-". Return one entry per crop.';
  const out = await chat(settings, SYS, text, sheet, 'tiles', schema);
  const letters = cells.map((_, i) => { const t = out.tiles.find(t => t.n === i + 1); return !t || t.letter === '-' ? '' : t.letter; });
  return {letters, extra: majorityExtra(out.tiles.filter(t => t.letter !== '-').map(t => t.up))};
}

/** Read the whole board. Returns 15 strings of 15 chars ('.' empty, '?' blank). */
export async function readBoard(settings, frame, corners, knownRows, k = 0) {
  const img = labelledBoard(frame, corners, k);
  const schema = {type: 'object', additionalProperties: false, required: ['rows', 'up'], properties: {rows: {type: 'array', items: {type: 'string'}}, up: {type: 'string', enum: UP}}};
  const text = 'This is a top-down Scrabble board, 15x15, columns A-O left to right and rows 1-15 top to bottom (magenta grid lines). ' +
    'Return "rows": exactly 15 strings, each exactly 15 characters: the tile letter, "?" for a blank tile (tile with no letter), or "." for a square with no tile. ' +
    'Read letters even if tiles are rotated. Also return "up": the direction the TOP of most letters points in this image (up/right/down/left). ' +
    'Printed premium-square labels (TW, DW, TL, DL, TRIPLE WORD SCORE etc.) are NOT tiles. Tiles are cream/ivory squares with one large letter.\n' +
    'Tiles already known from earlier turns (may help you align):\n' + knownRows.join('\n');
  const out = await chat(settings, SYS, text, img, 'board', schema);
  const rows = Array.from({length: N}, (_, r) => ((out.rows[r] || '') + '.'.repeat(N)).slice(0, N).toUpperCase());
  return {rows, extra: majorityExtra([out.up])};
}
