// On-device tile reading: warp grid like tools/make_dataset.py (26 px cells + 3 px margin), classify 32x32 crops with the tiny CNN.
import {homography, applyH} from './geom.js';
import {makeModel, cropInput, CLASSES} from './cnn.js';
const CELL = 26, M = 3, S = 15 * CELL + 2 * M;
let modelP = null;
export function loadReader(url = 'model/tiles-v1.f16') {
  if (!modelP) modelP = fetch(url).then(r => { if (!r.ok) throw new Error('model ' + r.status); return r.arrayBuffer(); }).then(makeModel);
  return modelP;
}
export function warpGrid(img, corners) {
  const dst = [{x: M, y: M}, {x: M + 15 * CELL, y: M}, {x: M + 15 * CELL, y: M + 15 * CELL}, {x: M, y: M + 15 * CELL}];
  const H = homography(dst, corners), out = new Uint8ClampedArray(S * S * 4), {width: W, height: Hh, data} = img;
  for (let v = 0; v < S; v++) for (let u = 0; u < S; u++) {
    const p = applyH(H, u, v);
    const x = Math.max(0, Math.min(W - 1.001, p.x)), y = Math.max(0, Math.min(Hh - 1.001, p.y));
    const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i00 = (y0 * W + x0) * 4, i01 = i00 + W * 4, o = (v * S + u) * 4;
    for (let c = 0; c < 3; c++) out[o + c] = (data[i00 + c] * (1 - fx) + data[i00 + 4 + c] * fx) * (1 - fy) + (data[i01 + c] * (1 - fx) + data[i01 + 4 + c] * fx) * fy;
    out[o + 3] = 255;
  }
  return {width: S, height: S, data: out};
}
/** Classify one cell (view coords). Returns {empty, blank, letter, conf, rot} where rot = CCW quarter-turns that made it upright. */
export function readCell(model, grid, r, c) {
  let best = {letter: '', conf: 0, rot: 0}, empty = 0, blank = 0;
  for (let q = 0; q < 4; q++) {
    const p = model.forward(cropInput(grid, c * CELL, r * CELL, q));
    empty += p[27] / 4; blank += p[26] / 4;
    for (let k = 0; k < 26; k++) if (p[k] > best.conf) best = {letter: CLASSES[k], conf: p[k], rot: q};
  }
  return {empty, blank, ...best};
}
