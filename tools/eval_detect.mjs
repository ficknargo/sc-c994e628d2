// Detection accuracy on held-out synthetic handheld photos: corner error in cell units.
import fs from 'node:fs'; import {createRequire} from 'node:module';
import {detectBoard, refineAt} from '../js/detect.js';
import {downscale, normalizeCorners} from '../js/geom.js';
const require = createRequire(import.meta.url);
let cv = require('../vendor/opencv.js'); if (cv instanceof Promise) cv = await cv; else if (!cv.Mat) await new Promise(r => { cv.onRuntimeInitialized = r; });
const dir = 'tools/data/eval/', meta = JSON.parse(fs.readFileSync(dir + 'meta.json'));
const lim = +process.argv[2] || meta.length;
let ok = 0, errs = [], times = [];
for (const m of meta.slice(0, lim)) {
  const img = {width: m.w, height: m.h, data: new Uint8ClampedArray(fs.readFileSync(dir + m.file + '.rgba'))};
  const t0 = performance.now();
  const small = downscale(img, 480), s = small.width / img.width;
  const det = detectBoard(cv, small);
  let err = Infinity;
  if (det && det.found) {
    const ref = refineAt(img, det.quad, 1 / s);
    const gt = normalizeCorners(m.corners.map(([x, y]) => ({x, y}))), q = normalizeCorners(ref.quad);
    const cell = Math.hypot(gt[1].x - gt[0].x, gt[1].y - gt[0].y) / 15;
    err = Math.min(...[0, 1, 2, 3].map(k => Math.max(...gt.map((p, i) => Math.hypot(p.x - q[(i + k) % 4].x, p.y - q[(i + k) % 4].y))))) / cell;
    if (process.env.DUMP) fs.writeFileSync('/tmp/det_' + m.file + '.json', JSON.stringify({gt, q, raw: det.quad.map(p => ({x: p.x / s, y: p.y / s}))}));
  }
  times.push(performance.now() - t0);
  errs.push(err); if (err < 0.25) ok++;
  const gq = m.corners, cellpx = Math.hypot(gq[1][0] - gq[0][0], gq[1][1] - gq[0][1]) / 15;
  let rawErr = Infinity;
  if (det && det.found) { const gt = normalizeCorners(gq.map(([x, y]) => ({x, y}))), q = normalizeCorners(det.quad.map(p => ({x: p.x / s, y: p.y / s}))); rawErr = Math.min(...[0, 1, 2, 3].map(k => Math.max(...gt.map((p, i) => Math.hypot(p.x - q[(i + k) % 4].x, p.y - q[(i + k) % 4].y))))) / cellpx; }
  console.log(m.file, det ? det.score.toFixed(2) : 'none', det?.found, err.toFixed(3), 'raw', rawErr.toFixed(3), 'cellpx', cellpx.toFixed(1));
}
errs.sort((a, b) => a - b);
console.log(`detected within 0.25 cell: ${ok}/${errs.length}; median err ${errs[errs.length >> 1].toFixed(3)} cells; median time ${times.sort((a, b) => a - b)[times.length >> 1].toFixed(0)} ms`);
