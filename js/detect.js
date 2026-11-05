// Board detection: OpenCV.js proposes quadrilaterals; a premium-square layout score (pure JS) verifies and refines them.
import {homography, applyH, normalizeCorners} from './geom.js';
import {PREMIUM} from './scoring.js';

const GROUP = PREMIUM.map(row => row.map(p => ({'': 0, DL: 1, TL: 2, DW: 3, TW: 4})[p]));

function sampler(img) {
  const {width: W, height: H, data} = img;
  return (x, y) => {
    x = Math.max(0, Math.min(W - 1.001, x)); y = Math.max(0, Math.min(H - 1.001, y));
    const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = (y0 * W + x0) * 4, j = i + W * 4;
    const out = [0, 0, 0];
    for (let c = 0; c < 3; c++) out[c] = (data[i + c] * (1 - fx) + data[i + 4 + c] * fx) * (1 - fy) + (data[j + c] * (1 - fx) + data[j + 4 + c] * fx) * fy;
    return out;
  };
}
const UNIT = [{x: 0, y: 0}, {x: 1, y: 0}, {x: 1, y: 1}, {x: 0, y: 1}];

/** Mean colour of each of the 225 cells (k x k samples in the central part of the cell). */
export function cellColors(img, quad, k = 3, spread = 0.5) {
  const Hm = homography(UNIT, quad), s = sampler(img), out = new Float32Array(225 * 3);
  for (let r = 0; r < 15; r++) for (let c = 0; c < 15; c++) {
    let a = [0, 0, 0];
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
      const du = k === 1 ? 0 : (i / (k - 1) - 0.5) * spread, dv = k === 1 ? 0 : (j / (k - 1) - 0.5) * spread;
      const p = applyH(Hm, (c + 0.5 + du) / 15, (r + 0.5 + dv) / 15), v = s(p.x, p.y);
      a[0] += v[0]; a[1] += v[1]; a[2] += v[2];
    }
    const o = (r * 15 + c) * 3; out[o] = a[0] / k / k; out[o + 1] = a[1] / k / k; out[o + 2] = a[2] / k / k;
  }
  return out;
}
/** Fisher ratio of cell colours grouped by premium type (rotation invariant: the layout is 4-fold symmetric). */
export function layoutScore(cols) {
  const n = [0, 0, 0, 0, 0], mu = Array.from({length: 5}, () => [0, 0, 0]), all = [0, 0, 0];
  for (let i = 0; i < 225; i++) { const g = GROUP[(i / 15) | 0][i % 15]; n[g]++; for (let c = 0; c < 3; c++) { mu[g][c] += cols[i * 3 + c]; all[c] += cols[i * 3 + c]; } }
  for (let g = 0; g < 5; g++) for (let c = 0; c < 3; c++) mu[g][c] /= n[g];
  for (let c = 0; c < 3; c++) all[c] /= 225;
  let between = 0, within = 0;
  for (let g = 0; g < 5; g++) for (let c = 0; c < 3; c++) between += n[g] * (mu[g][c] - all[c]) ** 2;
  for (let i = 0; i < 225; i++) { const g = GROUP[(i / 15) | 0][i % 15]; for (let c = 0; c < 3; c++) within += (cols[i * 3 + c] - mu[g][c]) ** 2; }
  return between / (within + 1e-6 + 225 * 4); // small floor so flat areas don't score
}
export const quadScore = (img, quad, k = 3, spread = 0.5) => layoutScore(cellColors(img, quad, k, spread));

/** Inner quad at margin f (fraction) inside an outer quad, in the quad's own projective coordinates. */
export function insetQuad(quad, fx0, fy0, fx1 = fx0, fy1 = fy0) {
  const Hm = homography(UNIT, quad);
  return [applyH(Hm, fx0, fy0), applyH(Hm, 1 - fx1, fy0), applyH(Hm, 1 - fx1, 1 - fy1), applyH(Hm, fx0, 1 - fy1)];
}

/** Find the grid inside a candidate outline (board edge or grid edge), then polish corners. */
export function fitGrid(img, quad, {polish = true} = {}) {
  let best = {score: -1, quad};
  for (let f = -0.02; f <= 0.16; f += 0.005) {
    const q = insetQuad(quad, f, f), s = quadScore(img, q);
    if (s > best.score) best = {score: s, quad: q};
  }
  // per-side margins
  let q = best.quad;
  for (const step of [0.01, 0.004]) for (let it = 0; it < 3; it++) for (let side = 0; side < 4; side++) for (const d of [-step, step]) {
    const m = [0, 0, 0, 0]; m[side] = d;
    const cand = insetQuad(q, m[0], m[1], m[2], m[3]), s = quadScore(img, cand);
    if (s > best.score) { best = {score: s, quad: cand}; q = cand; }
  }
  if (polish) best = polishCorners(img, best.quad);
  return best;
}
/** Coordinate ascent on the 8 corner coordinates with a sharper (smaller-region) score. */
export function polishCorners(img, quad) {
  let q = quad.map(p => ({...p})), side = Math.hypot(q[1].x - q[0].x, q[1].y - q[0].y) / 15;
  const sc = qq => quadScore(img, qq, 3, 0.4);
  let best = sc(q);
  for (const st of [0.25, 0.12, 0.06]) for (let it = 0; it < 2; it++) for (let i = 0; i < 4; i++) for (const ax of ['x', 'y']) for (const d of [-st, st]) {
    const cand = q.map(p => ({...p})); cand[i][ax] += d * side;
    const s = sc(cand); if (s > best) { best = s; q = cand; }
  }
  return {score: quadScore(img, q), quad: q};
}

/** OpenCV quadrilateral proposals. img: ImageData (small, ~480px). */
export function proposeQuads(cv, img) {
  const src = cv.matFromImageData(img), gray = new cv.Mat(), blur = new cv.Mat(), edges = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);
  const area = img.width * img.height, quads = [];
  const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
  for (const [t1, t2] of [[30, 90], [60, 160], [15, 45]]) {
    cv.Canny(blur, edges, t1, t2);
    cv.dilate(edges, edges, kernel);
    const contours = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(edges, contours, hier, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i), a = cv.contourArea(cnt);
      if (a > 0.04 * area) {
        const hull = new cv.Mat(); cv.convexHull(cnt, hull);
        for (const eps of [0.02, 0.04, 0.07]) {
          const ap = new cv.Mat(); cv.approxPolyDP(hull, ap, eps * cv.arcLength(hull, true), true);
          if (ap.rows === 4) { const d = ap.data32S; quads.push(normalizeCorners([0, 1, 2, 3].map(k => ({x: d[2 * k], y: d[2 * k + 1]})))); ap.delete(); break; }
          ap.delete();
        }
        // also the min-area rectangle of large contours
        const rr = cv.minAreaRect(hull), pts = cv.RotatedRect.points(rr);
        quads.push(normalizeCorners(pts.map(p => ({x: p.x, y: p.y}))));
        hull.delete();
      }
      cnt.delete();
    }
    contours.delete(); hier.delete();
  }
  [src, gray, blur, edges, kernel].forEach(m => m.delete());
  // dedupe
  const out = [];
  for (const q of quads) if (!out.some(o => o.every((p, i) => Math.hypot(p.x - q[i].x, p.y - q[i].y) < 4))) out.push(q);
  return out;
}

/** Full detection on a small image. Returns {quad, score} (quad in img px, clockwise from top-left) or null. */
export function detectBoard(cv, img, {minScore = 0.6, maxCands = 12} = {}) {
  const cands = proposeQuads(cv, img)
    .map(q => ({q, s: quadScore(img, insetQuad(q, 0.05, 0.05), 2, 0.4)}))
    .sort((a, b) => b.s - a.s).slice(0, maxCands);
  let best = null;
  for (const {q} of cands) {
    const f = fitGrid(img, q, {polish: false});
    if (!best || f.score > best.score) best = f;
  }
  if (!best || best.score < minScore) return best ? {...best, found: false} : null;
  return {...polishCorners(img, best.quad), found: true};
}

/** Scale a quad from one image size to another and polish it there (full-res refinement before reading). */
export function refineAt(img, quad, scale) {
  const q = quad.map(p => ({x: p.x * scale, y: p.y * scale}));
  const F = gradField(img), f0 = quadScore(img, q);
  const side = Math.hypot(q[1].x - q[0].x, q[1].y - q[0].y) / 15;
  let best = {quad: q, lineScore: -Infinity, score: f0};
  // multi-start (the line objective is periodic); keep only solutions that preserve the premium-layout fit
  for (const dx of [-0.4, 0, 0.4]) for (const dy of [-0.4, 0, 0.4]) {
    const start = q.map(p => ({x: p.x + dx * side, y: p.y + dy * side}));
    const r = snapToLines(F, start), fs = quadScore(img, r.quad);
    if (fs >= 0.35 * f0 && r.lineScore > best.lineScore) best = {quad: r.quad, lineScore: r.lineScore, score: fs};
  }
  return best;
}

/* ---------- Precise refinement: align the predicted 16+16 grid lines with image edges ---------- */
export function gradField(img) {
  const {width: W, height: H, data} = img, g = new Float32Array(W * H), m = new Float32Array(W * H);
  for (let i = 0, j = 0; i < W * H; i++, j += 4) g[i] = 0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2];
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x;
    const gx = g[i - W + 1] + 2 * g[i + 1] + g[i + W + 1] - g[i - W - 1] - 2 * g[i - 1] - g[i + W - 1];
    const gy = g[i + W - 1] + 2 * g[i + W] + g[i + W + 1] - g[i - W - 1] - 2 * g[i - W] - g[i - W + 1];
    m[i] = Math.sqrt(gx * gx + gy * gy);
  }
  return {W, H, m};
}
function fsample(f, x, y) {
  if (x < 0 || y < 0 || x >= f.W - 1 || y >= f.H - 1) return 0;
  const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * f.W + x0, m = f.m;
  return (m[i] * (1 - fx) + m[i + 1] * fx) * (1 - fy) + (m[i + f.W] * (1 - fx) + m[i + f.W + 1] * fx) * fy;
}
export let OUTER_W = 1;
export function setOuterW(v) { OUTER_W = v; }
export function lineScore(f, quad, per = 6) {
  const Hm = homography(UNIT, quad);
  let on = 0, off = 0, n = 0;
  for (let i = 0; i <= 15; i++) {
    const w = i === 0 || i === 15 ? OUTER_W : 1, a = i / 15, b = Math.min(i + 0.5, 14.5) / 15;
    for (let k = 0; k < 15 * per; k++) {
      const t = (k + 0.5) / (15 * per);
      let p = applyH(Hm, t, a); on += w * fsample(f, p.x, p.y);
      p = applyH(Hm, a, t); on += w * fsample(f, p.x, p.y);
      p = applyH(Hm, t, b); off += w * fsample(f, p.x, p.y);
      p = applyH(Hm, b, t); off += w * fsample(f, p.x, p.y);
      n += 2 * w;
    }
  }
  return (on - off) / n;
}
export function snapToLines(f, quad) {
  let q = quad.map(p => ({...p}));
  const side = () => (Math.hypot(q[1].x - q[0].x, q[1].y - q[0].y) + Math.hypot(q[2].x - q[3].x, q[2].y - q[3].y)) / 30;
  let best = lineScore(f, q);
  for (const st of [0.3, 0.15, 0.08, 0.04, 0.02]) {
    for (let it = 0; it < 4; it++) {
      let improved = false;
      for (let i = 0; i < 4; i++) for (const ax of ['x', 'y']) for (const d of [-st, st]) {
        const cand = q.map(p => ({...p})); cand[i][ax] += d * side();
        const s = lineScore(f, cand); if (s > best) { best = s; q = cand; improved = true; }
      }
      // whole-grid translation moves help escape coupled corners
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const cand = q.map(p => ({x: p.x + dx * st * side(), y: p.y + dy * st * side()}));
        const s = lineScore(f, cand); if (s > best) { best = s; q = cand; improved = true; }
      }
      if (!improved) break;
    }
  }
  return {quad: q, lineScore: best};
}
