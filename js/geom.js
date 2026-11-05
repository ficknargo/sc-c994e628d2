// Pure image geometry + cell diff (no DOM). Images are {width, height, data: RGBA Uint8ClampedArray}.
export const N = 15;

function solve(A, b) { // Gaussian elimination, A: n x n
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let i = 0; i < n; i++) {
    let p = i; for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k;
    [M[i], M[p]] = [M[p], M[i]];
    for (let k = i + 1; k < n; k++) { const f = M[k][i] / M[i][i]; for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j]; }
  }
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
  return x;
}

/** Homography mapping from[i] -> to[i] (4 points {x,y}). Returns 3x3 as flat array. */
export function homography(from, to) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const {x, y} = from[i], {x: X, y: Y} = to[i];
    A.push([x, y, 1, 0, 0, 0, -x * X, -y * X]); b.push(X);
    A.push([0, 0, 0, x, y, 1, -x * Y, -y * Y]); b.push(Y);
  }
  return [...solve(A, b), 1];
}
export function applyH(H, x, y) {
  const w = H[6] * x + H[7] * y + H[8];
  return {x: (H[0] * x + H[1] * y + H[2]) / w, y: (H[3] * x + H[4] * y + H[5]) / w};
}

/** Warp quad (corners TL,TR,BR,BL in image px) to a size x size square. */
export function warp(img, corners, size) {
  const sq = [{x: 0, y: 0}, {x: size, y: 0}, {x: size, y: size}, {x: 0, y: size}];
  const H = homography(sq, corners);
  const out = new Uint8ClampedArray(size * size * 4), {width: W, height: Hh, data} = img;
  for (let v = 0; v < size; v++) for (let u = 0; u < size; u++) {
    const p = applyH(H, u + 0.5, v + 0.5);
    let x = p.x - 0.5, y = p.y - 0.5;
    x = Math.max(0, Math.min(W - 1.001, x)); y = Math.max(0, Math.min(Hh - 1.001, y));
    const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0;
    const i00 = (y0 * W + x0) * 4, i10 = i00 + 4, i01 = i00 + W * 4, i11 = i01 + 4, o = (v * size + u) * 4;
    for (let c = 0; c < 3; c++)
      out[o + c] = (data[i00 + c] * (1 - fx) + data[i10 + c] * fx) * (1 - fy) + (data[i01 + c] * (1 - fx) + data[i11 + c] * fx) * fy;
    out[o + 3] = 255;
  }
  return {width: size, height: size, data: out};
}

const G = 10; // sample grid per cell (covers cell +10% each side)
/** Per-cell features from a warped square board image. */
export function cellFeatures(img) {
  const S = img.width, cs = S / N, d = img.data;
  // integral images of gray and rgb
  const W1 = S + 1, ig = new Float64Array(W1 * W1), ir = new Float64Array(W1 * W1), igr = new Float64Array(W1 * W1), ib = new Float64Array(W1 * W1);
  let sum = 0, sum2 = 0;
  const cs3 = [0, 0, 0];
  for (let y = 0; y < S; y++) {
    let rg = 0, rr = 0, rgg = 0, rb = 0;
    for (let x = 0; x < S; x++) {
      const i = (y * S + x) * 4, r = d[i], g = d[i + 1], b = d[i + 2], gray = 0.299 * r + 0.587 * g + 0.114 * b;
      rg += gray; rr += r; rgg += g; rb += b; sum += gray; sum2 += gray * gray; cs3[0] += r; cs3[1] += g; cs3[2] += b;
      const k = (y + 1) * W1 + x + 1;
      ig[k] = ig[k - W1] + rg; ir[k] = ir[k - W1] + rr; igr[k] = igr[k - W1] + rgg; ib[k] = ib[k - W1] + rb;
    }
  }
  const n = S * S, mean = sum / n, std = Math.sqrt(Math.max(1, sum2 / n - mean * mean));
  const cm = cs3.map(v => v / n + 1);
  const box = (I, x0, y0, x1, y1) => {
    x0 = Math.max(0, Math.min(S - 1, Math.round(x0))); x1 = Math.max(x0 + 1, Math.min(S, Math.round(x1)));
    y0 = Math.max(0, Math.min(S - 1, Math.round(y0))); y1 = Math.max(y0 + 1, Math.min(S, Math.round(y1)));
    return (I[y1 * W1 + x1] - I[y0 * W1 + x1] - I[y1 * W1 + x0] + I[y0 * W1 + x0]) / ((x1 - x0) * (y1 - y0));
  };
  const cells = [];
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const x0 = (c - 0.1) * cs, y0 = (r - 0.1) * cs, step = cs * 1.2 / G;
    const tex = new Float32Array(G * G);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++)
      tex[j * G + i] = (box(ig, x0 + i * step, y0 + j * step, x0 + (i + 1) * step, y0 + (j + 1) * step) - mean) / std;
    const a = [c * cs + cs * 0.2, r * cs + cs * 0.2, (c + 1) * cs - cs * 0.2, (r + 1) * cs - cs * 0.2];
    const col = [box(ir, ...a) / cm[0], box(igr, ...a) / cm[1], box(ib, ...a) / cm[2]];
    cells.push({tex, col});
  }
  return cells;
}

/** Diff score per cell between baseline and current features (min over +-1 sample shift). */
export function diffScores(base, cur) {
  return cur.map((fc, k) => {
    const fb = base[k];
    let best = Infinity;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      let s = 0;
      for (let j = 1; j < G - 1; j++) for (let i = 1; i < G - 1; i++)
        s += Math.abs(fc.tex[j * G + i] - fb.tex[(j + dy) * G + i + dx]);
      best = Math.min(best, s / ((G - 2) * (G - 2)));
    }
    const cd = Math.abs(fc.col[0] - fb.col[0]) + Math.abs(fc.col[1] - fb.col[1]) + Math.abs(fc.col[2] - fb.col[2]);
    return best + 1.5 * cd;
  });
}

/**
 * Pick the newly placed tiles: cells empty in `occupied` (225 bools) whose diff score stands out,
 * constrained to a single row/column through the strongest change. Returns [{r,c,score}].
 */
export function pickNewCells(scores, occupied, {floor = 0.45, k = 6} = {}) {
  const free = [];
  scores.forEach((s, i) => { if (!occupied[i]) free.push(s); });
  const sorted = [...free].sort((a, b) => a - b);
  const med = sorted[sorted.length >> 1] || 0;
  const mad = sorted.map(v => Math.abs(v - med)).sort((a, b) => a - b)[sorted.length >> 1] || 0;
  const thr = Math.max(floor, med + k * mad);
  let cand = [];
  scores.forEach((s, i) => { if (!occupied[i] && s > thr) cand.push({r: Math.floor(i / N), c: i % N, score: s}); });
  cand.sort((a, b) => b.score - a.score);
  if (cand.length > 1) {
    const t = cand[0];
    const inRow = cand.filter(x => x.r === t.r), inCol = cand.filter(x => x.c === t.c);
    cand = inRow.length >= inCol.length ? inRow : inCol;
  }
  return {cells: cand.slice(0, 7), threshold: thr};
}

export function packFeatures(f) { return f.map(c => ({t: Array.from(c.tex, v => Math.round(v * 100)), c: c.col.map(v => Math.round(v * 1000))})); }
export function unpackFeatures(p) { return p.map(c => ({tex: Float32Array.from(c.t, v => v / 100), col: c.c.map(v => v / 1000)})); }

/* ---------- Orientation ---------- */
/** One clockwise quarter-turn of grid coords, k times. */
export function rotCell(r, c, k, n = N) {
  k = ((k % 4) + 4) % 4;
  for (let i = 0; i < k; i++) [r, c] = [c, n - 1 - r];
  return [r, c];
}
/** Put 4 corners (any order) into clockwise order starting with the top-left-most (min x+y). Removes mirror ambiguity. */
export function normalizeCorners(pts) {
  const cx = pts.reduce((s, p) => s + p.x, 0) / 4, cy = pts.reduce((s, p) => s + p.y, 0) / 4;
  const cw = [...pts].sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)); // y-down => clockwise
  let s = 0; cw.forEach((p, i) => { if (p.x + p.y < cw[s].x + cw[s].y) s = i; });
  return [...cw.slice(s), ...cw.slice(0, s)];
}
/** Rotate a full-board feature set clockwise by d quarter-turns (cell positions and in-cell textures). */
export function rotateFeatures(F, d) {
  d = ((d % 4) + 4) % 4; if (!d) return F;
  const out = new Array(F.length);
  F.forEach((f, i) => {
    const [r, c] = rotCell(Math.floor(i / N), i % N, d);
    const tex = new Float32Array(G * G);
    for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) { const [ty, tx] = rotCell(y, x, d, G); tex[ty * G + tx] = f.tex[y * G + x]; }
    out[r * N + c] = {tex, col: f.col};
  });
  return out;
}
/** Which clockwise rotation d of the old view best matches the new view? (premium layout is symmetric, so tiles decide) */
export function findRotation(oldF, newF) {
  const costs = [0, 1, 2, 3].map(d => diffScores(rotateFeatures(oldF, d), newF).reduce((a, b) => a + b, 0));
  const d = costs.indexOf(Math.min(...costs));
  return {d, costs};
}
/** Rotate an RGBA image clockwise by k quarter-turns (square or not). */
export function rotateImage(img, k) {
  k = ((k % 4) + 4) % 4; if (!k) return img;
  let {width: w, height: h, data} = img;
  for (let t = 0; t < k; t++) {
    const out = new Uint8ClampedArray(data.length);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const o = (x * h + (h - 1 - y)) * 4, i = (y * w + x) * 4; out[o] = data[i]; out[o + 1] = data[i + 1]; out[o + 2] = data[i + 2]; out[o + 3] = data[i + 3]; }
    data = out; [w, h] = [h, w];
  }
  return {width: w, height: h, data};
}
