// Tiny CNN inference in plain JS (weights trained by tools/train.py, float16 file). Layout CHW.
export const CLASSES = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'BLANK', 'EMPTY', 'ROT'];
const ARCH = {c1: 16, c2: 24, c3: 48, fc: 96, nc: 29};

function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

export function makeModel(buffer) {
  const w = f16ToF32(new Uint16Array(buffer));
  let o = 0; const take = n => { const a = w.subarray(o, o + n); o += n; return a; };
  const {c1, c2, c3, fc, nc} = ARCH;
  const L = {
    w1: take(c1 * 3 * 9), b1: take(c1), w2: take(c2 * c1 * 9), b2: take(c2), w3: take(c3 * c2 * 9), b3: take(c3),
    wf: take(fc * c3 * 16), bf: take(fc), wo: take(nc * fc), bo: take(nc),
  };
  if (o !== w.length) throw new Error('model size mismatch ' + o + ' vs ' + w.length);
  return {forward: x => forward(L, x)};
}

// conv3x3 (pad 1) + ReLU + maxpool2
function convBlock(inp, Cin, H, W, wt, b, Cout) {
  const P = W + 2, pad = new Float32Array(Cin * (H + 2) * P);
  for (let c = 0; c < Cin; c++) for (let y = 0; y < H; y++) pad.set(inp.subarray(c * H * W + y * W, c * H * W + y * W + W), c * (H + 2) * P + (y + 1) * P + 1);
  const acc = new Float32Array(H * W), Ho = H >> 1, Wo = W >> 1, out = new Float32Array(Cout * Ho * Wo);
  for (let co = 0; co < Cout; co++) {
    acc.fill(b[co]);
    for (let ci = 0; ci < Cin; ci++) {
      const base = ci * (H + 2) * P, wb = (co * Cin + ci) * 9;
      for (let k = 0; k < 9; k++) {
        const wv = wt[wb + k]; if (wv === 0) continue;
        const off = base + ((k / 3) | 0) * P + (k % 3);
        for (let y = 0; y < H; y++) { const ro = off + y * P, ao = y * W; for (let x = 0; x < W; x++) acc[ao + x] += wv * pad[ro + x]; }
      }
    }
    const ob = co * Ho * Wo;
    for (let y = 0; y < Ho; y++) for (let x = 0; x < Wo; x++) {
      const i = 2 * y * W + 2 * x;
      const m = Math.max(acc[i], acc[i + 1], acc[i + W], acc[i + W + 1]);
      out[ob + y * Wo + x] = m > 0 ? m : 0;
    }
  }
  return out;
}

function forward(L, x) { // x: Float32Array(3*32*32) normalised
  const {c1, c2, c3, fc, nc} = ARCH;
  let h = convBlock(x, 3, 32, 32, L.w1, L.b1, c1);
  h = convBlock(h, c1, 16, 16, L.w2, L.b2, c2);
  h = convBlock(h, c2, 8, 8, L.w3, L.b3, c3);
  const n = c3 * 16, z = new Float32Array(fc);
  for (let j = 0; j < fc; j++) { let s = L.bf[j]; const r = j * n; for (let i = 0; i < n; i++) s += L.wf[r + i] * h[i]; z[j] = s > 0 ? s : 0; }
  const logits = new Float32Array(nc);
  for (let k = 0; k < nc; k++) { let s = L.bo[k]; const r = k * fc; for (let j = 0; j < fc; j++) s += L.wo[r + j] * z[j]; logits[k] = s; }
  let mx = -Infinity; for (const v of logits) mx = Math.max(mx, v);
  let sum = 0; const p = new Float32Array(nc); for (let k = 0; k < nc; k++) { p[k] = Math.exp(logits[k] - mx); sum += p[k]; }
  for (let k = 0; k < nc; k++) p[k] /= sum;
  return p;
}

/** Crop (HWC uint8 RGBA source image of the warped grid, top-left x0,y0) rotated CCW by q quarter-turns -> normalised CHW input. */
export function cropInput(img, x0, y0, q = 0, size = 32) {
  const out = new Float32Array(3 * size * size), {width: W, data} = img;
  let sum = 0;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    // np.rot90(a, q) (CCW): out[y][x] = a[x][n-1-y] for q=1
    let sy = y, sx = x;
    for (let t = 0; t < q; t++) { const ny = sx, nx = size - 1 - sy; sy = ny; sx = nx; }
    const i = ((y0 + sy) * W + x0 + sx) * 4;
    for (let c = 0; c < 3; c++) { const v = data[i + c]; out[c * size * size + y * size + x] = v; sum += v; }
  }
  const n = out.length, mean = sum / n; let ss = 0;
  for (let i = 0; i < n; i++) { const d = out[i] - mean; ss += d * d; }
  const sd = Math.sqrt(ss / (n - 1)) + 8;
  for (let i = 0; i < n; i++) out[i] = (out[i] - mean) / sd;
  return out;
}
