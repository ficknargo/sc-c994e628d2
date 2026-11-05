"""Cell-crop dataset from synthetic photos. Crops are extracted exactly like the app does (js/reader.js):
warp grid to 15*26 px (+3 px margin), take 32x32 window around each cell."""
import sys, random, numpy as np, cv2, multiprocessing as mp
sys.path.insert(0, 'tools'); import synth
CELL, M, CROP = 26, 3, 32
S = 15 * CELL + 2 * M
CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['BLANK', 'EMPTY', 'ROT']

def warp_grid(img, corners):
    dst = np.array([[M, M], [M + 15 * CELL, M], [M + 15 * CELL, M + 15 * CELL], [M, M + 15 * CELL]], np.float32)
    Hm = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    return cv2.warpPerspective(img, Hm, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def crop(w, r, c): return w[r * CELL: r * CELL + CROP, c * CELL: c * CELL + CROP]

def board_samples(seed, fonts, jitter=0.12):
    r = random.Random(seed)
    st = synth.random_style(r, fonts)
    tiles = synth.random_tiles(r, style=st)
    img, corners, _ = synth.photo(tiles, r, style=st)
    side = np.linalg.norm(corners[1] - corners[0]) / 15
    cj = corners + np.array([[r.gauss(0, jitter * side) for _ in range(2)] for _ in range(4)])
    w = warp_grid(img, cj)
    X, Y = [], []
    for row in range(15):
        for col in range(15):
            cr = crop(w, row, col)
            t = tiles.get((row, col))
            if t is None:
                if r.random() < 0.3: X.append(np.rot90(cr, r.randint(0, 3))); Y.append(27)
                continue
            letter, blank, q = t
            if blank:
                X.append(np.rot90(cr, r.randint(0, 3))); Y.append(26); continue
            up = np.rot90(cr, q)          # np.rot90 is CCW: undo the tile's CW rotation
            X.append(up); Y.append(ord(letter) - 65)
            if letter in 'OX': continue                     # look the same at every quarter-turn (digit aside)
            rots = [1, 3] if letter in 'HINSZ' else [1, 2, 3]   # 180-degree symmetric letters
            X.append(np.rot90(up, r.choice(rots))); Y.append(28)
    return np.array(X, np.uint8), np.array(Y, np.uint8)

def build(n, fonts, seed0, out, workers=8):
    with mp.Pool(workers) as p:
        res = p.starmap(board_samples, [(seed0 + i, fonts) for i in range(n)], chunksize=8)
    X = np.concatenate([a for a, _ in res]); Y = np.concatenate([b for _, b in res])
    np.savez_compressed(out, X=X, Y=Y); print(out, X.shape, np.bincount(Y, minlength=29))

if __name__ == '__main__':
    import os; os.makedirs('tools/data', exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    build(n, synth.TRAIN_FONTS, 0, 'tools/data/train.npz')
    build(300, synth.TRAIN_FONTS, 10_000_000, 'tools/data/val.npz')
    build(300, synth.TEST_FONTS, 20_000_000, 'tools/data/test_heldout_fonts.npz')
