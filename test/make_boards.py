"""Render synthetic photos of a Scrabble board (perspective-warped, noisy) for tests."""
import json, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PREM = {}
for k in ['0,0','0,7','0,14','7,0','7,14','14,0','14,7','14,14']: PREM[k]='TW'
for k in ['1,1','2,2','3,3','4,4','10,10','11,11','12,12','13,13','1,13','2,12','3,11','4,10','10,4','11,3','12,2','13,1','7,7']: PREM[k]='DW'
for k in ['1,5','1,9','5,1','5,5','5,9','5,13','9,1','9,5','9,9','9,13','13,5','13,9']: PREM[k]='TL'
for k in ['0,3','0,11','2,6','2,8','3,0','3,7','3,14','6,2','6,6','6,8','6,12','7,3','7,11','8,2','8,6','8,8','8,12','11,0','11,7','11,14','12,6','12,8','14,3','14,11']: PREM[k]='DL'
COL = {'TW':(205,60,55),'DW':(240,170,170),'TL':(60,110,190),'DL':(170,210,235),'':(200,195,165)}
VAL = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',[1,3,3,2,1,4,2,4,1,8,5,1,3,1,1,3,10,1,1,1,1,4,4,8,4,10]))
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

def render_flat(tiles, cell=60):
    S = cell*15
    im = Image.new('RGB', (S, S), (30, 90, 70)); d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, int(cell*0.62)); fs = ImageFont.truetype(FONT, int(cell*0.22)); ft = ImageFont.truetype(FONT, int(cell*0.16))
    for r in range(15):
        for c in range(15):
            p = PREM.get(f'{r},{c}', ''); x, y = c*cell, r*cell
            d.rectangle([x+2, y+2, x+cell-3, y+cell-3], fill=COL[p])
            if p: d.text((x+cell/2, y+cell/2), p, fill=(255,255,255), font=ft, anchor='mm')
    for (r, c), (l, blank) in tiles.items():
        x, y = c*cell, r*cell
        d.rounded_rectangle([x+3, y+3, x+cell-4, y+cell-4], radius=6, fill=(238,222,185), outline=(180,160,120))
        if not blank:
            d.text((x+cell*0.45, y+cell*0.52), l, fill=(25,20,15), font=f, anchor='mm')
            d.text((x+cell*0.82, y+cell*0.8), str(VAL[l]), fill=(25,20,15), font=fs, anchor='mm')
    return im

def find_coeffs(dst, src):
    A = []
    for (x, y), (X, Y) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -X*x, -X*y]); A.append([0, 0, 0, x, y, 1, -Y*x, -Y*y])
    return np.linalg.solve(np.array(A, float), np.array(src, float).reshape(8)).tolist()

def photo(tiles, corners, size=(1280, 960), seed=0, bright=1.0, rot=0):
    rnd = random.Random(seed)
    flat = render_flat(tiles); S = flat.size[0]
    for _ in range(rot): flat = flat.transpose(Image.ROTATE_270)  # clockwise quarter-turns = viewing from another side
    # board frame margin around the grid
    m = 40; framed = Image.new('RGB', (S+2*m, S+2*m), (120, 30, 35)); framed.paste(flat, (m, m))
    # map grid corners -> photo corners, extend to frame
    src_sq = [(m, m), (S+m, m), (S+m, S+m), (m, S+m)]
    coeffs = find_coeffs(corners, src_sq)  # photo -> framed coords
    bg = Image.new('RGB', size, (150, 110, 70))
    nz = (np.random.default_rng(seed).normal(0, 10, (size[1], size[0], 1))).clip(-30, 30)
    bga = np.array(bg, float) + nz
    w = framed.transform(size, Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=(0, 0, 0))
    mask = Image.new('L', framed.size, 255).transform(size, Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=0)
    out = np.where(np.array(mask)[..., None] > 127, np.array(w, float), bga)
    out = out * bright + np.random.default_rng(seed + 1).normal(0, 4, out.shape)
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    return img

def word(tiles, r, c, w, vertical=False, blanks=()):
    for i, l in enumerate(w):
        if l == '_': continue
        tiles[(r + i*vertical, c + i*(not vertical))] = (l, i in blanks)

if __name__ == '__main__':
    out = sys.argv[1]
    corners = [(230, 120), (1060, 150), (1150, 900), (150, 870)]
    jitter = lambda k: [(x + random.Random(k*10+i).uniform(-3, 3), y + random.Random(k*10+i+5).uniform(-3, 3)) for i, (x, y) in enumerate(corners)]
    t = {}
    states = []
    states.append(dict(t))                                   # 0 empty
    word(t, 7, 6, 'QUIZ'); states.append(dict(t))            # 1 QUIZ H7..: (7,6)-(7,9) covers centre
    word(t, 8, 8, 'AX'); states.append(dict(t))              # 2 parallel AX under I,Z
    word(t, 3, 9, 'JOKE_', vertical=True, blanks=(3,))  # (6,9) blank E, runs into existing Z
    states.append(dict(t))
    meta = []
    for k, s in enumerate(states):
        img = photo(s, jitter(k), seed=k, bright=1.0 - 0.05*k)
        img.save(f'{out}/turn{k}.png')
        a = np.array(img.convert('RGBA')); a.tofile(f'{out}/turn{k}.rgba')
        meta.append({'file': f'turn{k}', 'w': img.size[0], 'h': img.size[1], 'tiles': [[r, c, l, b] for (r, c), (l, b) in sorted(s.items())]})
    rotated = []
    for rot in (1, 2, 3):   # turn-2 board seen from the other three sides, corners in shuffled order
        img = photo(states[2], jitter(10 + rot), seed=20 + rot, bright=0.97, rot=rot)
        name = f'turn2_rot{rot}'; img.save(f'{out}/{name}.png'); np.array(img.convert('RGBA')).tofile(f'{out}/{name}.rgba')
        rotated.append({'file': name, 'rot': rot, 'w': img.size[0], 'h': img.size[1], 'corners': [corners[(i + rot) % 4] for i in (2, 0, 3, 1)]})
    json.dump({'corners': corners, 'frames': meta, 'rotated': rotated}, open(f'{out}/meta.json', 'w'))
    print('ok')
