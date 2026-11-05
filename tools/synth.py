"""Synthetic Scrabble board photos (handheld, any rotation, clutter) + ground truth.
render_board(state, style) -> flat RGB board (grid exactly at [m, m+15*cell]); photo(...) -> image + grid corners in photo.
"""
import random, math, glob, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PREM = {}
for k in ['0,0','0,7','0,14','7,0','7,14','14,0','14,7','14,14']: PREM[k]='TW'
for k in ['1,1','2,2','3,3','4,4','10,10','11,11','12,12','13,13','1,13','2,12','3,11','4,10','10,4','11,3','12,2','13,1','7,7']: PREM[k]='DW'
for k in ['1,5','1,9','5,1','5,5','5,9','5,13','9,1','9,5','9,9','9,13','13,5','13,9']: PREM[k]='TL'
for k in ['0,3','0,11','2,6','2,8','3,0','3,7','3,14','6,2','6,6','6,8','6,12','7,3','7,11','8,2','8,6','8,8','8,12','11,0','11,7','11,14','12,6','12,8','14,3','14,11']: PREM[k]='DL'
VAL = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',[1,3,3,2,1,4,2,4,1,8,5,1,3,1,1,3,10,1,1,1,1,4,4,8,4,10]))
FREQ = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',[9,2,2,4,12,2,3,2,9,1,1,4,2,6,8,2,1,6,4,6,4,2,2,1,2,1]))

GF = '/usr/share/fonts/truetype/sand-box/google/'
FONT_FILES = {
 'Lato':'Lato/Lato-Bold.ttf','Poppins':'Poppins/Poppins-Bold.ttf','PTSans':'PT Sans/PTSans-Bold.ttf','Ubuntu':'Ubuntu/Ubuntu-Bold.ttf',
 'Fira':'Fira Sans/FiraSans-Bold.ttf','Barlow':'Barlow/Barlow-Bold.ttf','Tinos':'Tinos/Tinos-Bold.ttf','Kanit':'Kanit/Kanit-Bold.ttf',
 'Crimson':'Crimson Text/CrimsonText-Bold.ttf','Arvo':'Arvo/Arvo-Bold.ttf','Zilla':'Zilla Slab/ZillaSlab-Bold.ttf','Solway':'Solway/Solway-Bold.ttf',
 'Caladea':'Caladea/Caladea-Bold.ttf','GothicA1':'Gothic A1/GothicA1-Bold.ttf','Hind':'Hind/Hind-Bold.ttf',
 'Inter':'Inter/Inter-VariableFont_opsz,wght.ttf','Jost':'Jost/Jost-VariableFont_wght.ttf','Spartan':'League Spartan/LeagueSpartan-VariableFont_wght.ttf',
 'Lora':'Lora/Lora-VariableFont_wght.ttf','Merri':'Merriweather/Merriweather-VariableFont_opsz,wdth,wght.ttf','MerriSans':'Merriweather Sans/MerriweatherSans-VariableFont_wght.ttf',
 'Montserrat':'Montserrat/Montserrat-VariableFont_wght.ttf','OpenSans':'Open Sans/OpenSans-VariableFont_wdth,wght.ttf','Roboto':'Roboto/Roboto-VariableFont_wdth,wght.ttf',
 'RobotoCond':'Roboto Condensed/RobotoCondensed-VariableFont_wght.ttf','RobotoSlab':'Roboto Slab/RobotoSlab-VariableFont_wght.ttf','Rubik':'Rubik/Rubik-VariableFont_wght.ttf',
 'Source':'Source Sans 3/SourceSans3-VariableFont_wght.ttf','NotoSans':'Noto Sans/NotoSans-VariableFont_wdth,wght.ttf','NotoSerif':'Noto Serif/NotoSerif-VariableFont_wdth,wght.ttf',
 'WorkSans':'Work Sans/WorkSans-VariableFont_wght.ttf','Nunito':'Nunito/Nunito-VariableFont_wght.ttf','Cabin':'Cabin/Cabin-VariableFont_wdth,wght.ttf',
 'Arimo':'Arimo/Arimo-VariableFont_wght.ttf','Bitter':'Bitter/Bitter-VariableFont_wght.ttf','Franklin':'Libre Franklin/LibreFranklin-VariableFont_wght.ttf',
 'Josefin':'Josefin Sans/JosefinSans-VariableFont_wght.ttf','Plex':'IBM Plex Sans/IBMPlexSans-VariableFont_wdth,wght.ttf','Baskerville':'Libre Baskerville/LibreBaskerville-VariableFont_wght.ttf',
 'DMSans':'DM Sans/DMSans-VariableFont_opsz,wght.ttf','Archivo':'Archivo/Archivo-VariableFont_wdth,wght.ttf','Oswald':'Oswald/Oswald-VariableFont_wght.ttf',
 'Playfair':'Playfair Display/PlayfairDisplay-VariableFont_wght.ttf','Domine':'Domine/Domine-VariableFont_wght.ttf','Alegreya':'Alegreya/Alegreya-VariableFont_wght.ttf',
 'DejaVu':'/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
}
TEST_FONTS = ['Montserrat', 'Tinos', 'Rubik', 'NotoSerif', 'Barlow', 'Franklin']   # held out from training
TRAIN_FONTS = [f for f in FONT_FILES if f not in TEST_FONTS]
_fc = {}
def font(name, size):
    k = (name, size)
    if k not in _fc:
        p = FONT_FILES[name]; p = p if p.startswith('/') else GF + p
        f = ImageFont.truetype(p, size)
        try: f.set_variation_by_axes([700 if a.get('name', b'') in (b'Weight', 'Weight') else a['default'] for a in f.get_variation_axes()])
        except Exception: pass
        _fc[k] = f
    return _fc[k]

def jit(c, r, a=18):
    return tuple(int(max(0, min(255, v + r.uniform(-a, a)))) for v in c)

def random_style(r, fonts):
    plain = r.choice([(205,200,170),(200,205,180),(215,205,175),(190,200,175),(225,220,200),(170,190,160),(235,230,215),(60,110,90)])
    s = dict(
        plain=jit(plain, r), TW=jit(r.choice([(205,60,55),(230,90,60),(220,40,50),(240,120,60)]), r),
        DW=jit(r.choice([(240,170,170),(245,180,190),(230,140,150),(250,200,180)]), r),
        TL=jit(r.choice([(60,110,190),(40,90,170),(30,120,160),(70,80,170)]), r),
        DL=jit(r.choice([(170,210,235),(150,200,230),(180,220,220),(160,190,240)]), r),
        line=r.choice([(255,255,255),(235,230,215),(40,60,50),(120,120,110),(200,190,160)]), lw=r.uniform(0.02, 0.07),
        labels=r.choice(['none','abbr','long','long']), star=r.random() < 0.7,
        border=jit(r.choice([(120,30,35),(30,70,50),(20,20,25),(150,110,70),(60,40,30),(200,190,170)]), r), bw=r.uniform(0.02, 0.09),
        tile=jit(r.choice([(238,222,185),(245,235,210),(230,210,170),(250,245,235),(225,200,150),(240,228,190)]), r, 10),
        ink=jit(r.choice([(25,20,15),(10,10,10),(50,35,25),(30,30,40)]), r, 10),
        font=r.choice(fonts), lsize=r.uniform(0.55, 0.72), vsize=r.uniform(0.2, 0.28), lshift=r.uniform(-0.06, 0.0),
        tsize=r.uniform(0.86, 0.95), tround=r.uniform(0.05, 0.2), shadow=r.uniform(0, 0.08), tilejit=r.uniform(0, 0.06), tilerot=r.uniform(0, 5),
        rot_tiles=r.random() < 0.25,
    )
    return s

LABELS = {'TW': ['TRIPLE', 'WORD', 'SCORE'], 'DW': ['DOUBLE', 'WORD', 'SCORE'], 'TL': ['TRIPLE', 'LETTER', 'SCORE'], 'DL': ['DOUBLE', 'LETTER', 'SCORE']}

def draw_tile(letter, blank, st, cell, r):
    """Return RGBA tile image of size cell x cell (upright)."""
    T = Image.new('RGBA', (cell, cell), (0, 0, 0, 0)); d = ImageDraw.Draw(T)
    ts = st['tsize'] * cell; o = (cell - ts) / 2
    sh = st['shadow'] * cell
    if sh > 0.5: d.rounded_rectangle([o + sh, o + sh, o + ts + sh * 0.6, o + ts + sh * 0.6], radius=st['tround'] * ts, fill=(0, 0, 0, 90))
    tc = st['tile']
    d.rounded_rectangle([o, o, o + ts, o + ts], radius=st['tround'] * ts, fill=tc + (255,), outline=tuple(max(0, v - 40) for v in tc) + (255,), width=max(1, int(cell * 0.02)))
    if not blank:
        f = font(st['font'], max(8, int(ts * st['lsize'])))
        d.text((cell / 2 + st['lshift'] * ts, cell / 2 + st['lshift'] * ts * 0.5), letter, fill=st['ink'] + (255,), font=f, anchor='mm')
        fv = font(st['font'], max(6, int(ts * st['vsize'])))
        d.text((o + ts * 0.86, o + ts * 0.84), str(VAL[letter]), fill=st['ink'] + (255,), font=fv, anchor='mm')
    return T

def render_board(tiles, st, r, cell=64):
    """tiles: {(row,col): (letter, blank, rot)} rot = CW quarter-turns of the tile relative to the board.
    Returns flat image and margin m (grid occupies [m, m+15*cell])."""
    S = 15 * cell; m = int(st['bw'] * S) + 2
    im = Image.new('RGB', (S + 2 * m, S + 2 * m), st['border']); d = ImageDraw.Draw(im)
    d.rectangle([m, m, m + S, m + S], fill=st['line'])
    lw = max(1, int(st['lw'] * cell))
    small = font('DejaVu', max(6, int(cell * 0.14)))
    for row in range(15):
        for col in range(15):
            p = PREM.get(f'{row},{col}', ''); x, y = m + col * cell, m + row * cell
            d.rectangle([x + lw / 2, y + lw / 2, x + cell - lw / 2, y + cell - lw / 2], fill=st[p] if p else st['plain'])
            if p and st['labels'] != 'none':
                tc = (255, 255, 255) if p in ('TW', 'TL') else (40, 40, 40)
                if st['labels'] == 'abbr': d.text((x + cell / 2, y + cell / 2), p, fill=tc, font=font('DejaVu', int(cell * 0.3)), anchor='mm')
                else:
                    for i, w in enumerate(LABELS[p]): d.text((x + cell / 2, y + cell * (0.3 + 0.2 * i)), w, fill=tc, font=small, anchor='mm')
            if row == 7 and col == 7 and st['star']:
                cx, cy, R = x + cell / 2, y + cell / 2, cell * 0.32
                pts = [(cx + (R if k % 2 == 0 else R * 0.45) * math.sin(k * math.pi / 5), cy - (R if k % 2 == 0 else R * 0.45) * math.cos(k * math.pi / 5)) for k in range(10)]
                d.polygon(pts, fill=(30, 30, 30))
    for (row, col), (letter, blank, rot) in tiles.items():
        T = draw_tile(letter, blank, st, cell, r)
        ang = -90 * rot + r.uniform(-st['tilerot'], st['tilerot'])   # PIL rotate is CCW
        T = T.rotate(ang, resample=Image.BICUBIC)
        dx, dy = (r.uniform(-1, 1) * st['tilejit'] * cell for _ in range(2))
        im.paste(T, (int(m + col * cell + dx), int(m + row * cell + dy)), T)
    return im, m

def clutter_bg(w, h, r):
    rng = np.random.default_rng(r.randrange(1 << 30))
    base = np.array(jit(r.choice([(150,110,70),(90,60,40),(200,200,195),(60,60,65),(170,150,120),(120,130,140),(230,225,215)]), r, 25), float)
    img = np.ones((h, w, 3)) * base
    if r.random() < 0.6:  # wood grain
        y = np.arange(h)[:, None]; x = np.arange(w)[None, :]
        img += (np.sin((y + 30 * np.sin(x / r.uniform(50, 200))) / r.uniform(3, 12)) * r.uniform(5, 20))[..., None]
    img += rng.normal(0, r.uniform(2, 12), (h, w, 1))
    pim = Image.fromarray(img.clip(0, 255).astype(np.uint8)); d = ImageDraw.Draw(pim)
    for _ in range(r.randint(0, 14)):   # objects: cups, papers, racks with tiles, phones
        x, y, s = r.uniform(-100, w), r.uniform(-100, h), r.uniform(30, 300)
        col = tuple(r.randint(0, 255) for _ in range(3)); k = r.random()
        if k < 0.3: d.ellipse([x, y, x + s, y + s * r.uniform(0.5, 1.5)], fill=col)
        elif k < 0.6: d.rectangle([x, y, x + s, y + s * r.uniform(0.2, 1.5)], fill=col)
        elif k < 0.8:  # paper with score lines
            d.rectangle([x, y, x + s, y + s * 1.3], fill=(245, 245, 240))
            for i in range(8): d.line([x + 5, y + 15 + i * s / 7, x + s - 5, y + 15 + i * s / 7], fill=(100, 100, 200), width=2)
        else:  # loose tiles / rack
            for i in range(r.randint(1, 7)):
                t = draw_tile(r.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), False, dict(tsize=0.95, tround=0.1, shadow=0.02, tile=(238, 222, 185), ink=(20, 20, 20), font='DejaVu', lsize=0.6, vsize=0.24, lshift=0, tilejit=0, tilerot=0), int(s / 5) + 20, r)
                pim.paste(t, (int(x + i * (s / 5 + 22)), int(y)), t)
    return np.array(pim, float)

def photo(tiles, r, style=None, size=None, fonts=TRAIN_FONTS, cell=64, hard=True):
    """Simulated handheld phone frame. Returns (uint8 RGB HxWx3, grid corners 4x2 float [TL,TR,BR,BL of the board's canonical grid], style)."""
    st = style or random_style(r, fonts)
    flat, m = render_board(tiles, st, r, cell)
    Sf = flat.size[0]
    W, H = size or r.choice([(1280, 960), (960, 1280), (1280, 720), (720, 1280)])
    # board quad in photo: random scale, rotation, perspective tilt, translation (board fully visible)
    for _ in range(100):
        scale = r.uniform(0.55, 0.95) * min(W, H)
        ang = r.uniform(0, 2 * math.pi)
        tilt = r.uniform(0, 0.35 if hard else 0.15)
        tdir = r.uniform(0, 2 * math.pi)
        sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * (Sf / 2)
        # perspective: 3D rotate square about axis tdir by tilt, project
        ca, sa = math.cos(ang), math.sin(ang)
        pts = sq @ np.array([[ca, sa], [-sa, ca]])
        u = np.array([math.cos(tdir), math.sin(tdir)])
        dpt = pts @ u
        f = 1.0 / (1 + np.tan(tilt) * dpt / (Sf * 1.2))
        pts = pts * f[:, None]
        pts = pts / (np.ptp(pts, 0).max()) * scale
        cx, cy = r.uniform(0.1, 0.9) * W, r.uniform(0.1, 0.9) * H
        pts = pts - pts.mean(0) + [cx, cy]
        if pts[:, 0].min() > 4 and pts[:, 1].min() > 4 and pts[:, 0].max() < W - 4 and pts[:, 1].max() < H - 4: break
    src = np.array([[0, 0], [Sf, 0], [Sf, Sf], [0, Sf]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, pts.astype(np.float32))
    bg = clutter_bg(W, H, r)
    fl = np.array(flat)
    warped = cv2.warpPerspective(fl, Hm, (W, H), flags=cv2.INTER_AREA if scale < Sf else cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones(fl.shape[:2], np.float32), Hm, (W, H))
    img = bg * (1 - mask[..., None]) + warped * mask[..., None]
    # lighting: gradient, colour cast, shadow, highlight
    yy, xx = np.mgrid[0:H, 0:W]
    g = 1 + r.uniform(-0.3, 0.3) * ((xx / W - 0.5) * math.cos(r.uniform(0, 6.3)) + (yy / H - 0.5) * math.sin(r.uniform(0, 6.3)))
    img = img * g[..., None] * np.array([r.uniform(0.85, 1.1) for _ in range(3)]) * r.uniform(0.6, 1.15)
    if r.random() < 0.4:  # soft shadow (hand/phone)
        sm = np.zeros((H, W), np.float32)
        cv2.ellipse(sm, (int(r.uniform(0, W)), int(r.uniform(0, H))), (int(r.uniform(80, 400)), int(r.uniform(80, 400))), r.uniform(0, 180), 0, 360, 1, -1)
        sm = cv2.GaussianBlur(sm, (0, 0), 40)
        img *= (1 - r.uniform(0.2, 0.5) * sm)[..., None]
    if r.random() < 0.3:  # glare
        gm = np.zeros((H, W), np.float32)
        cv2.circle(gm, (int(r.uniform(0, W)), int(r.uniform(0, H))), int(r.uniform(20, 120)), 1, -1)
        img += (cv2.GaussianBlur(gm, (0, 0), 30) * r.uniform(40, 120))[..., None]
    img = img.clip(0, 255).astype(np.uint8)
    if r.random() < 0.7: img = cv2.GaussianBlur(img, (0, 0), r.uniform(0.3, 1.6))
    if r.random() < 0.2:  # motion blur
        k = r.randint(3, 9); ker = np.zeros((k, k)); ker[k // 2] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), r.uniform(0, 180), 1); ker = cv2.warpAffine(ker, M, (k, k)); ker /= ker.sum()
        img = cv2.filter2D(img, -1, ker)
    img = np.clip(img.astype(float) + np.random.default_rng(r.randrange(1 << 30)).normal(0, r.uniform(1, 6), img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode('.jpg', img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, r.randint(55, 95)])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    grid = np.array([[m, m], [m + 15 * cell, m], [m + 15 * cell, m + 15 * cell], [m, m + 15 * cell]], np.float32)
    corners = cv2.perspectiveTransform(grid[None], Hm)[0]
    return np.ascontiguousarray(img), corners, st

def random_tiles(r, n=None, style=None):
    n = r.randint(0, 110) if n is None else n
    cells = r.sample(range(225), n); out = {}
    letters = ''.join(k * v for k, v in FREQ.items())
    rot_all = style['rot_tiles'] if style else False
    for i in cells:
        blank = r.random() < 0.03
        rot = r.randint(0, 3) if rot_all and r.random() < 0.5 else 0
        out[(i // 15, i % 15)] = (r.choice(letters), blank, rot)
    return out

def grow_game(r, nmoves, tile_rot_mix=False):
    """Sequence of board states via plausible moves (line placements touching existing tiles)."""
    letters = ''.join(k * v for k, v in FREQ.items())
    board = {}; states = [dict(board)]; moves = []
    for mv in range(nmoves):
        for _ in range(200):
            horiz = r.random() < 0.5
            if not board:
                L = r.randint(2, 7); start = r.randint(7 - L + 1, 7); line = 7
                cells = [(line, start + i) if horiz else (start + i, line) for i in range(L)]
            else:
                ar, ac = r.choice(list(board))
                L = r.randint(2, 8); off = r.randint(0, L - 1)
                cells = [(ar, ac - off + i) if horiz else (ar - off + i, ac) for i in range(L)]
            if any(not (0 <= a < 15 and 0 <= b < 15) for a, b in cells): continue
            new = [c for c in cells if c not in board]
            if not (1 <= len(new) <= 7): continue
            break
        else: continue
        mvp = []
        for c in new:
            blank = r.random() < 0.04
            rot = r.randint(0, 3) if tile_rot_mix and r.random() < 0.3 else 0
            board[c] = (r.choice(letters), blank, rot); mvp.append(c)
        states.append(dict(board)); moves.append(mvp)
    return states, moves
