"""Held-out handheld photo sequences (held-out fonts) for detection + full-pipeline evaluation in node.
Writes tools/data/eval/g{game}_{turn}.rgba and meta.json"""
import sys, json, random, os, numpy as np
sys.path.insert(0, 'tools'); import synth
out = 'tools/data/eval'; os.makedirs(out, exist_ok=True)
ngames = int(sys.argv[1]) if len(sys.argv) > 1 else 8
turns = int(sys.argv[2]) if len(sys.argv) > 2 else 6
meta = []
for g in range(ngames):
    r = random.Random(1000 + g)
    st = synth.random_style(r, synth.TEST_FONTS)
    states, moves = synth.grow_game(r, turns, tile_rot_mix=st['rot_tiles'])
    for t, s in enumerate(states):
        img, corners, _ = synth.photo(s, r, style=st)
        name = f'g{g}_{t}'
        np.dstack([img, np.full(img.shape[:2], 255, np.uint8)]).tofile(f'{out}/{name}.rgba')
        meta.append({'file': name, 'game': g, 'turn': t, 'w': img.shape[1], 'h': img.shape[0], 'corners': corners.tolist(),
                     'tiles': [[r_, c_, l, bool(b), q] for (r_, c_), (l, b, q) in sorted(s.items())],
                     'move': [list(c) for c in moves[t - 1]] if t else []})
        if g < 2 and t == turns: synth.Image.fromarray(img).save(f'{out}/{name}.jpg')
json.dump(meta, open(f'{out}/meta.json', 'w'))
print(len(meta), 'frames')
