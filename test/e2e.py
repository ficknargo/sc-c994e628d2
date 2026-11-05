"""Browser e2e: synthetic board photos fed as camera frames (?test), vision API mocked. Run from repo root."""
import json, subprocess, time, sys, os
from playwright.sync_api import sync_playwright

meta = json.load(open('test/fixtures/meta.json'))
W, H = meta['frames'][0]['w'], meta['frames'][0]['h']
corners = [{'x': x / W, 'y': y / H} for x, y in meta['corners']]
truth = [{(r, c): (l, b) for r, c, l, b in f['tiles']} for f in meta['frames']]
srv = subprocess.Popen([sys.executable, '-m', 'http.server', '8765'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
os.makedirs('test/shots', exist_ok=True)
calls = []
try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={'width': 390, 'height': 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
        pg.on('console', lambda m: print('console:', m.text) if m.type in ('error', 'warning') else None)
        pg.on('dialog', lambda d: d.accept())
        frame_idx = {'k': 0}
        def handle(route):
            body = json.loads(route.request.post_data)
            calls.append(body['model'])
            cells = pg.evaluate('window.__lastCells')
            t = truth[frame_idx['k']]
            tiles = [{'n': i + 1, 'letter': ('?' if t[(r, c)][1] else t[(r, c)][0]) if (r, c) in t else '-'} for i, (r, c) in enumerate(cells)]
            route.fulfill(status=200, content_type='application/json', body=json.dumps({'choices': [{'message': {'content': json.dumps({'tiles': tiles})}}]}))
        pg.route('https://api.openai.com/**', handle)
        pg.goto('http://localhost:8765/index.html?test')
        pg.click('#count button[data-n="3"]'); assert pg.locator('#names input').count() == 3
        pg.click('#count button[data-n="2"]')
        pg.screenshot(path='test/shots/setup.png')
        pg.evaluate("""c => localStorage.setItem('scrabblecam.game.v1', JSON.stringify({players:['Ann','Bob'],history:[],corners:c,baseline:null}))""", corners)
        pg.reload()
        pg.evaluate("window.scrabbleCam.setFrame('test/fixtures/turn0.png')")
        pg.click('#resumeGame'); time.sleep(0.5)
        pg.click('#alignBtn'); pg.click('#alignBtn')   # align done -> sync baseline from empty board
        pg.screenshot(path='test/shots/0-empty.png')

        # Turn 1: no API key -> diff finds tiles, user types QUIZ
        frame_idx['k'] = 1; pg.evaluate("window.scrabbleCam.setFrame('test/fixtures/turn1.png')")
        pg.click('#scoreBtn'); pg.wait_for_selector('#editor:not(.hidden)')
        asks = pg.eval_on_selector_all('.cell.ask', 'els => els.map(e => [+e.dataset.r, +e.dataset.c])')
        print('turn1 highlighted cells:', asks)
        assert sorted(map(tuple, asks)) == [(7, 6), (7, 7), (7, 8), (7, 9)], asks
        for r, c in sorted(map(tuple, asks)):
            pg.click(f'.cell[data-r="{r}"][data-c="{c}"]'); pg.click(f'#keys button[data-k="{truth[1][(r, c)][0]}"]')
        pg.screenshot(path='test/shots/1-editor.png')
        print('editor result:', pg.inner_text('#edResult'))
        pg.click('#edSave'); time.sleep(0.3)
        pg.screenshot(path='test/shots/1-flash.png')
        print('flash:', pg.inner_text('#flash').replace('\n', ' '))

        # Turn 2: with (mocked) API key -> fully automatic
        pg.evaluate("localStorage.setItem('scrabblecam.settings.v1', JSON.stringify({apiKey:'sk-test',model:'gpt-4.1-mini'}))")
        pg.reload(); pg.evaluate("window.scrabbleCam.setFrame('test/fixtures/turn2.png')"); pg.click('#resumeGame')
        frame_idx['k'] = 2; time.sleep(0.3)
        pg.click('#scoreBtn'); pg.wait_for_selector('#flash:not(.hidden)', timeout=10000)
        print('turn2 flash:', pg.inner_text('#flash').replace('\n', ' '))
        assert pg.is_hidden('#editor')

        # Phone moved to the other side (90°), corners dragged in a shuffled order -> re-align picks the rotation
        rf = meta['rotated'][0]
        def set_corners(cs):
            pg.evaluate("c => { window.scrabbleCam.game.corners = c; }", [{'x': x / W, 'y': y / H} for x, y in cs])
        pg.click('#flash')
        pg.evaluate(f"window.scrabbleCam.setFrame('test/fixtures/{rf['file']}.png')"); set_corners(rf['corners'])
        pg.click('#alignBtn'); pg.click('#alignBtn'); time.sleep(0.2)
        print('rotation after moving phone:', pg.evaluate('window.scrabbleCam.game.rot'), 'expected', (4 - rf['rot']) % 4)
        assert pg.evaluate('window.scrabbleCam.game.rot') == (4 - rf['rot']) % 4
        pg.screenshot(path='test/shots/2-rotated.png')
        pg.evaluate("window.scrabbleCam.setFrame('test/fixtures/turn2.png')"); set_corners(meta['corners'])
        pg.click('#alignBtn'); pg.click('#alignBtn'); time.sleep(0.2)
        assert pg.evaluate('window.scrabbleCam.game.rot') == 0

        # Turn 3: includes a blank tile -> editor asks for its letter only
        frame_idx['k'] = 3; pg.evaluate("window.scrabbleCam.setFrame('test/fixtures/turn3.png')"); time.sleep(0.3)
        pg.click('#scoreBtn'); pg.wait_for_selector('#editor:not(.hidden)', timeout=10000)
        print('turn3 hint:', pg.inner_text('#edHint'))
        pg.click('#keys button[data-k="E"]')
        print('turn3 editor result:', pg.inner_text('#edResult'))
        pg.click('#edSave'); time.sleep(0.3)
        print('turn3 flash:', pg.inner_text('#flash').replace('\n', ' '))
        pg.screenshot(path='test/shots/3-game.png')
        st = pg.evaluate('window.scrabbleCam.state().scores')
        print('scores:', st)
        # history edit: change turn-2 player to Ann? just open history
        pg.click('#flash'); pg.click('#historyBtn'); pg.screenshot(path='test/shots/4-history.png')
        print(pg.inner_text('#histList'))
        pg.click('#history .closeSheet')
        # stats before end: then apply penalties
        pg.click('#endBtn'); pg.select_option('#outPlayer', '0'); pg.fill('#racks input[data-i="1"]', 'QE?')
        print('preview:', pg.inner_text('#racks').replace('\n', ' '))
        pg.click('#applyPenalty'); time.sleep(0.2)
        print('final scores:', pg.evaluate('window.scrabbleCam.state().scores'))
        pg.screenshot(path='test/shots/5-stats.png', full_page=True)
        print(pg.inner_text('#final'))
        pg.click('#endgame .closeSheet')
        pg.click('#undoBtn'); time.sleep(0.2)
        print('after undo:', pg.evaluate('window.scrabbleCam.state().scores'))
        print('api calls:', calls)
        b.close()
finally:
    srv.terminate()
