# Scrabble Cam

Static mobile web app that keeps Scrabble score from a live camera view of the board.

* Setup 2–4 players → point the rear camera at the board (phone on a stand works best) → drag the 4 dots onto the corners of the 15×15 grid once.
* After each turn tap **Score turn**: the current frame is grabbed in memory (nothing saved), perspective-warped to 15×15, diffed against the previous frame to find new tiles, letters are read, the move is scored (all words, premiums only when first covered, 50-pt bingo, blanks 0), added to the current player, and play rotates.
* Corrections: Undo, History (tap a turn to fix letters/blanks/tiles/player — all scores recalc), tap a player's score to adjust, End → leftover-tile penalties. Game persists in localStorage.

## Recognition
1. Cell diff (client-side, `js/geom.js`): per-cell normalized texture+color features vs. the last frame, tolerant to ±1 px-ish jitter and brightness changes; picks the outlier cells in one row/column.
2. Letters: with an OpenAI-compatible API key (Settings; stored in localStorage only) a contact sheet of just the changed cells is sent to a vision model (default `gpt-4.1-mini`, strict JSON schema). If no diff baseline exists, the whole labelled board is sent instead.
3. No key → the highlighted cells open in the editor and you type the letters. Blank tiles always ask for the letter they represent.

## Tests
```
npm test                                   # scoring + CV pipeline unit tests (node --test)
python3 test/make_boards.py test/fixtures  # synthetic perspective board photos (needed for vision/e2e tests)
python3 test/e2e.py                        # Playwright: frames fed as camera, vision API mocked
```
