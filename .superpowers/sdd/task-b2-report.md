# Story B2 report — prefer mid-window hook candidates

Commit: `e8fdebb95ac95208f249e2d5975fe07654ca68c1` (unsigned, `git log --format=%G? -1` → `N`)
Branch: `feat/improve-video`

## What changed and why

### `reels-clips-automator/reelsfy.py`

1. **Two new module constants** next to `HOOK_DUPLICATE_GUARD_SECONDS` (line ~4784):

   ```python
   HOOK_MID_BAND_START_FRACTION = 0.20
   HOOK_MID_BAND_END_FRACTION = 0.90
   ```

   with a comment explaining that the mid band is a *preference*: the first 20 %
   leads with a truncated thought, the last 10 % pre-empts the payoff.

2. **`select_hook_candidate` (~line 2632)** — the loop body is unchanged; only the
   *commit point* moved. Before, the function `return`ed the first survivor. Now it:

   - computes `window_duration`, `guard_end`, `band_start`, `band_end` **once** before
     the loop (this is also ride-along cleanup #2 — B1 recomputed `guard_end` every
     iteration even though it is invariant);
   - builds the same survivor dict as before and, if
     `band_start - 1e-9 <= span_start <= band_end + 1e-9`, returns it immediately;
   - otherwise stores it in `first_survivor` (only if not already set) and continues;
   - returns `first_survivor` at the end instead of `None`.

   Both tiers preserve the LLM's original candidate order, so "first" is still list
   order — never timestamp or shortest span. Band bounds are inclusive with the
   neighbouring `- 1e-9` / `+ 1e-9` epsilon style, so a candidate exactly on a
   boundary counts as inside.

3. **Docstring** — a `B2:` paragraph added directly after the `B1:` one, stating the
   band, the rationale, and that it is a preference rather than a filter.

Nothing else was touched: the outer `try/except` with the "keeping the hook optional"
print, the per-candidate `except Exception: continue`, and the `window_start` /
`window_end` computation are all unchanged, so the degradation and recency-anchor
invariants hold by construction.

### `tests/test_hook_first.py`

- **Rename** (ride-along cleanup #1): `test_span_overlapping_the_guard_rejected` →
  `test_span_straddling_the_guard_end_rejected`, and the comment now says the span
  straddles the guard *end*. Only the name/comment changed; the assertions and the
  candidate exercised are identical (verified: the full B1 class still passes).
- **New `TestMidBandPreference` class** with a `B2_WINDOW_*` pair (window 25–85 s,
  duration 60 s → guard 25–30 s, mid band 37–79 s) and a `BAND_CUES` fixture:

  | cue | span | position |
  |---|---|---|
  | 1 | 25–28 | inside the opening guard (rejected) |
  | 2 | 31–34 | ~10 % — edge, past the guard |
  | 3 | 52–55 | ~45 % — mid band |
  | 4 | 60–63 | ~58 % — mid band |
  | 5 | 80–83 | ~92 % — last 10 % |
  | 6 | 37–40 | exactly on the mid-band start boundary |

  10 tests: mid-band preferred over an earlier-listed edge candidate; preferred over a
  later-listed edge candidate; a single early edge candidate still selected; a single
  late edge candidate still selected; first of two mid-band candidates wins; boundary
  candidate counts as inside; an all-edge ranking is byte-identical to pre-B2 order;
  empty list → `None`; all-malformed list → `None`; and the two fraction constants.

## TDD sequence

1. Wrote the 10 new tests + the rename first.
2. `./.venv/bin/python -m pytest tests/test_hook_first.py -q` →
   **`4 failed, 42 passed`**. The 4 failures were the 3 ordering tests and the constant
   test; the 6 tests that describe *preserved* behaviour (edge-still-selected,
   first-of-tier, empty, malformed, byte-identity) passed even before the
   implementation, which is the intended evidence that B2 is a pure preference.
3. Implemented the constants + the two-pass commit point.
4. Re-ran → all green.

## Test commands and output

```
$ ./.venv/bin/python -m pytest tests/test_hook_first.py tests/test_hook_teaser.py -q
98 passed in 2.34s

$ ./.venv/bin/python -m pytest tests/ -q --continue-on-collection-errors
75 failed, 759 passed, 5 skipped, 22 warnings, 7 errors in 41.95s
```

Baseline measured the same way on this machine, with `reelsfy.py` **and**
`tests/test_hook_first.py` restored from `HEAD` (6778f3e) and both re-run to defeat
flakiness:

```
75 failed, 749 passed, 5 skipped, 22 warnings, 7 errors in 41.20s
```

`diff` of the sorted `FAILED` node-id lists (error text stripped) is **empty** —
same 75 pre-existing environmental failures before and after, and the pass count rises
by exactly the 10 new tests. The brief quotes 743/75; this box measures 749/75 for the
same HEAD, so the delta from the quoted figure is pre-existing, not from B2.

## Byte-identity verification when no mid-band candidate exists

Beyond the unit test, I ran a randomized differential check: the HEAD (pre-B2)
`select_hook_candidate` was loaded side by side with the new one as a second module
(`git show HEAD:... > _b1_reelsfy_tmp.py`, imported via `importlib`, temp file deleted
afterwards) and both were called on 4000 generated layouts — random cue counts (4–9)
covering a fixed 25–85 s window, random candidate lists of 0–5 entries with random
`cue_start`/`cue_end`/`text`. Results were compared as parsed values.

- **4000 trials, 59 mismatches.**
- Every one of the 59 was classified and confirmed to be the *intended* B2 change:
  the new function returned a survivor whose start lay inside the mid band, while the
  old function returned an earlier edge survivor. No mismatches had `None` on one side
  only, and no mismatch was outside that pattern (the script printed
  `UNEXPECTED DIFF` / `SHAPE MISMATCH` and broke otherwise — it never did).
- The other 3941 trials were byte-identical, including the empty / all-malformed /
  all-rejected cases, which return `None` on both sides.

So: whenever no mid-band candidate exists, the output is the same candidate, the same
two SRT strings and the same `hook_text` as before B2. A separate committed test
(`test_edge_only_ranking_is_byte_identical_to_pre_b2_order`) pins this as a literal
expected dict.

## Concerns / uncertainties

- The measured full-suite baseline on this machine is **749/75**, not the 743/75 in the
  brief. The failure *sets* are identical, so I am confident this is drift in unrelated
  tests rather than anything B2 touched, but I could not explain the 6-test difference.
- The differential harness compares parsed return values, not raw bytes of any
  serialized artifact. The two SRT strings and the text are the function's entire
  output, so this is equivalent in practice, but it is not literally a byte comparison.
- `band_end` is computed and used but the "last 10 %" edge is only ever reachable via
  the *end* of the window; a candidate can never start past `band_end` and still be
  inside the window for long, since `band_end` is 90 % of the window. The end bound is
  therefore nearly always satisfied — kept because the brief specifies the two-fraction
  band and the inclusive boundary semantics.
- I did not touch `HOOK_TEASER_MIN_DURATION`/`MAX_DURATION` or add env gating for the
  fractions; the brief specified named module constants, not env knobs.
