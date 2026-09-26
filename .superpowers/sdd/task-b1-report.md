# Story B1 — Anti-duplication rule in hook selection

## What changed and why

`select_hook_candidate` returned the first candidate whose cue span lies inside
the resolved window and is 1.5-6 s long. Since the window starts at the topic
start, that first candidate was usually the topic's opening sentence — so the
prepended cold-open teaser replayed the clip's first words. B1 rejects any
candidate overlapping the first ~5 s of the resolved window.

### `reels-clips-automator/reelsfy.py`

1. New named constant, placed with the other hook constants (next to
   `HOOK_TEASER_MIN_DURATION` / `HOOK_TEASER_MAX_DURATION`):

   ```python
   # Story B1: the cold-open teaser is prepended to the clip, so a teaser taken
   # from the clip's own first seconds plays the same words twice. Any candidate
   # overlapping this opening span of the resolved window is rejected; the clip
   # then simply publishes without a teaser.
   HOOK_DUPLICATE_GUARD_SECONDS = 5.0
   ```

2. `select_hook_candidate` docstring: a B1 paragraph explaining why the guard
   exists (the teaser is prepended, so a candidate from the clip's own opening
   replays the first sentence; the window starts at the topic start, which is
   where the LLM's top-ranked candidate tends to sit; with no survivor the
   teaser is skipped).

3. One new guard clause, appended after the existing duration check and before
   the `return`, inside the per-candidate `try`:

   ```python
   guard_end = window_start + HOOK_DUPLICATE_GUARD_SECONDS
   if span_start < guard_end - 1e-9:
       continue
   ```

   The `- 1e-9` tolerance matches the style of the neighbouring containment and
   duration comparisons.

Everything else is untouched: the containment check, the duration check, the
`continue`-on-malformed behaviour, the outer `try/except` and its "keeping the
hook optional" print, and the window start/end computation (the recency anchor
is unchanged — only the start moves, and this change does not move it).

### `tests/test_hook_first.py`

Added a `GUARD_CUES` fixture (a 25-60 s window whose cue 1 is 25-28 s, cue 2 is
28-33 s, cue 3 is 35-40 s, cue 4 is 45-60 s) and a new class
`TestOpeningDuplicateGuard` with six cases:

| test | what it asserts |
|---|---|
| `test_candidate_inside_opening_guard_rejected` | span `window_start+0.5 .. window_start+3` is rejected (brief case 1) |
| `test_candidate_starting_past_the_guard_selected` | a candidate ~10 s past the start is still returned, byte-identical (brief case 2) |
| `test_only_guarded_candidate_returns_none` | all candidates guarded → `None`, no raise (brief case 3) |
| `test_span_overlapping_the_guard_rejected` | a span that straddles the guard boundary is rejected — overlap, not merely "starts inside" (brief case 4) |
| `test_guarded_candidate_skipped_for_a_later_survivor` | the guarded candidate is skipped and ranking order of the survivors is preserved |
| `test_guard_constant_is_five_seconds` | the constant exists and is exactly `5.0` |

## TDD cycle

**RED** — tests written first, run before any production change:

```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/python -m pytest tests/test_hook_first.py -q
FAILED tests/test_hook_first.py::TestOpeningDuplicateGuard::test_candidate_inside_opening_guard_rejected
FAILED tests/test_hook_first.py::TestOpeningDuplicateGuard::test_only_guarded_candidate_returns_none
FAILED tests/test_hook_first.py::TestOpeningDuplicateGuard::test_span_overlapping_the_guard_rejected
FAILED tests/test_hook_first.py::TestOpeningDuplicateGuard::test_guarded_candidate_skipped_for_a_later_survivor
FAILED tests/test_hook_first.py::TestOpeningDuplicateGuard::test_guard_constant_is_five_seconds
5 failed, 31 passed in 1.97s
```

Each failed for the right reason — the guarded candidate was being returned
(`'Opening sentence'` instead of `'Mid-window hook'`) and the constant did not
exist. `test_candidate_starting_past_the_guard_selected` passed already, by
design: it is the byte-identity guard for the candidates the change must NOT
affect.

**GREEN** — after the implementation:

```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/python -m pytest \
    tests/test_hook_first.py tests/test_hook_teaser.py \
    tests/test_boundary_snap.py tests/test_topic_boundary.py -q
........................................................................ [ 39%]
........................................................................ [ 79%]
.....................................                                    [100%]
181 passed in 2.30s
```

**Full suite**:

```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/python -m pytest tests/ -q \
    --continue-on-collection-errors
75 failed, 749 passed, 5 skipped, 22 warnings, 7 errors in 51.37s
```

Against the known baseline of **743 passed / 75 failed**: 749 passed is exactly
743 + the 6 new tests, and the 75 failures are unchanged. No new failures.

Note on the invocation: the brief's literal command
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q` aborts with
`Interrupted: 7 errors during collection` — 7 test modules fail to import for
environment reasons (`filter_transcript_by_timestamps` missing from `reelsfy`,
`audio_analyzer` / `stt_config` / `reels_clips_automator` not importable). I
confirmed these 7 collection errors are pre-existing by stashing my
`reelsfy.py` change and re-running: identical `7 errors during collection`, so
they are not mine. Adding `--continue-on-collection-errors` lets the remaining
modules run and reproduces the stated baseline. I also had to use
`./.venv/bin/python -m pytest` rather than the bare `pytest` on PATH: the PATH
interpreter is anaconda3 and has no `cv2`, so `reelsfy` cannot even be
imported under it.

## Existing tests rewritten

**None.** No existing test had to be moved or rewritten, because every existing
candidate in `test_hook_first.py` sits at 45-50 s, comfortably past the 25-30 s
guard in that file's 25-60 s window — including the ones the brief's item 5
anticipated (`test_first_passing_candidate_returned`,
`test_malformed_candidates_skipped_without_raising`,
`test_span_shorter_than_min_duration_skipped`, and the four
`TestMandatoryHookInjection` cases, which all use cue 3 = 45-50 s). No test was
deleted or weakened.

## Concerns / things I am unsure about

- **The guard is `span_start`-based, so brief case 4 is only partly expressible.**
  A candidate's span start is the earliest point of its span, so a span can
  never literally "start past the guard and reach back before it". The only
  genuinely distinct overlap case is a span that *straddles* the guard boundary
  (starts inside the guard, ends past it) — that is what
  `test_span_overlapping_the_guard_rejected` covers, with cue 2 at 28-33 s
  against a 25-30 s guard. A strictly-later-starting candidate can never reach
  back, so the guard is exact.
- **Boundary tolerance.** A candidate starting at exactly
  `window_start + 5.0` survives; one at `window_start + 4.999` does not. The
  `- 1e-9` slack is copied from the neighbouring comparisons, so a candidate
  starting at exactly `window_start + 4.9999999995` would survive. This is
  sub-microsecond and matches existing style; flagging it only for
  completeness.
- **Teaser-supply risk.** For topics whose whole body is shorter than ~5 s the
  guard can now return `None` more often, and the mandatory-hook block in the
  LLM prompt will be omitted more often. That is the intended degradation
  (no teaser, clip still publishes), and no new prints were added, so the
  "no hook" case is quieter than the "hook chosen" case.
- **Constant placement.** `HOOK_DUPLICATE_GUARD_SECONDS` sits at ~line 4775,
  *after* `select_hook_candidate` (~line 2632), as the brief specified. This is
  fine because the function body reads the module global at call time — the
  same pattern the existing `HOOK_TEASER_MIN_DURATION` reference already relies
  on — but it is worth knowing the guard is not defined above its use site.
