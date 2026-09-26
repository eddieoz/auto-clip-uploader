# EPIC: Content-Complete, Engaging Clips

**Status:** Ready for implementation
**Branch:** `feat/improve-video`
**Workflow:** Subagent-driven development (brief → implement → review per story), one commit per story, tests gitignored → `git add -f`, run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/`.

## Background (user reports, 2026-09-26)

1. **Videos are too short** — clips miss the complete context of the topic.
2. **The initial hook is most of the time just the first phrase duplicated** — the cold-open teaser shows the same sentence that opens the main clip. With longer videos the engaging moment would sit in the middle of the clip instead.
3. **The end card is not showing** — on Instagram the video freezes before the final static scene. *(Immediate mitigation shipped: end card disabled by default, commit `4728e95`.)*

## Root-cause findings (verified in code)

| # | Finding | Evidence |
|---|---------|----------|
| R1 | `monitor.py` invokes reelsfy **without** `--min-duration/--max-duration`, so the CLI defaults `min=1.0 / max=120.0` flow into `generate_viral`. The LLM may legally return a 5-second clip. | `monitor.py:224`, `reelsfy.py:3411-3421` |
| R2 | The LIVE prompt **hard-codes "maximum of 59 seconds"**, ignoring the passed `max_duration`. The 59 s value itself is the ratified product cap (see decisions), but it must come from the parameter, not a literal. | `reelsfy.py:3049,3051` |
| R3 | `generate_viral`'s own signature defaults (35/59) disagree with the CLI defaults (1.0/120.0). | `reelsfy.py:2719-2720` |
| R4 | The boundary LLM pass only ever **trims** the window start (previous-topic tail). It can never *extend* the start earlier to recover topic context that the heuristic window cut off. | `trim_window_to_topic_start`, `resolve_topic_start_cue` |
| R5 | ~~The OBS replay buffer is now 60 s — the source itself bounds how much topic context exists.~~ **RESOLVED 2026-09-26:** user raised the OBS clip duration to 90 s. | user report |
| R6 | `select_hook_candidate` picks the **first** candidate inside the window and the prompt welds it as MANDATORY with "EXACTLY these values". Because the window starts at the topic start, the first sentence is usually the "winner" → teaser duplicates the clip opening. | `reelsfy.py:3007-3036`, `select_hook_candidate` |
| R7 | The end-card assembly stream-copies three **differently encoded** pieces (teaser re-encode + main burn + lavfi/NVENC end card) with `-c copy`; local players tolerate the discontinuity, Instagram's transcoder freezes at the last main frame and never renders the end card. | `assemble_hook_teaser_final`, `build_teaser_concat_command`, `build_end_card_command` |

## Ratified decisions (user, 2026-09-26)

1. **OBS replay buffer = 90 s** (raised from 60 s by the user). The capture now holds enough topic context.
2. **Maximum clip duration is a hard 59 s** — a product requirement driven by the YouTube video limit. Do NOT raise it; the earlier idea of allowing 120 s is dropped.
3. Therefore the "too short" problem is **not** the cap: it is the effective 1 s minimum reaching the LLM via the CLI defaults (R1) and the LLM choosing a sub-window of the topic. The fix is to unify all bounds at min 35 s / max 59 s and let the boundary pass extend the start earlier within the 90 s capture (A3/A4), clamped so the clip never exceeds 59 s.

## Ratified invariants (must hold in every story)

- **Recency anchor:** the topic CONCLUDES near the end of the capture (OBS replay-buffer model). The clip END stays anchored to the capture end; only the START may move (trim or extend).
- **Exactly one clip** per capture.
- **Degradation:** every LLM/audio failure falls back to the current heuristic behavior — never fails the run.
- **Mastering once:** the audio mastering chain is applied exactly once, in the final burn/assembly.
- **Publish-final-only:** only the complete video (audio + subtitles + overlays) is published; every assembly failure degrades to direct publish.

---

## Theme A — Complete-context clip duration (R1-R5)

### Story A1 — Unify duration bounds across CLI, signature, and prompt (1 pt)

All three layers must agree on the ratified bounds (**min 35 s, max 59 s** — see decisions: 59 s is the YouTube-driven hard cap, not to be raised), and the LLM-facing prompt must derive its bounds from the actual `min_duration`/`max_duration` parameters instead of the hard-coded "59 seconds" literal.

**TDD (write first):**
- `tests/test_duration_bounds.py`: the reelsfy source contains no `"maximum of 59 seconds"` literal; the prompt template interpolates `{max_duration}`; `generate_viral` signature defaults are 35.0/59.0; the CLI `--min-duration`/`--max-duration` defaults are 35.0/59.0 (what monitor.py's no-flag invocation lands on).

**BDD:**
```gherkin
Scenario: Prompt reflects configured bounds
  Given generate_viral is called with min_duration=35 and max_duration=59
  When the LIVE prompt is built
  Then the prompt says "minimum of 35 seconds and maximum of 59 seconds"
  And the literal "maximum of 59 seconds" appears nowhere in the source

Scenario: Pipeline defaults are clip-sane and respect the YouTube cap
  Given reelsfy is run the way monitor.py runs it (no duration flags)
  Then the effective minimum duration is 35 seconds
  And the effective maximum duration is exactly 59 seconds
```

**Files:** `reelsfy.py` (prompt f-string ~3049-3051, CLI defaults ~3411-3421), `tests/test_duration_bounds.py`.

### Story A2 — Document the OBS replay-buffer setup ✅ DONE

Shipped as commit `18c9600`: a README "OBS Replay Buffer (capture length)" section, an AGENTS.md bullet, and a capture-length warning in `generate_viral`. R5 is resolved operationally (buffer raised to 90 s), but the setup must be written down so future changes don't silently re-break it.

**TDD:** none (documentation story) — verify by doc presence.

**BDD:**
```gherkin
Scenario: Operator guidance exists
  Given a user reads the README video-processing section
  Then they find the recommended replay-buffer length (90 s) and why
  And an explanation that capture length bounds topic completeness
```

**Files:** `README.md`, `AGENTS.md` (one bullet). Optional: log a warning in reelsfy when capture duration < min_duration + 10 s ("capture shorter than the minimum clip window — check the OBS replay buffer").

### Story A3 — Boundary LLM pass reports topic completeness ✅ DONE

Extend `resolve_topic_start_cue`'s JSON reply contract with `topic_complete` (bool) and, when incomplete, `earlier_start_cue` (the cue where the final topic actually begins, earlier than the heuristic cue). Parsing failure degrades to today's `(cue_index, hook_candidates)`.

**TDD (write first):**
- `tests/test_topic_boundary.py` additions: reply containing `topic_complete:false, earlier_start_cue:N` parses to an extended result; malformed/missing fields parse to the legacy shape; LLM error returns legacy shape (degradation invariant).

**BDD:**
```gherkin
Scenario: LLM says the topic starts earlier than the heuristic cue
  Given the boundary pass returns topic_complete=false and earlier_start_cue=12
  When the reply is parsed
  Then the result carries earlier_start_cue=12 for the caller to consider
  And a parse failure yields exactly today's legacy result
```

**Files:** `reelsfy.py` (`resolve_topic_start_cue` prompt ~2319, `_parse_boundary_reply`), `tests/test_topic_boundary.py`.

### Story A4 — Apply start extension to the resolved window ✅ DONE

Shipped as commit `13aa524`. `extend_window_to_earlier_topic_start` is a sibling of `trim_window_to_topic_start` (the two directions of start movement stay separately named and testable). It **moves the window start earlier** to `earlier_start_cue` when the parsed result says the topic is incomplete, clamped to the capture start (`max(0.0, …)`) and to the 59 s max reach — because the END stays anchored, an extension that would breach the cap is clamped to `end − max_duration`. Supporting segments are returned **unchanged**: the extension only grows the window, so it must never drop any.

Two invariants were tightened during review:
- `min_start_srt` is a **floor** for `snap_window_to_boundaries` / `refine_window_start_with_pause`, so it is taken from whichever cue owns the window start after the extension — otherwise the snapper silently undoes the extension.
- `refine_window_start_with_pause` picks the nearest pause in ±1.5 s with no lower bound and could push the start *earlier*, breaching the 59 s cap for a window sitting exactly at it. It now takes a `max_duration` and rejects candidates below `end − max_duration` (default `None`, inert for other callers).

Byte-identity is pinned by a golden resolved-window prompt section genuinely captured from a pre-A4 worktree, not hand-written to match current output.

**TDD (write first):**
- `tests/test_boundary_snap.py` additions: extension moves start earlier and recomputes duration; never extends beyond capture start; never changes end; never produces a window longer than `max_duration` (59 s) — extension is clamped by the cap; when `topic_complete=true` or no earlier cue, window is byte-identical to today.

**BDD:**
```gherkin
Scenario: Incomplete topic gets more context
  Given the heuristic window starts at cue 20 and the LLM reports earlier_start_cue=12
  When the resolved window is built
  Then the window starts at cue 12 and still ends at the capture end
  And the window duration stays within the 59 s cap

Scenario: Complete topic is untouched
  Given topic_complete=true
  When the resolved window is built
  Then the window equals the current trim-only behavior
```

**Depends on:** A3. **Files:** `reelsfy.py`, `tests/test_boundary_snap.py`.

### Story A5 — Completeness observability ✅ DONE (1 pt)

Shipped as commit `e79045f`. After selection, `build_selection_summary(...)` — a **pure** function returning a dict — is printed as one strictly-parseable `SELECTION_SUMMARY {…}` line, so the run log can be scraped to tune A1-A4. It changes no behavior. An empty selection still emits one line, so every run produces at least one. Note the line count is **per validated segment**, not per run: an LLM reply with N segments emits N lines sharing `capture_span`, `resolved_window` and `topic_complete`, differing only in `chosen_duration` — a scraper keying by run must not assume one line.

**TDD (write first):** unit test on the log builder (pure function returning the dict), plus wiring tests through the existing `_run_generate_viral` harness.

**BDD:**
```gherkin
Scenario: Selection reports its own completeness
  Given a segment was selected
  When the selection summary is emitted
  Then it contains capture_span, resolved_window, duration and topic_complete
```

**Files:** `reelsfy.py`, `tests/test_selection_summary.py`.

Details worth knowing before tuning against this signal:
- Cues inside the window are selected by **overlap**, not containment — `snap_window_to_boundaries` can land the start mid-cue, and the cue owning the start begins before the window.
- The first-sentence rule is the simple heuristic, and it is **biased toward `complete`**: decimals (`3.14`), `e.g.` and ellipses read as sentence ends. The bias is documented in-code and characterized by tests, because this metric is what A1–A4 get tuned against.
- Non-finite values (`nan`/`inf`, reachable via `_srt_like_to_seconds("nan")`) degrade to `null` so the marker line is always strictly parseable — Python's lenient `json.loads` accepts `NaN`, so a `jq`/`JSON.parse` consumer would otherwise have been the only one to notice.
- `capture_span` and `topic_complete` were only bound inside conditional blocks; both are now seeded to `None` so the fallback path cannot `NameError`.

---

## Theme B — Non-duplicative, mid-video hook (R6)

### Story B1 — Anti-duplication rule in hook selection ✅ DONE

Shipped as commit `6778f3e`. `HOOK_DUPLICATE_GUARD_SECONDS = 5.0` — a candidate whose cue span starts before `window_start + 5s` is rejected; no survivor → `None` (teaser skipped, degradation intact). A cue span is contiguous and starts at `span_start`, so start-based rejection is exactly overlap-based rejection; a candidate starting exactly at `window_start + 5.0` survives.

**TDD (write first):**
- `tests/test_hook_first.py` additions: candidate at window start+1 s is rejected; candidate at window start+10 s is selected; no surviving candidate → None; existing ordering tests updated.

**BDD:**
```gherkin
Scenario: Hook candidate sits on the clip opening
  Given a hook candidate spanning window_start+0.5s..window_start+3s
  When select_hook_candidate runs
  Then it returns None (the teaser would duplicate the opening)

Scenario: Mid-window candidate wins
  Given candidates at +2s (rejected) and +18s (valid)
  When select_hook_candidate runs
  Then the +18s candidate is returned
```

**Files:** `reelsfy.py` (`select_hook_candidate` ~2520), `tests/test_hook_first.py`.

### Story B2 — Prefer mid-window hook candidates ✅ DONE

Shipped as commit `bd41559`. Bands `HOOK_MID_BAND_START_FRACTION = 0.20` / `HOOK_MID_BAND_END_FRACTION = 0.90` (inclusive, epsilon-widened to match the neighbouring comparisons). The loop body and check order are unchanged: the first survivor in LLM order is kept as a fallback, and the first **mid-band** survivor is returned when one exists — so the behavior is byte-identical whenever no mid-band candidate exists. The 90% upper bound is near-inert given the containment and 1.5-6 s duration caps; it is kept because the brief specifies it and it costs nothing.

**TDD (write first):** ordering tests — two valid candidates, the mid-band one wins even if listed second; single edge candidate still selected; empty list → None.

**BDD:**
```gherkin
Scenario: Mid-band candidate outranks an early one
  Given valid candidates at 10% and 45% of the window
  When select_hook_candidate runs
  Then the 45% candidate is returned
```

**Depends on:** B1 (so "valid" already excludes opening-duplicates). **Files:** `reelsfy.py`, `tests/test_hook_first.py`.

### Story B3 — Boundary prompt steers away from the first sentence ✅ DONE

Shipped as commit `06c3c7c`. One clause appended to the existing `hook_candidates` bullet in the `resolve_topic_start_cue` prompt: candidates must not be the final topic's first sentence, prefer an engaging mid-topic sentence. Prose only — the JSON reply contract, `_parse_boundary_reply` and every failure path are untouched, and gate-off stays byte-identical (the early return precedes prompt construction). The enforcing layer remains B1/B2.

**TDD (write first):** prompt-content test in `tests/test_topic_boundary.py` (instruction present; legacy mode prompt unchanged).

**BDD:**
```gherkin
Scenario: LLM is told where hooks may live
  Given the boundary pass prompt is built
  Then it contains the "not the first sentence / prefer mid-topic" instruction
  And TOPIC_BOUNDARY_LLM=false produces a prompt without the block
```

**Files:** `reelsfy.py` (~2319), `tests/test_topic_boundary.py`.

### Story B4 — Publish-level dedup guard ✅ DONE

Shipped as commit `9305f3f`, hardened by `1e7e81c` and `0d79d55`. `publish_segment_with_optional_teaser` gains a `clip_start_time=None` keyword, populated at the single production call site with `segment["start_time"]` and converted through `_srt_like_to_seconds` (segment dicts are dual-shape: `start_time_srt` and/or `start_time`). Two nets, either of which skips the teaser to the existing direct-publish path:

- **Time overlap** — `teaser_duplicates_clip_opening`, reusing `HOOK_DUPLICATE_GUARD_SECONDS`. Inert when `clip_start_time` is `None` (legacy callers only; production always passes it). This is the **stronger** of the two when the clip start was clamped later than the resolved window start, which is exactly the residual B1 misses.
- **Caption text** — `teaser_repeats_first_caption`, running regardless. The original terminator-delimited exact-sentence form was found to be **largely inert in production**: `_first_sentence_ends_complete`'s documented bias runs toward "complete" on decimals and abbreviations, which screencast content hits constantly, so the extracted "sentence" was usually a fragment. It now compares the caption words inside the first 5 s of the clip (time-windowed, terminator-free), requiring a common leading prefix of ≥ 2 words (`HOOK_CAPTION_MIN_MATCH_WORDS`) covering ≥ 60% of the shorter sequence (`HOOK_CAPTION_MIN_MATCH_COVERAGE`) — the coverage half is what stops a formulaic bigram ("so today", "in this video") from dropping a good teaser. Anchored at the opening's first word, so free containment is not used and a sentence appearing later in the clip is still not a repeat. Punctuation-only tokens are filtered.

Invariants: skipping leaves `teaser_path`/`main_output_path` as `None`, so `generate_hook_teaser` is never called and **no second burn** is added (mastering-once holds); the end card is built before the hook block, so the end-card-on + skip case falls into the pre-existing branch.

**TDD (write first):**
- `tests/test_hook_teaser.py` additions: overlapping hook → `generate_hook_teaser` not called, direct publish; non-overlapping hook → current assembly.

**BDD:**
```gherkin
Scenario: Teaser would repeat the opening
  Given hook_start=window_start and hook_end=window_start+3s
  When publish_segment_with_optional_teaser runs
  Then no teaser is generated and the main clip is published alone
```

**Depends on:** nothing (can run before B1/B2). **Files:** `reelsfy.py` (`publish_segment_with_optional_teaser`), `tests/test_hook_teaser.py`.

---

## Theme C — End card (R7)

### Story C1 — Disable end card by default ✅ DONE

Shipped as commit `4728e95`: `resolve_end_card_enabled` now defaults to `false` (`END_CARD_ENABLED=true` opts in), `.env.example` and tests updated. Keep this story in the EPIC for traceability.

### Story C2 — Diagnose the Instagram freeze (1 pt)

Reproduce the assembly locally and identify the exact incompatibility. Working hypothesis (R7): `-c copy` concat across three independently encoded streams (different SPS/PPS, timebase, or edit lists) — desktop players bridge the discontinuity, Instagram's transcoder stalls on the last main frames and drops the end card.

**TDD:** characterization test capturing current piece parameters via `ffprobe` (codec/profile/timebase per piece) as a fixture doc in this file — the fix story asserts against it.

**BDD:**
```gherkin
Scenario: Root cause is identified and written down
  Given the three assembly pieces from a real render
  When they are probed and the concat inspected
  Then the EPIC carries the concrete mismatch (stream parameters, timestamps)
  And the reproduction command is recorded
```

**Files:** `docs/tasks/EPIC-content-complete-engaging-clips.md` (findings appendix), throwaway probe script.

### Story C3 — Fix the assembly so the end card survives Instagram (1 pt)

Based on C2's finding, rework `assemble_hook_teaser_final` so the assembled file is a single continuous stream: either (a) one `filter_complex` concat + NVENC re-encode pass, or (b) end card/teaser generated with byte-compatible encoder settings and proper timebase. Audio is never re-mastered (invariant). Verify with `ffprobe`: final duration ≈ sum of pieces, single video parameter set across boundaries.

**TDD (write first):**
- `tests/test_polish.py` / `tests/test_hook_teaser.py` additions: the assembly command for the enabled path is the fixed strategy (e.g., single re-encode concat); ladder fallbacks preserved; mastering invariant asserts still pass.

**BDD:**
```gherkin
Scenario: Assembled file is boundary-safe
  Given teaser, main and end card pieces
  When the assembly runs with END_CARD_ENABLED=true
  Then the published file has one continuous video stream
  And its duration is within 0.5 s of the sum of the pieces
  And the audio mastering filter was applied exactly once overall

Scenario: Fixed assembly still degrades safely
  Given the assembly command fails
  When the ladder runs out
  Then the main clip is published directly (publish-final-only invariant)
```

**Depends on:** C2. **Files:** `reelsfy.py`, `tests/test_polish.py`, `tests/test_hook_teaser.py`.

### Story C4 — Re-enable decision (user-driven, 1 pt)

After C3, the user A/B tests performance with and without the end card. If the fixed assembly survives Instagram and metrics don't regress, flip the default back to `true` (one-line change + test/doc updates). No code beyond the default flip.

**BDD:**
```gherkin
Scenario: Decision is recorded
  Given the user's A/B results
  When the decision is made
  Then the EPIC records the outcome and the default is flipped or left off
```

---

## Epic completion

### Story D1 — Regression sweep + docs refresh (1 pt)

Run the full `tests/` sweep (only pre-existing environmental failures allowed), re-assert the ratified invariants above still have binding tests, and update `README.md`, `AGENTS.md`, `reels-clips-automator/.env.example` for everything A-C changed. Update this EPIC's status per story.

**BDD:**
```gherkin
Scenario: Epic closes green
  Given all stories A1-C3 are merged
  When the full regression suite runs
  Then no new failures appear and the docs describe the shipped behavior
```

---

## Story index (13 pts total; A, B and C1 done)

| Story | Theme | Pt | Depends on |
|-------|-------|----|------------|
| A1 unify duration bounds ✅ | A | 1 | done (`5146779`) |
| A2 OBS buffer doc + warn ✅ | A | 1 | done (`18c9600`) |
| A3 completeness in boundary pass ✅ | A | 1 | done (`e3acda0`) |
| A4 apply start extension ✅ | A | 1 | done (`13aa524`) |
| A5 completeness observability ✅ | A | 1 | done (`e79045f`) |
| B1 anti-duplication rule ✅ | B | 1 | done (`6778f3e`) |
| B2 prefer mid-window hooks ✅ | B | 1 | B1 — done (`bd41559`) |
| B3 prompt steering ✅ | B | 1 | done (`06c3c7c`) |
| B4 publish-level dedup guard ✅ | B | 1 | done (`9305f3f`, +`1e7e81c`, `0d79d55`) |
| C1 disable end card ✅ | C | 1 | done (`4728e95`) |
| C2 diagnose Instagram freeze | C | 1 | — |
| C3 fix assembly | C | 1 | C2 |
| C4 re-enable decision | C | 1 | C3, user |
| D1 regression + docs | D | 1 | A1-A5, B1-B4, C3 |

**Theme A is complete.** Every invariant in the ratified list now has a binding test: the recency anchor (the extension never assigns an end time), the 59 s cap (clamped in the extension and re-checked after prosodic refinement), degradation (every LLM/audio failure path returns the analyzed window untouched), and the 35–59 s bounds from A1.

**Theme B is complete.** The duplication hole R6 described is closed at three layers — deterministic
rejection in the selection layer (B1), a mid-window preference so the teaser lands where it is
complementary (B2), model steering so it aims there in the first place (B3), and a publish-time
guard that catches whatever survives (B4). A final whole-theme review verified all six ratified
invariants hold across the four stories, and two follow-up fix waves hardened B4's caption net
after review showed it would rarely fire on real screencast content.

Notes for whoever tunes this next:
- The 5 s guard is defined **once** (`HOOK_DUPLICATE_GUARD_SECONDS`) and shared by the selection
  and publish layers. There is no second 5.0 literal, and the B3 prompt deliberately speaks in
  sentences rather than seconds so it cannot drift.
- The two nets answer different questions. The **time** net is the strong one: it compares
  against the clip's *actual* start, which `validate_segment_against_window` may have clamped
  later than the resolved window start — the case B1 alone misses. The **caption** net is the
  fallback for when timings shifted but the words did not; it is deliberately conservative and
  will sometimes miss rather than drop a good teaser.
- Coordinate spaces differ by design and each comparison stays inside one of them: the selection
  layer and the time net work in **source-video seconds**, while the caption net works in
  **clip-relative** seconds (the caption words are already clip-relative). Do not compare across
  them.
- The caption net's thresholds (`2` words, `60%` coverage) are unmeasured against real captions —
  there is no corpus. They were chosen so the cheap direction is "don't drop a good teaser".
  They are the first knobs to revisit if teasers feel too eager or too shy.

Parallelizable now: C2.

### Deferred follow-ups (Theme B, non-blocking)

- `HOOK_DUPLICATE_GUARD_SECONDS` and the three hook-selection constants are the only editing
  knobs in this file that are **not** env-gated, unlike the rest of `reels-clips-automator/.env.example`.
- `opening_caption_prefix` assumes the caption word list is sorted by start time; that is
  undocumented and unverified against the jump-cut remapper.
- Several tests pin constant values and the B3 prompt's literal English wording. They are not
  vacuous — they are the ones that catch a revert — but they will fail on a harmless retune.
- Separate ticket: 7 test modules fail to import (`filter_transcript_by_timestamps` and friends),
  so the full suite needs `--continue-on-collection-errors` to reproduce its baseline.
