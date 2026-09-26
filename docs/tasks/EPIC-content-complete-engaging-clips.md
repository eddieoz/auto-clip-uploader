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

### Story A2 — Document the OBS replay-buffer setup (1 pt)

R5 is resolved operationally (buffer raised to 90 s), but the setup must be written down so future changes don't silently re-break it.

**TDD:** none (documentation story) — verify by doc presence.

**BDD:**
```gherkin
Scenario: Operator guidance exists
  Given a user reads the README video-processing section
  Then they find the recommended replay-buffer length (90 s) and why
  And an explanation that capture length bounds topic completeness
```

**Files:** `README.md`, `AGENTS.md` (one bullet). Optional: log a warning in reelsfy when capture duration < min_duration + 10 s ("capture shorter than the minimum clip window — check the OBS replay buffer").

### Story A3 — Boundary LLM pass reports topic completeness (1 pt)

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

### Story A4 — Apply start extension to the resolved window (1 pt)

`trim_window_to_topic_start` (or a sibling) may **move the window start earlier** to `earlier_start_cue` when the parsed result says the topic is incomplete — clamped to cue 0 and to the 59 s max reach (the resulting window duration must never exceed `max_duration`; within the 90 s capture the END stays anchored, so extending the start is bounded by the cap).

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

### Story A5 — Completeness observability (1 pt)

After selection, log one structured line: capture span, resolved window, chosen duration, `topic_complete`, and whether the first caption sentence is grammatically complete. This is the feedback signal for tuning A1-A4.

**TDD (write first):** unit test on the log builder (pure function returning the dict/string).

**BDD:**
```gherkin
Scenario: Selection reports its own completeness
  Given a segment was selected
  When the selection summary is emitted
  Then it contains capture_span, resolved_window, duration and topic_complete
```

**Files:** `reelsfy.py`, new small test file.

---

## Theme B — Non-duplicative, mid-video hook (R6)

### Story B1 — Anti-duplication rule in hook selection (1 pt)

`select_hook_candidate` rejects candidates whose cue span **overlaps the first ~5 s of the resolved window** — returning such a hook is what duplicates the opening. If no candidate survives, return None (teaser is skipped; degradation invariant).

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

### Story B2 — Prefer mid-window hook candidates (1 pt)

Among valid candidates, prefer ones in the **middle band** of the window (not within the first 20% nor the last 10%) — an engaging moment mid-clip makes the teaser complementary instead of repetitive. Pure preference, same None-degradation.

**TDD (write first):** ordering tests — two valid candidates, the mid-band one wins even if listed second; single edge candidate still selected; empty list → None.

**BDD:**
```gherkin
Scenario: Mid-band candidate outranks an early one
  Given valid candidates at 10% and 45% of the window
  When select_hook_candidate runs
  Then the 45% candidate is returned
```

**Depends on:** B1 (so "valid" already excludes opening-duplicates). **Files:** `reelsfy.py`, `tests/test_hook_first.py`.

### Story B3 — Boundary prompt steers away from the first sentence (1 pt)

The `resolve_topic_start_cue` prompt instructs: hook candidates must NOT be the first sentence of the final topic; prefer an engaging sentence mid-topic. Soft steering (validation still clamps deterministically).

**TDD (write first):** prompt-content test in `tests/test_topic_boundary.py` (instruction present; legacy mode prompt unchanged).

**BDD:**
```gherkin
Scenario: LLM is told where hooks may live
  Given the boundary pass prompt is built
  Then it contains the "not the first sentence / prefer mid-topic" instruction
  And TOPIC_BOUNDARY_LLM=false produces a prompt without the block
```

**Files:** `reelsfy.py` (~2319), `tests/test_topic_boundary.py`.

### Story B4 — Publish-level dedup guard (1 pt)

Defense in depth: before assembling, if the teaser time range overlaps the main clip's first 5 s (or `hook_text` equals the first caption sentence), skip the teaser and publish the main clip only. Catches cases the selection layer missed.

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

## Story index (13 pts total; C1 done)

| Story | Theme | Pt | Depends on |
|-------|-------|----|------------|
| A1 unify duration bounds | A | 1 | — |
| A2 OBS buffer doc + warn | A | 1 | — |
| A3 completeness in boundary pass | A | 1 | — |
| A4 apply start extension | A | 1 | A3 |
| A5 completeness observability | A | 1 | A4 |
| B1 anti-duplication rule | B | 1 | — |
| B2 prefer mid-window hooks | B | 1 | B1 |
| B3 prompt steering | B | 1 | — |
| B4 publish-level dedup guard | B | 1 | — |
| C1 disable end card ✅ | C | 1 | done (`4728e95`) |
| C2 diagnose Instagram freeze | C | 1 | — |
| C3 fix assembly | C | 1 | C2 |
| C4 re-enable decision | C | 1 | C3, user |
| D1 regression + docs | D | 1 | A1-A5, B1-B4, C3 |

Parallelizable now: A1, A2, A3, B1, B3, B4, C2.
