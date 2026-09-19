# Viral-Part Selection Improvement Plan

Improves how the pipeline finds and carves the engaging topic from a replay-buffer
capture. Companion to `VIDEO_EDITING_IMPROVEMENT_PLAN.md` (rendering) — this plan
covers selection only.

## Input model (user-ratified constraints)

During a livestream, a replay click is pressed as soon as a topic concludes. OBS
saves a rolling buffer (last 90 s, now 60 s). Therefore:

1. **The topic concludes near the end of the capture — by design.** The recency
   bias in `TopicAnalyzer.identify_concluding_topic` is correct and MUST be kept.
   It is not a heuristic guess about the video; it encodes the capture workflow.
2. **One clip per capture.** No candidate competition, no multi-clip harvesting.
   Improvements go into the quality of that single clip.
3. **Truncation (topic longer than the buffer) is rare.** Keep handling simple:
   snap the in-point to the nearest clean sentence/pause; no elaborate recovery.
4. The buffer's *start* typically contains the tail of the previous topic. The
   improved pipeline trims the clip to the final topic's start.

A useful consequence of the input model: the whole transcript is ≤ 90 s, so the
LLM can cheaply see EVERYTHING in the capture. The current architecture hides it
behind cue-level heuristics; this plan flips that while keeping the recency anchor.

## Gap analysis

| # | Gap | Where |
|---|-----|-------|
| G1 | Phrase matching is substring-based ("so" fires inside "person"), and topic boundaries are decided per-SRT-cue by bag-of-words Jaccard — weakest exactly where it matters: where the final topic starts. | `topic_analyzer.py:77-95`, `_measure_semantic_connection` |
| G2 | The LLM never sees the full capture in the primary path (only the concluding sentence + three 80-char snippets), so it cannot verify completeness or self-containedness. | `create_content_centric_prompt` |
| G3 | The LLM returns the final timestamps unvalidated — it may drift from the scored window, cut mid-sentence, or leave dangling references. | `generate_segments` consumers |
| G4 | Audio-derived natural boundaries are computed then discarded (`# TODO`); audio_context is built but unused by the content-centric prompt (wasted librosa pass). | `reelsfy.py:2322`, `:2370` |
| G5 | Hook is chosen after the segment is fixed, from inside it; on short-form the first 2 s decide retention, so the hook should be scored first within the topic. | prompt-side only |
| G6 | JSON-repair hacks (`'` → `"`) can corrupt real apostrophes; no structured output. | `generate_metadata`, `generate_viral` |
| G7 | No publish-time validation (duration, sentence-complete ends, hook-inside-segment is validated, completeness is not) and no feedback loop from platform metrics. | pipeline tail |

## Plan (stories, in execution order)

### S1 — Word-boundary phrase matching + robust SRT cue parsing (G1a)
Fix `analyze_topic_transitions`/`_identify_narrative_role` to match transition,
conclusion, coherence, and role phrases on word boundaries (`\b` regex), not
substrings. Pin with unit tests for the false-positive words found in review.

### S2 — LLM topic-boundary pass: trim previous-topic tail (G1b, G2)
New prompt pass over the FULL capture transcript (cheap — it is ≤ 90 s):
"the final topic concludes at the end; identify the cue index where the final
topic begins; list any previous-topic cues at the start." Structured output
(json schema). The heuristic `identify_concluding_topic` stays as the
deterministic anchor (recency + confidence) and the tie-break fallback when the
LLM pass fails or is unavailable — failure must degrade to today's behavior.
Output: `topic_start_index` (and per-cue attribution) consumed when building the
optimized segment's window.

### S3 — Boundary snapping with word timestamps and prosody (G3, G4)
Snap the segment start to the first sentence start at/after `topic_start_index`
and the end to the last sentence end, using the word timestamps already carried
through the pipeline; additionally use `EnhancedAudioAnalyzer.detect_natural_boundaries`
(pause detection) to avoid opening on a half-finished word. Remove the discarded
`# TODO` path: the boundary helper becomes the single implementation, and the
unused audio-analysis pass in the content-centric path is dropped (compute saved).

### S4 — LLM completeness verification + structured outputs (G2, G3, G6)
Second LLM pass (or the same call, two-part response): given the snapped window,
verify the concluding topic is complete and self-contained; allowed to nudge the
start later (never past the first sentence of the topic) if leading sentences are
throat-clearing. Replace `json.loads` repair chains with `response_format`
JSON-schema structured output in both `generate_metadata` and `generate_viral`.
Validation before returning: duration in range, sentence-complete ends, hook
fields inside the segment (existing `parse_hook_fields` reused).

### S5 — Hook-first within the concluding topic (G5)
Score hook sentences across the final topic window (LLM-proposed, validated
1.5–6 s as today) BEFORE the caption/description generation call, so the hook
text and the cold-open teaser come from the same chosen sentence. Failure keeps
the current "hook optional" behavior.

### S6 — Publish-time validation + metrics feedback (G7, optional later)
Deterministic pre-publish checklist (already partly enforced): log + skip-hook
degradation on violation. Optionally, later: correlate `audio_analysis_summary.txt`
features with dashboard view/retention data to weight future scoring. Out of
scope until the dashboard exposes per-clip metrics.

## Invariants for every story

- Recency anchor and concluding-topic architecture preserved (constraint 1).
- Exactly one published clip per capture (constraint 2).
- Byte-identical OFF-path: any env gate (e.g. `TOPIC_BOUNDARY_LLM`, default off
  or on per approval) must degrade to today's behavior when unavailable.
- Output naming and publish-queue contract unchanged.
- TDD, commit per story, tests force-added under `tests/`.
