# Batch Clip Selection (publish the 6 best, not all) — Plan

Changes what enters the publish queue. Today every edited clip is enqueued the
moment its edit finishes, so a 25-clip day becomes 25 posts. This plan makes the
pipeline rank the day's finished clips and enqueue only the top N (default 6).

Companion to `VIRAL_SELECTION_IMPROVEMENT_PLAN.md` (picks the single best clip
*inside* one capture) and `VIDEO_EDITING_IMPROVEMENT_PLAN.md` (renders it). This
plan ranks finished clips *across* captures.

## Current behaviour (verified in the code)

| Stage | Where | Note |
|---|---|---|
| Watch + edit | `monitor.py:129` `_queue_worker` | Single thread, strictly sequential; one `reelsfy` subprocess per file. Clips finish minutes apart, never all at once. |
| Collect | `monitor.py:317` `_organize_output_files` | Moves `final-*.mp4` + `content.txt` / `transcript.txt` / `description.txt` into `output/<name>/`. Runs only on success. |
| Enqueue | `monitor.py:360` `_publish_to_social_media` | **The only `enqueue()` call site.** Fires immediately, per clip. |
| Publish | `social_media_publisher/publish_queue.py:185` `PublishScheduler` | FIFO, `PUBLISH_INTERVAL_MINUTES` apart. |

So "batch" does not exist yet as a concept — the enqueue is per-clip and
unconditional.

## What signal already exists

`output/<name>/content.txt` is JSON written by reelsfy and already carries, per
segment: `title`, `description`, `tags`, `hook_start_time`, `hook_end_time`,
`hook_text`, `duration`. `output/<name>/audio_analysis_summary.txt` carries
`Energy`, `Speech Rate`, `Emotional Content`, `Onset Strength`, `Combined Score`.

Nothing is ever written back as a "virality" number — the ranking is new. But
the inputs for it are already on disk, so no new metadata has to be produced
during editing.

## Decisions (user-ratified)

1. **Batch trigger — quiet period after the edit queue drains.** Rank when the
   edit queue is empty *and* no edit has finished for `SELECTION_QUIET_MINUTES`
   (default 20). An optional batch cap allows earlier firing. Nothing about how
   the monitor is launched changes.
2. **Scoring — one LLM ranking call per batch.** A single call ranks the whole
   batch from `content.txt`. Cheap (one call per batch, not per clip) and it
   reuses the "Viral Segment Identifier" role reelsfy already uses. Any failure
   degrades to a deterministic fallback order; a provider outage can never
   block publishing.
3. **Losers — keep on disk, marked as not selected.** Rendered files stay in
   `output/<name>/`; the decision (rank, score, selected or not) is recorded so a
   mis-ranked clip can be published by hand later.

### Consequence of decision 1 worth stating plainly

With no cap, a clip edited at 09:00 and one edited at 23:00 are compared against
each other in the same ranking — that is the point. But a clip that *loses* a
ranking is decided and will not compete in a later batch of the same day. With
`SELECTION_BATCH_CAP=0` (the default, no cap) that only happens at the end of the
day, which is the intended semantics. Setting a cap makes ranking fire mid-day
and permanently discards that batch's runners-up.

## Plan (stories, in execution order)

### S1 — `social_media_publisher/clip_selection.py`: candidates + store

New module. `ClipCandidate` dataclass (`video_name`, `output_dir`,
`source_link`, `edited_at`, and lazily-read `title` / `hook_text` /
`hook_duration` / `tags` / `audio_score` / `metadata_error`).

`read_candidate(output_dir, video_name, source_link)` parses `content.txt` (JSON)
and `audio_analysis_summary.txt`. A clip whose `content.txt` is missing or
unparseable is still a candidate — it carries `metadata_error` and sorts to the
bottom — it is never silently dropped.

`SelectionStore` mirrors `PublishQueue`'s persistence exactly: same
temp-file-plus-`os.replace` atomic write, same "corrupt state file must not crash
the pipeline" tolerance. It holds `pending` (edited, not yet ranked) and
`decided` (`video_name -> {rank, score, selected, reason, decided_at}`).
Because pending candidates are persisted the moment an edit finishes, a crash or
restart does not lose the day's work.

### S2 — One-call LLM ranking, with a deterministic fallback

`rank_candidates(candidates, limit)`:

- Loads `reels-clips-automator/.env` via `dotenv_values` and builds an OpenAI
  client with `base_url=AI_ENDPOINT` — the same wiring as `reelsfy.py:127-131`.
  (The key is *not* in the root `.env` that `monitor.py` loads, so the module
  must read that file itself.)
- Asks for `response_format={"type": "json_object"}`, with the same single retry
  without that parameter on rejection, as `request_json_completion`
  (`reelsfy.py:2133`).
- Prompt lists each candidate by index: title, `hook_text`, hook span length,
  duration, tags, audio scores. Asks for every candidate back as
  `{"ranking": [{"video_name", "score": 0-100, "reason"}]}`.
- **Total order is guaranteed regardless of what comes back:** unknown names are
  dropped and names the model omitted are appended in fallback order. `rank_candidates`
  returns the FULL order — truncation is `select_and_enqueue`'s job, via
  `Ranking.selected` / `Ranking.rejected`, so losing clips can still have their
  decision recorded instead of being discarded by truncation.
- Fallback order (used when the call fails, times out, or returns nothing
  usable): audio `Combined Score` descending, then `edited_at` ascending.
- `rank_candidates` never raises and never returns fewer than `min(limit, len)`.

Model: `RANKING_MODEL`, else `OPENAI_MODEL`, else `gpt-5-mini`.

### S3 — `select_and_enqueue`: the only place enqueue is called

`select_and_enqueue(store, publish_queue, limit, ...)` ranks, then enqueues the
top `limit` **in rank order** (so the best clip publishes first — the scheduler
is FIFO), then records every candidate's decision. Clips are kept on disk; the
decision record is the only new artifact.

Ordering rule: **enqueue first, then mark decided.** If the process dies between
the two, the startup reconciliation drops the candidate on the next run because
its name is already in `publish_queue.json` — it cannot be published twice.

### S4 — `SelectionScheduler` thread in `clip_selection.py`

Same daemon-thread shape as `PublishScheduler` (`start` / `stop` /
`_stop_event.wait`), polling a few seconds. Fires only when **all** of:

- selection is enabled,
- there is at least one pending candidate,
- editing is not busy (edit queue empty and nothing in flight),
- either `now - last_edit_finished >= SELECTION_QUIET_MINUTES`, or
  `len(pending) >= SELECTION_BATCH_CAP` (when the cap is non-zero).

Injected as callables (`is_editing_busy`, `last_edit_finished_at`) so the
trigger is testable with a fake clock and no editor.

### S5 — `monitor.py` wiring

- `_organize_output_files` calls `selection_store.add_candidate(...)` instead of
  `_publish_to_social_media(...)`. `edited_total` still increments.
- Startup reconciliation: any candidate whose name is already `pending` /
  `publishing` / `published` in `publish_queue.json` is marked decided, so a
  restart never re-selects something already queued.
- `_publish_to_social_media` is kept and is the path used when selection is
  disabled, preserving today's behaviour exactly.
- Dashboard gains a candidates line (`candidates pending: N, last edit HH:MM:SS`)
  alongside the existing editing and publishing rows; each ranking prints a
  ranked list with `✅ selected` / `— not selected` marks, scores and reasons.
- CLI: `--select-limit`, `--select-quiet-minutes`, `--select-batch-cap`,
  `--no-selection`. Each mirrors the existing `--publish-delay` pattern (flag
  overrides env). `--publish-delay` keeps working — it now spaces the *selected*
  clips.

### S6 — Config, docs, dry-run

- `.env.example`: `SELECTION_ENABLED=true`, `SELECTION_LIMIT=6`,
  `SELECTION_QUIET_MINUTES=20`, `SELECTION_BATCH_CAP=0`, `RANKING_MODEL=`.
- `README.md` / `AGENTS.md`: document the new phase and the env knobs.
- `python3 -m social_media_publisher.clip_selection --dry-run output/<dir>...`:
  prints the ranking for existing output dirs and enqueues nothing. Lets the
  ranking be judged against the 50 clips already in `output/` before trusting it.

## Invariants

- **A failed, timed-out, or garbage LLM ranking never stops publishing.** The
  fallback order is always produced, and `rank_candidates` does not raise.
- **Fewer candidates than the limit publishes all of them.** The limit is a
  ceiling, never a reason to hold clips back.
- **Nothing is deleted.** Losing clips stay in `output/<name>/`.
- **A clip is enqueued at most once**, across restarts (startup reconciliation
  plus the enqueue-before-mark order).
- **`--no-selection` restores today's behaviour exactly** — same enqueue call,
  same timing.
- Selection config lives in `monitor.py` / `clip_selection.py`, not in
  `PostizConfig`; it is not a Postiz concern.

## Tests

New file `tests/test_clip_selection.py` (note: `tests/` is gitignored — force-add
with `git add -f`). Run with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_clip_selection.py tests/test_publish_queue.py tests/test_monitor.py -q`.

1. `read_candidate` parses `content.txt` + audio summary; missing/invalid
   `content.txt` still yields a candidate carrying `metadata_error`.
2. Valid LLM ranking → LLM order and scores returned.
3. LLM raising / returning garbage → fallback order (audio score desc), no raise.
4. LLM omitting a candidate → omitted ones appended, no duplicates.
5. `limit < len(candidates)` → exactly `limit`; `limit > len` → all.
6. `SelectionStore` round-trips `pending` and `decided` across instances; corrupt
   state file does not raise.
7. `select_and_enqueue` enqueues only the top N in rank order and records the
   rest as `selected: false`.
8. Re-running `select_and_enqueue` on the same batch enqueues nothing extra.
9. `SelectionScheduler`: does not fire while editing is busy; fires after the
   quiet period; fires early at the cap; no-op when nothing is pending.
10. Regression: `tests/test_publish_queue.py` and `tests/test_monitor.py` unchanged
    and green.

## Corrections made during execution

- **S2 as written had a bug:** truncating inside `rank_candidates` would discard
  the runners-up before their decision could be recorded. Implemented as above:
  the ranking is a full order, `select_and_enqueue` applies the limit.
- **`last_edit_finished` is scheduler-internal state** set by
  `on_edit_finished()`, not an injected callable — one source of truth.
- Two findings from the first real dry-run over `output/`:
  the audio `Combined Score` is a constant `1.00` across all clips, so the
  fallback order is effectively FIFO (stable, but not a quality signal — the LLM
  ranking is where the value is); and `openai.OpenAI(...)` only constructs under
  the `reels-clips-automator` conda env (openai 1.99.9), not anaconda base — so a
  degraded ranking is now logged loudly rather than silently (`Ranking.error` is
  surfaced by `select_and_enqueue`).

## Out of scope

- Ranking *within* a capture — that is `VIRAL_SELECTION_IMPROVEMENT_PLAN.md`.
- Feeding real platform metrics back into the ranking (a later story, once
  Postiz reports are available).
- Parallelising the editing worker (still strictly sequential).