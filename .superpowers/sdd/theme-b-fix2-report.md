# Theme B — second fix wave: report

Branch `feat/improve-video`. Two findings closed, one unsigned commit:
`fix(selection): require caption repeat coverage, drop empty tokens`.

## Item 1 — coverage fraction on top of the 2-word floor

**File:** `reels-clips-automator/reelsfy.py`

- New constant beside the existing one (same block, not moved):
  `HOOK_CAPTION_MIN_MATCH_COVERAGE = 0.6` — named for what it means (how much
  of the shorter side the shared prefix must cover), not how it is computed.
- `teaser_repeats_first_caption` now returns:

  ```
  shared >= HOOK_CAPTION_MIN_MATCH_WORDS
  and shared / min(len(hook_words), len(caption_words)) >= HOOK_CAPTION_MIN_MATCH_COVERAGE
  ```

- Docstrings of the constant block and of `teaser_repeats_first_caption` were
  rewritten so the rule is stated in one place: the word floor is the absolute
  minimum, and the coverage requirement sits on top of it. Anchoring (opening's
  first word) and the direction bias are unchanged.

**Why 0.6 / min length.** A real repeat restates the opening, so the teaser is
usually the shorter side and the shared prefix covers nearly all of it. A
formulaic-bigram collision leaves a much longer remainder. `min(...)` keeps the
asymmetry: the case where one side is a long version of the other (the
"repeat the opening in pieces" case) still scores 1.0.

Notably this is *not* free containment — the prefix is still anchored at the
opening's first word, so `test_sentence_after_the_opening_is_not_a_repeat`
("Then it broke." vs a 6-word opening: shared 0) still passes, and a mid-window
sentence can never gain coverage it did not have at position 0.

## Item 2 — punctuation-only tokens

`_comparable_word_list` now drops tokens that normalize to the empty string
after per-word punctuation stripping. "…", "--" and friends no longer occupy
positions 0/1 and fake a two-word match. Docstring updated to state why.

## Tests

`tests/test_hook_teaser.py` (gitignored, force-added):

- `TestTeaserRepeatsFirstCaptionHelper::test_formulaic_opening_bigram_is_not_a_repeat`
  — a "So today …" teaser against a six-word "So today we ship the pipeline."
  opening is **not** a repeat; a genuine restatement still is.
- `TestTeaserRepeatsFirstCaptionHelper::test_punctuation_only_opening_does_not_fake_a_match`
  — "… --" vs a "…"/"--" opening, and a real teaser against the same opening,
  both return `False`.
- `TestPublishHookDedupGuard::test_teaser_sharing_only_a_formulaic_bigram_still_assembles`
  — end-to-end: the teaser is generated, `main_output_path` is passed to
  `generate_subtitle`, and `assemble_hook_teaser_final` is called once (i.e. the
  clip publishes *with* its teaser).

All three were confirmed failing before the implementation (3 failed, 75 passed)
and passing after.

### Commands and output

```
$ ./.venv/bin/python -m pytest tests/test_hook_teaser.py tests/test_hook_first.py -q
124 passed in 2.14s

$ ./.venv/bin/python -m pytest tests/ -q --continue-on-collection-errors
75 failed, 787 passed, 5 skipped, 22 warnings, 7 errors in 41.66s
```

Baseline on this branch before the change was 784 passed / 75 failed. The three
new tests account for the delta; the 75 failures and 7 collection errors are
unchanged and pre-existing (environmental; the errors are the separate import
ticket).

## Concerns / not sure

- The 0.6 threshold is a judgement call from the brief, not a measured value.
  It is exact-comparison-friendly: with the 2-word floor it admits ratios of
  1.0 and 0.67 and rejects 0.5, so there is no floating-point knife-edge at the
  boundary (a 3-word shorter side admits exactly 3/3 and 2/3; a 4-word side
  admits 4/4 and 3/4; 2/4 = 0.5 is rejected). Longer teaser texts are more
  forgiving, which matches intuition.
- `_comparable_word_list` is only used by the caption net, so the empty-token
  filter has no other caller to affect; worth re-checking if a second consumer
  is added later.
- No behavioural, prompt-wording or constant-identity assertion was converted,
  and no hook constant was env-gated or moved.
