# Theme B fix wave — report

Branch `feat/improve-video`, one unsigned commit. Report file left uncommitted.

## Item 1 (Important) — caption net made time-based

`reels-clips-automator/reelsfy.py`

- Renamed `first_caption_sentence` -> `opening_caption_prefix(subtitle_words, guard=HOOK_DUPLICATE_GUARD_SECONDS)`.
  It now joins every caption word starting inside the first `guard` seconds, in
  order, with **no** terminator scan (`_first_sentence_ends_complete` is no longer
  called there; it is still used by the selection layer at ~3073, untouched).
  Words without a usable numeric `start` are kept (unknown position, not
  known-outside); the scan breaks at the first word at or past `guard`.
  This kills both documented weaknesses: the terminator bias that turned the
  extract into a fragment on number-heavy screencast captions, and the
  "sentence running past 5 s makes the check inert" case.
- New helper `_comparable_word_list(text)` splits normalized text into words with
  per-word punctuation stripped (`_CAPTION_PUNCTUATION` constant next to
  `HOOK_DUPLICATE_GUARD_SECONDS`); the existing `_normalize_comparable_text`
  only strips the very end of the string.
- `teaser_repeats_first_caption` now compares word sequences instead of exact
  string equality.

### Overlap rule chosen

> The two word sequences must share a **common prefix** of at least
> `HOOK_CAPTION_MIN_MATCH_WORDS` words. Either side may be the longer one, but
> the overlap is always anchored at the clip opening's **first** word, never
> anywhere in the 5 s window. Floor = 2 words.

Two deliberate departures from the brief's suggested "contains" wording, both
forced by the brief's own requirement that the existing tests keep passing:

1. **Anchored prefix, not free containment.** Free containment would make the
   pinned test `test_sentence_after_the_opening_is_not_a_repeat` fail: with a
   time-based window the prefix is the whole opening, and the later sentence
   "Then it broke." *is* a substring of it. That sentence is not the opening, so
   the match must be anchored at word 0. A common prefix over the two word lists
   is the conservative reading of "prefix containment".
2. **Floor of 2, not more.** `SAMPLE_WORDS` (used by two pinned publish-path
   tests) is a 2-word opening — "Hello world" — so a 3-word floor would have
   thrown away behaviour the story already covered. 2 is exactly the width of
   the exact-match rule being replaced, so the new net is a strict superset of
   the old one's coverage on short openings while adding the mid-clip case. A
   single shared leading word is treated as coincidental phrasing.

Direction bias: the rule errs toward *missing*, since a false match drops a
fine teaser while a false miss ships the repetition. `HOOK_CAPTION_MIN_MATCH_WORDS`
is a module constant (not env-gated, per the out-of-scope list) if the floor ever
needs tuning.

### Tests (`tests/test_hook_teaser.py`)

- `test_hook_repeating_the_opening_words_in_pieces_is_a_repeat` — the required new
  case: punctuation-free caption words spanning the 5 s window, hook restating
  the opening in words at a mid-clip timestamp -> skipped. Also pins that an
  unrelated mid-clip teaser still passes.
- `test_one_word_overlap_is_not_enough` — pins the floor (1 shared word -> no
  match, 2 -> match).
- Existing exact-sentence, casing/punctuation and inert cases kept and passing.

TDD: both new tests were written first and observed failing
(`2 failed, 73 deselected`), then passed after the implementation. Verified
explicitly by stashing `reelsfy.py` and re-running (still 2 failed), then
unstashing (2 passed).

## Item 2 — duplicated comment

Deleted the second `# then simply publishes without a teaser.` at
`reelsfy.py:4807-4808`.

## Item 3 — `None`-inertness docstring overclaim

`publish_segment_with_optional_teaser` docstring now scopes the claim: `None`
leaves the **overlap check** (and hence teaser timing) byte-identical to pre-B4;
the caption net runs regardless because it only needs `subtitle_words`. Also
reworded "first caption sentence verbatim" -> "restating its opening words".

## Item 4 — skip log could raise

Both teaser logs (`overlaps the clip's first 5s` and the accepted-teaser line)
now format through `_srt_like_to_seconds`, falling back to plain `str()` of the
raw field when either side is unparseable, so the log cannot raise where the
documented never-raising predicates were called.

## Item 5 — overclaiming B1 test

`tests/test_hook_first.py`: `test_span_straddling_the_guard_end_rejected` ->
`test_candidate_starting_inside_the_guard_rejected`, comment reworded to pin
that the **start** inside the guard is what disqualifies the candidate.
Rename/comment only; what the test exercises is unchanged.

## Test commands and output

```
$ ./.venv/bin/python -m pytest tests/test_hook_teaser.py tests/test_hook_first.py tests/test_topic_boundary.py -q
160 passed in 2.42s

$ ./.venv/bin/python -m pytest tests/ -q --continue-on-collection-errors
75 failed, 784 passed, 5 skipped, 22 warnings, 7 errors in 42.76s
```

Baseline was 782 passed / 75 failed; now 784 passed / 75 failed — the two added
tests, no new failures. The 75 failures and 7 collection errors are the
pre-existing environmental ones.

## Uncertainties

- The word floor (2) is a judgement call made to keep the pinned 2-word
  `SAMPLE_WORDS` publish tests meaningful. If the reviewer would rather have 3,
  those two tests' fixture needs widening first.
- `HOOK_CAPTION_MIN_MATCH_WORDS` is untuned against real captions; no corpus was
  available to check the false-match rate on actual screencast output.
- Anchored-prefix vs. the brief's looser "contains" is a deliberate deviation
  (reason above); flagging it in case the intended semantics was looser.
- Commit is unsigned as instructed; `git log --format=%G? -1` prints `N`.
