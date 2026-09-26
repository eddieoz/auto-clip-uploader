"""
Tests for A5: completeness observability after selection.

`build_selection_summary` is a PURE builder (no I/O, never raises) that turns
the selection outcome into a small dict, and `generate_viral` prints it as a
single `SELECTION_SUMMARY {json}` line right after the S4 validation loop. The
line is the feedback signal for tuning A1-A4, so it must be honest and
machine-readable:

1. The happy path — all five reported facts.
2. `topic_complete` is passed through verbatim (True / False / None).
3. The first-caption heuristic: terminator -> complete; a first sentence
   that runs into the next cue -> incomplete.
4. Overlap, not containment: the cue that owns the window start counts even
   though it begins before the window.
5. Degradation: a missing / unparseable input degrades one field, never an
   exception, and never takes the rest of the line down with it.
6. Wiring: `generate_viral` really prints the line, with all five keys.

The OpenAI client and the heavy audio analyzer are mocked — no network and
no real audio processing. The `generate_viral` harness is reused from
`tests/test_boundary_snap.py` rather than reinvented.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
REELSFY_DIR = PROJECT_ROOT / "reels-clips-automator"

sys.path.insert(0, str(REELSFY_DIR))


def _ensure_stub(name):
    """Stub optional heavy modules missing from the test environment."""
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        sys.modules[name] = MagicMock()


for _name in ("pytube", "whisper", "audio_analyzer", "enhanced_audio_analyzer"):
    _ensure_stub(_name)

import reelsfy  # noqa: E402
import topic_analyzer  # noqa: E402

from tests.test_boundary_snap import (  # noqa: E402
    _run_generate_viral,
    _topic_analysis_result,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MARKER = "SELECTION_SUMMARY "
SUMMARY_KEYS = {
    "capture_span",
    "resolved_window",
    "chosen_duration",
    "topic_complete",
    "first_caption_sentence_complete",
}


def _srt(seconds):
    return reelsfy.format_timestamp(seconds)


def _cue(start, end, text):
    """A TopicAnalyzer-shaped cue: floats under start_time/end_time, SRT
    strings under start_time_srt/end_time_srt."""
    return {
        "start_time": start,
        "end_time": end,
        "start_time_srt": _srt(start),
        "end_time_srt": _srt(end),
        "text": text,
    }


def _window(start_srt, end_srt, duration=40.0):
    return {
        "start_time": start_srt,
        "end_time": end_srt,
        "duration": duration,
        "title": "Resolved window title",
    }


def _build(**kwargs):
    """build_selection_summary with the happy-path defaults, overridable."""
    params = {
        "segments": None,
        "resolved_window": None,
        "chosen_segment": None,
        "capture_span": 90.0,
        "topic_complete": None,
    }
    params.update(kwargs)
    return reelsfy.build_selection_summary(**params)


def _reject_constant(name):
    """json.loads parse_constant hook: the spec has no NaN / Infinity."""
    raise AssertionError(f"non-JSON constant {name!r} in the summary line")


def _strict_json_loads(payload):
    """json.loads with NaN / -Infinity / Infinity made hard errors.

    Plain ``json.loads`` happily accepts those three, which is exactly how a
    poisoned line kept passing every check; a strict parser is the contract.
    """
    return json.loads(payload, parse_constant=_reject_constant)


def _assert_strict_single_line(summary):
    """Assert the emitted line is one line of strict-JSON-parseable content."""
    line = MARKER + json.dumps(summary)
    assert "\n" not in line and "\r" not in line, (
        f"the marker line must stay a single line, got {line!r}"
    )
    return _strict_json_loads(line[len(MARKER):])


# ---------------------------------------------------------------------------
# build_selection_summary (pure builder)
# ---------------------------------------------------------------------------


class TestBuildSelectionSummaryHappyPath:
    def test_reports_all_five_facts(self):
        segments = [
            _cue(0.0, 20.0, "previous topic tail"),
            _cue(15.0, 40.0, "The renderer is fast. Now the details"),
            _cue(40.0, 60.0, "And that concludes the renderer topic"),
        ]
        window = _window(_srt(20.0), _srt(60.0))
        chosen = {
            "start_time": _srt(25.0),
            "end_time": _srt(59.0),
            "title": "T",
            "description": "D",
        }

        summary = _build(
            segments=segments,
            resolved_window=window,
            chosen_segment=chosen,
            capture_span=60.0,
            topic_complete=True,
        )

        assert isinstance(summary, dict)
        assert SUMMARY_KEYS.issubset(summary), "all five facts must be reported"
        assert summary["capture_span"] == pytest.approx(60.0)
        assert summary["resolved_window"] == {
            "start": _srt(20.0),
            "end": _srt(60.0),
        }
        # 59 - 25, recomputed from the chosen times.
        assert summary["chosen_duration"] == pytest.approx(34.0)
        assert summary["topic_complete"] is True
        assert summary["first_caption_sentence_complete"] is True

    def test_is_json_serializable_for_single_line_printing(self):
        summary = _build(
            segments=[_cue(0.0, 30.0, "A complete sentence.")],
            resolved_window=_window(_srt(0.0), _srt(30.0)),
            chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
            capture_span=30.0,
            topic_complete=False,
        )

        parsed = _assert_strict_single_line(summary)

        assert parsed["topic_complete"] is False

    def test_reports_exactly_the_five_facts_and_nothing_else(self):
        """A sixth key (caption text, title, notes) would silently bloat every
        logged line; the contract is the five completeness facts, so pin the
        key SET, not just the subset."""
        summary = _build(
            segments=[_cue(0.0, 30.0, "A complete sentence.")],
            resolved_window=_window(_srt(0.0), _srt(30.0)),
            chosen_segment={
                "start_time": _srt(0.0),
                "end_time": _srt(30.0),
                "title": "a title that must not leak into the line",
                "description": "a description that must not leak either",
            },
            capture_span=30.0,
            topic_complete=True,
        )

        assert set(summary) == SUMMARY_KEYS

    def test_line_stays_one_line_when_a_field_carries_control_characters(self):
        """The only field that can carry arbitrary text is the window's
        timestamp pair. A newline in it must still come out as one line."""
        summary = _build(
            segments=[_cue(0.0, 30.0, "A complete sentence.")],
            resolved_window={
                "start_time": "00:00:00,000",
                "end_time": "00:00:30,000",
                "duration": 30.0,
            },
            chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
            capture_span=30.0,
            topic_complete=True,
        )
        # A hostile caller can hand back anything under start_time/end_time;
        # pin the escaping at the dump level rather than trusting the inputs.
        poisoned = dict(summary, resolved_window={"start": "a\nb", "end": "c\rd"})

        line = MARKER + json.dumps(poisoned)

        assert "\n" not in line and "\r" not in line
        assert _strict_json_loads(line[len(MARKER):])["resolved_window"] == {
            "start": "a\nb",
            "end": "c\rd",
        }

    def test_resolved_window_is_none_on_the_fallback_path(self):
        summary = _build(
            segments=[_cue(0.0, 30.0, "A complete sentence.")],
            resolved_window=None,
            chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
            capture_span=30.0,
            topic_complete=None,
        )

        assert summary["resolved_window"] is None
        # No window -> no in-window cue selection is possible; report the
        # fact as unknown rather than guessing.
        assert summary["first_caption_sentence_complete"] is None
        assert summary["chosen_duration"] == pytest.approx(30.0)


class TestTopicCompleteIsVerbatim:
    @pytest.mark.parametrize("verdict", [True, False, None])
    def test_passthrough(self, verdict):
        summary = _build(
            segments=[_cue(0.0, 30.0, "A complete sentence.")],
            resolved_window=_window(_srt(0.0), _srt(30.0)),
            chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
            capture_span=30.0,
            topic_complete=verdict,
        )

        assert summary["topic_complete"] is verdict, (
            "the model's own verdict must be reported verbatim, never "
            "coerced to False"
        )

    def test_not_inferred_from_the_cue_list(self):
        """A verdict-less run with plenty of in-window cues still reports
        None: the summary never infers completeness from the cues."""
        segments = [_cue(0.0, 20.0, "One. Two."), _cue(20.0, 40.0, "Three. Four.")]

        summary = _build(
            segments=segments,
            resolved_window=_window(_srt(0.0), _srt(40.0)),
            chosen_segment={"start_time": _srt(0.0), "end_time": _srt(40.0)},
            capture_span=40.0,
            topic_complete=None,
        )

        assert summary["topic_complete"] is None
        assert summary["first_caption_sentence_complete"] is True


class TestFirstCaptionSentenceCompleteness:
    def test_first_sentence_ending_with_terminator_is_complete(self):
        segments = [
            _cue(20.0, 40.0, "The renderer is fast. and the rest continues"),
            _cue(40.0, 60.0, "more words here"),
        ]

        summary = _build(
            segments=segments,
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=True,
        )

        # The terminator ends the FIRST sentence; the cue text running on
        # afterwards must not turn the verdict negative.
        assert summary["first_caption_sentence_complete"] is True

    def test_terminator_only_at_the_very_end_is_complete(self):
        segments = [_cue(20.0, 60.0, "The whole cue is one finished thought.")]

        summary = _build(
            segments=segments,
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=True,
        )

        assert summary["first_caption_sentence_complete"] is True

    def test_terminatorless_cue_followed_by_a_new_sentence_is_complete(self):
        """No punctuation in the first cue, but the next cue opens a new
        sentence — the boundary is there, so the first sentence reads as
        finished."""
        segments = [
            _cue(20.0, 40.0, "this cue has no terminator at all"),
            _cue(40.0, 60.0, "A new sentence starts here"),
        ]

        summary = _build(
            segments=segments,
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=True,
        )

        assert summary["first_caption_sentence_complete"] is True

    def test_first_sentence_running_into_the_next_cue_is_incomplete(self):
        segments = [
            _cue(20.0, 40.0, "this cue has no terminator at all"),
            _cue(40.0, 60.0, "and it just keeps going lowercase"),
        ]

        summary = _build(
            segments=segments,
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=False,
        )

        assert summary["first_caption_sentence_complete"] is False

    def test_single_terminatorless_cue_is_incomplete(self):
        segments = [_cue(20.0, 60.0, "one lone cue that never finishes")]

        summary = _build(
            segments=segments,
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=True,
        )

        # No following cue to supply a boundary, and no terminator.
        assert summary["first_caption_sentence_complete"] is False

    def test_no_cue_overlapping_the_window_reports_unknown(self):
        segments = [_cue(0.0, 10.0, "Some text."), _cue(70.0, 80.0, "Other text.")]

        summary = _build(
            segments=segments,
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=True,
        )

        assert summary["first_caption_sentence_complete"] is None


class TestFirstSentenceRuleKnownBias:
    """The ratified rule is "the first sentence ends with a terminator", and
    that rule is NOT "the first sentence is really over". These tests
    CHARACTERIZE the known false positives so the bias is pinned in the suite
    instead of hidden in a docstring: decimals, `e.g.` and ellipses all read
    as sentence ends. The bias runs toward `complete`, and this fact is the
    tuning signal for A1-A4, so a systematically optimistic verdict matters.

    If one of these flips, the rule changed (deliberately) — do not "fix" the
    test to match the code."""

    @pytest.mark.parametrize(
        "text, why",
        [
            ("The matrix is 3.14 wide and it keeps going", "decimal"),
            ("so we set the scale to 0.5 and then the render", "decimal"),
            ("we shipped e.g. the new renderer today", "'e.g.'"),
            ("the pipeline is a bit slow... and then it dropped", "ellipsis"),
            ("wait... the render is still on screen", "leading ellipsis"),
        ],
    )
    def test_known_false_positive_reads_as_complete(self, text, why):
        """Screencasts about render settings mention decimals constantly, so
        this false-positive direction is systematic, not rare."""
        summary = _build(
            segments=[_cue(20.0, 60.0, text)],
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=True,
        )

        assert summary["first_caption_sentence_complete"] is True, (
            f"known {why} false positive — this is the documented bias, and it "
            f"is expected to be True; a flip here means the rule changed"
        )

    def test_genuinely_truncated_still_reads_as_incomplete(self):
        """The other direction of the bias must stay honest: a cue that really
        is cut off is still reported incomplete."""
        summary = _build(
            segments=[_cue(20.0, 60.0, "the render stuttered badly during playback")],
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=True,
        )

        assert summary["first_caption_sentence_complete"] is False


class TestFirstSentenceTerminatorScanIsFlat:
    """Pins the shape of the scan: a line break is not a boundary decision, it
    is just another character to look through."""

    def test_terminator_after_a_line_break_counts(self):
        assert reelsfy._first_sentence_ends_complete("first line\nsecond line.") is True

    def test_line_break_alone_is_not_a_terminator(self):
        assert reelsfy._first_sentence_ends_complete("first line\nsecond line") is False

    def test_carriage_return_is_not_a_terminator_either(self):
        assert reelsfy._first_sentence_ends_complete("first\r\nsecond") is False


class TestCueOpenerDoesNotFireOnContinuationPunctuation:
    """A Whisper cue that starts with `,` / `)` / a dash is a mid-sentence
    continuation, not a new sentence — reporting it as one claims the previous
    sentence finished when it demonstrably did not."""

    @pytest.mark.parametrize(
        "text, why",
        [
            (", which is why the render stalled", "leading comma"),
            (") and then it recovered", "leading paren"),
            ("- the second half runs on", "leading dash"),
            ("— the second half runs on", "leading em dash"),
            ("[sic] the render stuttered", "leading bracket"),
            ('"quoted continuation', "leading double quote"),
        ],
    )
    def test_continuation_punctuation_is_not_a_new_sentence(self, text, why):
        assert reelsfy._cue_opens_new_sentence({"text": text}) is False, (
            f"{why} must not be read as a new sentence"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "A new sentence starts here",
            "The renderer is fast",
        ],
    )
    def test_a_capital_still_counts_as_a_new_sentence(self, text):
        assert reelsfy._cue_opens_new_sentence({"text": text}) is True

    def test_continuation_punctuation_leaves_the_summary_incomplete(self):
        """The same case at the summary level: a terminatorless first cue
        followed by a comma-opening continuation is one unfinished sentence."""
        segments = [
            _cue(20.0, 40.0, "this cue has no terminator at all"),
            _cue(40.0, 60.0, ", which is why the render stalled"),
        ]

        summary = _build(
            segments=segments,
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=True,
        )

        assert summary["first_caption_sentence_complete"] is False


class TestCueSelectionUsesOverlap:
    def test_cue_owning_the_window_start_is_included(self):
        """The cue that owns the window start BEGINS before it. Full
        containment would drop it — the exact cue the check is about."""
        owner = _cue(15.0, 40.0, "The topic actually begins here. More words")
        segments = [_cue(0.0, 15.0, "previous topic tail"), owner]

        summary = _build(
            segments=segments,
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=True,
        )

        # The verdict is the OWNER's first sentence (which is complete), not
        # the discarded previous-topic tail's.
        assert summary["first_caption_sentence_complete"] is True

        # Make the owner incomplete and the tail complete: the verdict must
        # flip, which is only possible if the owner is the cue examined.
        owner["text"] = "the owner cue has no terminator anywhere"
        flipped = _build(
            segments=segments,
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=True,
        )

        assert flipped["first_caption_sentence_complete"] is False, (
            "the cue owning the window start must be the one judged; a "
            "containment filter would have reported the previous-topic tail"
        )

    def test_unparseable_cues_are_skipped_not_fatal(self):
        segments = [
            {"start_time": "not a time", "end_time": "nor this", "text": "junk"},
            _cue(20.0, 40.0, "A real sentence here."),
        ]

        summary = _build(
            segments=segments,
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=True,
        )

        assert summary["first_caption_sentence_complete"] is True

    def test_fallback_cue_shape_with_srt_strings_is_supported(self):
        """The SRT fallback parser emits SRT strings under start_time/end_time
        plus bare floats under start/end — the opposite dual shape."""
        segments = [
            {
                "start_time": _srt(20.0),
                "end_time": _srt(40.0),
                "start": 20.0,
                "end": 40.0,
                "text": "A fallback-shaped cue with a full stop.",
            }
        ]

        summary = _build(
            segments=segments,
            resolved_window=_window(_srt(20.0), _srt(60.0)),
            chosen_segment={"start_time": _srt(20.0), "end_time": _srt(59.0)},
            capture_span=60.0,
            topic_complete=True,
        )

        assert summary["first_caption_sentence_complete"] is True


class TestDegradation:
    def test_all_inputs_missing(self):
        summary = _build(
            segments=None,
            resolved_window=None,
            chosen_segment=None,
            capture_span=None,
            topic_complete=None,
        )

        assert isinstance(summary, dict)
        assert SUMMARY_KEYS.issubset(summary)
        assert summary["capture_span"] is None
        assert summary["resolved_window"] is None
        assert summary["chosen_duration"] is None
        assert summary["topic_complete"] is None
        assert summary["first_caption_sentence_complete"] is None

    def test_missing_capture_span_leaves_the_rest(self):
        summary = _build(
            segments=[_cue(0.0, 30.0, "A complete sentence.")],
            resolved_window=_window(_srt(0.0), _srt(30.0)),
            chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
            capture_span=None,
            topic_complete=True,
        )

        assert summary["capture_span"] is None
        assert summary["chosen_duration"] == pytest.approx(30.0)
        assert summary["first_caption_sentence_complete"] is True

    def test_unparseable_capture_span(self):
        summary = _build(
            capture_span="not a number",
            segments=[_cue(0.0, 30.0, "A complete sentence.")],
            resolved_window=_window(_srt(0.0), _srt(30.0)),
            chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
            topic_complete=True,
        )

        assert summary["capture_span"] is None
        assert summary["first_caption_sentence_complete"] is True

    def test_garbage_window_keeps_the_chosen_duration(self):
        summary = _build(
            segments=[_cue(0.0, 30.0, "A complete sentence.")],
            resolved_window={"start_time": "nope", "end_time": "nope"},
            chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
            capture_span=30.0,
            topic_complete=True,
        )

        assert summary["resolved_window"] is None
        assert summary["first_caption_sentence_complete"] is None
        assert summary["chosen_duration"] == pytest.approx(30.0)

    def test_chosen_segment_with_unparseable_times(self):
        summary = _build(
            segments=[_cue(0.0, 30.0, "A complete sentence.")],
            resolved_window=_window(_srt(0.0), _srt(30.0)),
            chosen_segment={"start_time": "???", "end_time": "???"},
            capture_span=30.0,
            topic_complete=True,
        )

        assert summary["chosen_duration"] is None
        assert summary["capture_span"] == pytest.approx(30.0)
        assert summary["topic_complete"] is True
        assert summary["first_caption_sentence_complete"] is True

    def test_chosen_segment_duration_falls_back_to_its_own_field(self):
        summary = _build(
            segments=[_cue(0.0, 30.0, "A complete sentence.")],
            resolved_window=_window(_srt(0.0), _srt(30.0)),
            chosen_segment={"start_time": "???", "end_time": "???", "duration": 42.0},
            capture_span=30.0,
            topic_complete=True,
        )

        assert summary["chosen_duration"] == pytest.approx(42.0)

    def test_segments_is_a_dict_not_a_list(self):
        summary = _build(
            segments={"not": "a list"},
            resolved_window=_window(_srt(0.0), _srt(30.0)),
            chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
            capture_span=30.0,
            topic_complete=True,
        )

        assert summary["first_caption_sentence_complete"] is None
        assert summary["chosen_duration"] == pytest.approx(30.0)

    def test_cue_without_text_reports_unknown(self):
        summary = _build(
            segments=[{"start_time": 0.0, "end_time": 30.0}],
            resolved_window=_window(_srt(0.0), _srt(30.0)),
            chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
            capture_span=30.0,
            topic_complete=True,
        )

        assert summary["first_caption_sentence_complete"] is None

    def test_topic_complete_of_an_unexpected_type_is_preserved_not_coerced(self):
        summary = _build(
            segments=[_cue(0.0, 30.0, "A complete sentence.")],
            resolved_window=_window(_srt(0.0), _srt(30.0)),
            chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
            capture_span=30.0,
            topic_complete="maybe",
        )

        assert summary["topic_complete"] == "maybe"

    def test_non_finite_capture_span_degrades_to_null(self):
        summary = _build(
            capture_span=float("nan"),
            segments=[_cue(0.0, 30.0, "A complete sentence.")],
            resolved_window=_window(_srt(0.0), _srt(30.0)),
            chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
            topic_complete=True,
        )

        assert summary["capture_span"] is None
        assert _assert_strict_single_line(summary)["capture_span"] is None

    def test_infinite_capture_span_degrades_to_null(self):
        for value in (float("inf"), float("-inf")):
            summary = _build(capture_span=value)

            assert summary["capture_span"] is None
            _assert_strict_single_line(summary)

    def test_non_finite_chosen_duration_degrades_to_null(self):
        """`capture_span` is a difference of two `_srt_like_to_seconds`
        results and `_srt_like_to_seconds("nan")` is nan, so the chosen
        duration is reachable with a non-finite value too."""
        for bad in (float("nan"), float("inf")):
            summary = _build(
                segments=[_cue(0.0, 30.0, "A complete sentence.")],
                resolved_window=_window(_srt(0.0), _srt(30.0)),
                chosen_segment={
                    "start_time": bad,
                    "end_time": _srt(30.0),
                },
                capture_span=30.0,
                topic_complete=True,
            )

            assert summary["chosen_duration"] is None
            _assert_strict_single_line(summary)

    def test_non_finite_chosen_duration_field_falls_back_to_null(self):
        summary = _build(
            chosen_segment={
                "start_time": "???",
                "end_time": "???",
                "duration": float("inf"),
            }
        )

        assert summary["chosen_duration"] is None
        _assert_strict_single_line(summary)

    def test_non_finite_window_reports_null(self):
        for bad in ("nan", "inf", "-inf"):
            summary = _build(
                segments=[_cue(0.0, 30.0, "A complete sentence.")],
                resolved_window={"start_time": bad, "end_time": _srt(30.0)},
                chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
                capture_span=30.0,
                topic_complete=True,
            )

            assert summary["resolved_window"] is None
            _assert_strict_single_line(summary)

    def test_never_raises_on_hostile_input(self):
        """Not "a dict came back" — the per-field contract: the poisoned
        capture span degrades to null, the four other facts still report their
        true values, and the emitted line survives a STRICT JSON parse (the
        lenient default parser is what hid this)."""

        class Boom:
            def get(self, *a, **k):
                raise RuntimeError("boom")

        summary = _build(
            segments=[Boom(), 42, None, _cue(0.0, 30.0, "A complete sentence.")],
            resolved_window=Boom(),
            chosen_segment=Boom(),
            capture_span=float("nan"),
            topic_complete=True,
        )

        assert isinstance(summary, dict)
        assert set(summary) == SUMMARY_KEYS

        # The naive `assert isinstance(summary, dict)` passed while
        # json.dumps wrote a bare NaN that jq and JSON.parse both reject.
        line = MARKER + json.dumps(summary)
        assert "NaN" not in line and "Infinity" not in line
        parsed = _assert_strict_single_line(summary)

        # Per field: only the poisoned / unusable ones degrade.
        assert parsed["capture_span"] is None
        assert parsed["resolved_window"] is None
        assert parsed["chosen_duration"] is None
        # ...while the two facts that need no parsing of the hostile objects
        # still report. The caption verdict is None, not False: a Boom window
        # is not a dict, so there is no defensible in-window cue list.
        assert parsed["topic_complete"] is True
        assert parsed["first_caption_sentence_complete"] is None

    def test_summary_field_isolate_degrades_a_blowing_up_helper(self):
        """The per-fact helpers guard themselves, so the builder's own
        try/except is only reachable if one of them is replaced by something
        that raises outright. What catches the mutation is `_summary_field`
        (the builder's outer guard is not involved), and the never-raises
        contract must still hold with the surviving facts still reported."""
        with patch.object(
            reelsfy,
            "_summary_chosen_duration",
            side_effect=RuntimeError("boom"),
        ):
            summary = _build(
                segments=[_cue(0.0, 30.0, "A complete sentence.")],
                resolved_window=_window(_srt(0.0), _srt(30.0)),
                chosen_segment={"start_time": _srt(0.0), "end_time": _srt(30.0)},
                capture_span=30.0,
                topic_complete=True,
            )

        assert isinstance(summary, dict)
        assert SUMMARY_KEYS.issubset(summary), (
            "a blowing-up helper must degrade to a partial line, not a crash"
        )
        assert summary["chosen_duration"] is None
        assert summary["capture_span"] == pytest.approx(30.0)
        assert summary["topic_complete"] is True
        assert summary["first_caption_sentence_complete"] is True
        _assert_strict_single_line(summary)


# ---------------------------------------------------------------------------
# Wiring: generate_viral really prints the line
# ---------------------------------------------------------------------------


def _summary_lines(capsys):
    return [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(MARKER)
    ]


class TestEmissionFromGenerateViral:
    def test_prints_one_line_with_all_five_keys(self, capsys):
        result, _captured, _resolve = _run_generate_viral(
            _topic_analysis_result(), resolve_return=(2, True, None, [])
        )

        assert result, "generate_viral must still return the merged JSON"
        lines = _summary_lines(capsys)
        assert len(lines) == 1, f"expected exactly one summary line, got {lines}"

        summary = json.loads(lines[0][len(MARKER):])
        assert SUMMARY_KEYS.issubset(summary)
        # PRIMARY_CUES span 0-60 s, so the capture span is 60.0.
        assert summary["capture_span"] == pytest.approx(60.0)
        # Cue 2 (15-45 s) carries START_TEXT; its terminator is at index 10,
        # so the snapper opens the window at 15 + 11/33*30 = 25.0 s.
        assert summary["resolved_window"] == {"start": _srt(25.0), "end": _srt(60.0)}
        assert summary["topic_complete"] is True
        assert "first_caption_sentence_complete" in summary
        assert "chosen_duration" in summary

    def test_reports_a_false_verdict_verbatim(self, capsys):
        _run_generate_viral(
            _topic_analysis_result(), resolve_return=(2, False, 1, [])
        )

        summary = json.loads(_summary_lines(capsys)[0][len(MARKER):])
        assert summary["topic_complete"] is False

    def test_reports_no_verdict_as_null(self, capsys):
        _run_generate_viral(
            _topic_analysis_result(), resolve_return=(2, None, None, [])
        )

        summary = json.loads(_summary_lines(capsys)[0][len(MARKER):])
        assert summary["topic_complete"] is None
        assert "topic_complete" in summary, "the key must be present even when null"

    def test_fallback_path_still_prints_a_line(self, capsys):
        """No topic analysis -> no resolved window, no verdict. The line must
        still be emitted (that is the point of a logging path)."""
        _run_generate_viral({"error": "topic analysis exploded"})

        lines = _summary_lines(capsys)
        assert len(lines) == 1
        summary = json.loads(lines[0][len(MARKER):])
        assert SUMMARY_KEYS.issubset(summary)
        assert summary["resolved_window"] is None
        assert summary["topic_complete"] is None
        assert summary["first_caption_sentence_complete"] is None

    def test_empty_selection_still_emits_one_summary_line(self, capsys):
        """Zero chosen segments used to emit ZERO lines, so the runs that
        selected nothing — the ones most worth reading — produced no summary
        at all. Every run must produce at least one parseable line."""
        result, _captured, _resolve = _run_generate_viral(
            _topic_analysis_result(),
            resolve_return=(2, True, None, []),
            final_reply=json.dumps({"segments": []}),
        )

        assert result, "generate_viral must still return the merged JSON"
        lines = _summary_lines(capsys)
        assert len(lines) == 1, f"expected exactly one summary line, got {lines}"

        summary = _strict_json_loads(lines[0][len(MARKER):])
        assert set(summary) == SUMMARY_KEYS
        # Nothing was chosen, so only the chosen-segment-derived fact is null —
        # the run-level facts still report.
        assert summary["chosen_duration"] is None
        assert summary["capture_span"] == pytest.approx(60.0)
        assert summary["resolved_window"] == {"start": _srt(25.0), "end": _srt(60.0)}
        assert summary["topic_complete"] is True

    def test_the_builder_never_raises_so_the_pipeline_survives(self, capsys):
        """Even if the builder blew up, the run must complete: the emission
        is wrapped so a diagnostics failure can never take the pipeline
        down."""
        result, _captured, _resolve = _run_generate_viral(
            _topic_analysis_result(), resolve_return=(2, True, None, [])
        )
        assert result
        assert len(_summary_lines(capsys)) == 1, "the healthy path prints a line"
        capsys.readouterr()  # drop the healthy run's output

        with patch.object(
            reelsfy, "build_selection_summary", side_effect=RuntimeError("boom")
        ):
            result, _captured, _resolve = _run_generate_viral(
                _topic_analysis_result(), resolve_return=(2, True, None, [])
            )

        assert result, "a raising summary builder must not break generate_viral"
        assert _summary_lines(capsys) == [], (
            "the emission is wrapped: a raising builder logs nothing rather "
            "than emitting a partial line"
        )
