"""
Tests for S4 (Task 15): deterministic window validation + structured outputs.

Plan: docs/VIRAL_SELECTION_IMPROVEMENT_PLAN.md, story S4 (gaps G3, G6) plus
the Task-14 ride-along. Covers:

1. ``validate_segment_against_window`` — pure, module-level validation of
   the LLM-returned segment: with a resolved window the returned times are
   clamped into it, and a clamped duration outside the allowed range is
   REPLACED by the resolved window's times (the heuristic + trimmed +
   snapped window is trusted over an out-of-range LLM answer); without a
   resolved window (fallback path) only the duration range is checked and
   out-of-range segments are left unchanged with a warning note (pre-S4
   behavior); unparseable returned times always leave the segment
   unchanged.
2. The Task-14 ride-along: the ``snap_window_to_boundaries`` call site in
   ``generate_viral`` forwards ``min_duration`` (source assertion plus a
   behavioral test with a non-default minimum).
3. Structured outputs: both LLM calls (``generate_metadata`` and
   ``generate_viral``) pass ``response_format={"type": "json_object"}``,
   and an endpoint that rejects the kwarg is retried once without it,
   with the legacy repair chain still able to produce the result.

The OpenAI client and heavy analyzers are replaced with mocks — no network
access and no real audio processing happens.
"""

import inspect
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


REELSFY_SOURCE = (REELSFY_DIR / "reelsfy.py").read_text()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# 33 chars, '.' at index 10 → snap offset (10 + 1) / 33
START_TEXT = "First part. Second part continues"
# 30 chars, '.' at index 9 → end offset (9 + 1) / 30
END_TEXT = "Something." + "w" * 20
NO_PUNCT = "no sentence punctuation anywhere in here"


def _srt(seconds):
    return reelsfy.format_timestamp(seconds)


def _cue(start, end, text):
    return {
        "start_time": start,
        "end_time": end,
        "start_time_srt": _srt(start),
        "end_time_srt": _srt(end),
        "text": text,
    }


def _segment(start_srt, end_srt, **extra):
    seg = {
        "start_time": start_srt,
        "end_time": end_srt,
        "title": "T",
        "description": "D",
    }
    seg.update(extra)
    return seg


def _window(start_srt, end_srt):
    return {
        "start_time": start_srt,
        "end_time": end_srt,
        "duration": 50.0,
        "title": "Resolved window title",
    }


# ---------------------------------------------------------------------------
# validate_segment_against_window (pure helper)
# ---------------------------------------------------------------------------

# A resolved window of 10 s → 60 s (span 50 s, inside [35, 59]).
WINDOW = _window("00:00:10,000", "00:01:00,000")


class TestValidateSegmentAgainstWindow:
    """Pure, module-level validation of the returned segment."""

    def test_start_before_resolved_start_is_clamped(self):
        seg = _segment("00:00:02,000", "00:00:50,000")

        result, notes = reelsfy.validate_segment_against_window(
            seg, WINDOW, min_duration=35.0, max_duration=59.0
        )

        assert result["start_time"] == "00:00:10,000"
        assert result["end_time"] == "00:00:50,000"
        # Duration recomputed from the clamped times: 50 - 10 = 40 s.
        assert result["duration"] == pytest.approx(40.0)
        assert notes, "a clamp must be explained in the notes"

    def test_end_after_resolved_end_is_clamped(self):
        seg = _segment("00:00:15,000", "00:01:30,000")

        result, notes = reelsfy.validate_segment_against_window(
            seg, WINDOW, min_duration=35.0, max_duration=59.0
        )

        assert result["start_time"] == "00:00:15,000"
        assert result["end_time"] == "00:01:00,000"
        assert result["duration"] == pytest.approx(45.0)
        assert notes

    def test_in_range_segment_unchanged_without_notes(self):
        seg = _segment("00:00:15,000", "00:00:50,000", duration=35.0)

        result, notes = reelsfy.validate_segment_against_window(
            seg, WINDOW, min_duration=35.0, max_duration=59.0
        )

        assert result["start_time"] == "00:00:15,000"
        assert result["end_time"] == "00:00:50,000"
        assert result["duration"] == pytest.approx(35.0)
        assert notes == []

    def test_input_segment_not_mutated_by_clamping(self):
        seg = _segment("00:00:02,000", "00:00:50,000")
        before = dict(seg)

        result, _ = reelsfy.validate_segment_against_window(
            seg, WINDOW, min_duration=35.0, max_duration=59.0
        )

        assert result is not seg
        assert seg == before, "the pure helper must not mutate its input"

    def test_out_of_range_after_clamping_replaced_by_window_times(self):
        seg = _segment("00:00:00,000", "00:00:05,000")

        result, notes = reelsfy.validate_segment_against_window(
            seg, WINDOW, min_duration=35.0, max_duration=59.0
        )

        # Clamping yields a degenerate 0 s segment → the resolved window
        # itself is trusted instead.
        assert result["start_time"] == "00:00:10,000"
        assert result["end_time"] == "00:01:00,000"
        assert result["duration"] == pytest.approx(50.0)
        assert notes, "the replacement must be explained in the notes"

    def test_no_resolved_window_out_of_range_left_unchanged_with_warning(self):
        seg = _segment("00:00:00,000", "00:01:30,000")  # 90 s

        result, notes = reelsfy.validate_segment_against_window(
            seg, None, min_duration=35.0, max_duration=59.0
        )

        # Fallback path (gate off / LLM pass failed): today's behavior —
        # the segment is left unchanged and the reason is logged.
        assert result["start_time"] == "00:00:00,000"
        assert result["end_time"] == "00:01:30,000"
        assert notes, "the warning must surface in the notes"

    def test_no_resolved_window_in_range_unchanged_without_notes(self):
        seg = _segment("00:00:00,000", "00:00:40,000")

        result, notes = reelsfy.validate_segment_against_window(
            seg, None, min_duration=35.0, max_duration=59.0
        )

        assert result["start_time"] == "00:00:00,000"
        assert result["end_time"] == "00:00:40,000"
        assert notes == []

    def test_no_resolved_window_too_short_left_unchanged_with_warning(self):
        seg = _segment("00:00:00,000", "00:00:05,000")  # 5 s

        result, notes = reelsfy.validate_segment_against_window(
            seg, None, min_duration=35.0, max_duration=59.0
        )

        assert result["start_time"] == "00:00:00,000"
        assert result["end_time"] == "00:00:05,000"
        assert notes

    @pytest.mark.parametrize("resolved", [None, {"start_time": "junk", "end_time": "junk"}])
    def test_unparseable_resolved_window_degrades_to_range_check(self, resolved):
        seg = _segment("00:00:00,000", "00:01:30,000")  # 90 s, out of range

        result, notes = reelsfy.validate_segment_against_window(
            seg, resolved, min_duration=35.0, max_duration=59.0
        )

        assert result["start_time"] == "00:00:00,000"
        assert result["end_time"] == "00:01:30,000"
        assert notes

    @pytest.mark.parametrize(
        "start,end", [("garbage", "00:00:50,000"), ("00:00:15,000", "not-a-time")]
    )
    def test_malformed_returned_times_left_unchanged(self, start, end):
        seg = _segment(start, end)

        result, notes = reelsfy.validate_segment_against_window(
            seg, WINDOW, min_duration=35.0, max_duration=59.0
        )

        assert result["start_time"] == start
        assert result["end_time"] == end
        assert notes, "the parse failure must be explained"

    def test_non_dict_segment_passed_through(self):
        result, notes = reelsfy.validate_segment_against_window(
            None, WINDOW, min_duration=35.0, max_duration=59.0
        )

        assert result is None
        assert notes


# ---------------------------------------------------------------------------
# Task-14 ride-along: the snap call site forwards min_duration
# ---------------------------------------------------------------------------


class TestSnapMinDurationRideAlong:
    def test_snap_call_site_passes_min_duration(self):
        gen_source = inspect.getsource(reelsfy.generate_viral)
        assert "min_duration=min_duration" in gen_source, (
            "the snap_window_to_boundaries call site must forward min_duration"
        )

    def test_snap_end_guard_honors_non_default_min_duration(self):
        """The resolved window's END depends on the caller's min_duration.

        Layout: window 0→45 with no topic-start trim (resolve returns
        None, so the start stays at 0); the end cue (30→60) holds a
        terminator that maps to 40 s. With the default minimum (35 s) the
        end trims to 40 s; with min_duration=50 the guard binds and the
        end stays at 45 s — only possible when the call site forwards
        min_duration.
        """
        cues = [
            _cue(0.0, 15.0, NO_PUNCT),
            _cue(15.0, 30.0, START_TEXT),
            _cue(30.0, 60.0, END_TEXT),
        ]
        topic_analysis = {
            "segments": cues,
            "concluding_topic": {
                "text": "The final topic text",
                "topic_keywords": ["final"],
                "topic_confidence": 0.9,
            },
            "supporting_segments": [cues[0]],
            "optimized_segment": {
                "start_time": "00:00:00,000",
                "end_time": "00:00:45,000",
                "duration": 45.0,
                "title": "Final topic title",
            },
        }

        _, calls_default = _run_generate_viral(
            topic_analysis,
            reply_segment=_segment("00:00:00,000", "00:00:40,000"),
            resolve_index=None,
        )
        _, calls_min50 = _run_generate_viral(
            topic_analysis,
            min_duration=50.0,
            reply_segment=_segment("00:00:00,000", "00:00:40,000"),
            resolve_index=None,
        )

        # The LIVE prompt carries the resolved window; only the
        # min_duration=50 run must keep the 45 s end.
        assert (
            "- Resolved end: 00:00:40,000"
            in calls_default[0]["messages"][1]["content"]
        )
        assert (
            "- Resolved end: 00:00:45,000"
            in calls_min50[0]["messages"][1]["content"]
        )


# ---------------------------------------------------------------------------
# Validation wiring in generate_viral
# ---------------------------------------------------------------------------


def _run_generate_viral(
    topic_analysis,
    reply_segment=None,
    min_duration=35.0,
    audio_path=None,
    resolve_index=2,
):
    """Run generate_viral with the LLM and heavy dependencies mocked.

    Returns (result, captured create-call kwargs list).
    """
    analyzer_instance = MagicMock()
    analyzer_instance.analyze_transcript.return_value = topic_analysis
    fake_topic_analyzer = MagicMock(return_value=analyzer_instance)

    metadata = {"content": json.dumps({"title": "M", "description": "MD"})}
    if reply_segment is None:
        reply_segment = _segment("00:00:25,000", "00:01:00,000")
    final_reply = json.dumps({"segments": [reply_segment]})

    calls = []

    def _capture(**kwargs):
        calls.append(kwargs)
        response = MagicMock()
        response.choices[0].message.content = final_reply
        return response

    openai_module = MagicMock()
    openai_module.OpenAI.return_value.chat.completions.create.side_effect = _capture

    with patch.object(
        topic_analyzer, "TopicAnalyzer", fake_topic_analyzer
    ), patch.object(
        reelsfy, "generate_metadata", return_value=metadata
    ), patch.object(
        reelsfy, "resolve_topic_start_cue", return_value=(resolve_index, [])
    ), patch.object(
        reelsfy.openai, "OpenAI", openai_module.OpenAI
    ), patch.object(
        reelsfy, "check_audio_file", return_value=(False, "stubbed")
    ):
        result = reelsfy.generate_viral(
            "srt content", audio_path=audio_path, min_duration=min_duration
        )

    return result, calls


PRIMARY_CUES = [
    # cue 1: 0-15  (previous-topic tail, no terminators)
    # cue 2: 15-45 (final topic starts here; START_TEXT snaps start to 25.0)
    # cue 3: 45-60 (conclusion, no terminators)
    _cue(0.0, 15.0, NO_PUNCT),
    _cue(15.0, 45.0, START_TEXT),
    _cue(45.0, 60.0, NO_PUNCT),
]


def _primary_topic_analysis():
    return {
        "segments": PRIMARY_CUES,
        "concluding_topic": {
            "text": "The final topic text",
            "topic_keywords": ["final"],
            "topic_confidence": 0.9,
        },
        "supporting_segments": [PRIMARY_CUES[0]],
        "optimized_segment": {
            "start_time": "00:00:00,000",
            "end_time": "00:01:00,000",
            "duration": 60.0,
            "title": "Final topic title",
        },
    }


class TestGenerateViralValidationWiring:
    """The validator runs right after the LLM reply parses, before the merge."""

    def test_primary_prompt_carries_resolved_window(self):
        result, calls = _run_generate_viral(_primary_topic_analysis())

        assert result, "generate_viral must return the merged JSON"
        user_content = calls[0]["messages"][1]["content"]
        assert "RESOLVED TOPIC WINDOW" in user_content
        assert _srt(25.0) in user_content  # snapped resolved start

    def test_out_of_range_reply_replaced_by_resolved_window(self):
        # Reply drifts before the resolved window (25 s → 60 s) with a
        # clamped duration of 5 s → the resolved window's times win.
        reply = _segment("00:00:00,000", "00:00:30,000")

        result, _ = _run_generate_viral(_primary_topic_analysis(), reply_segment=reply)

        merged = json.loads(result["content"])
        seg = merged["segments"][0]
        assert seg["start_time"] == _srt(25.0)
        assert seg["end_time"] == "00:01:00,000"
        assert seg["duration"] == pytest.approx(35.0)

    def test_in_range_reply_kept(self):
        # Exactly the resolved window → nothing to correct.
        reply = _segment("00:00:25,000", "00:01:00,000")

        result, _ = _run_generate_viral(_primary_topic_analysis(), reply_segment=reply)

        merged = json.loads(result["content"])
        seg = merged["segments"][0]
        assert seg["start_time"] == "00:00:25,000"
        assert seg["end_time"] == "00:01:00,000"

    def test_fallback_out_of_range_reply_left_unchanged(self):
        # Fallback path (topic analysis error): no resolved window, so the
        # 90 s reply is only warned about, not corrected.
        reply = _segment("00:00:00,000", "00:01:30,000")

        result, _ = _run_generate_viral(
            {"error": "topic analysis exploded"}, reply_segment=reply
        )

        merged = json.loads(result["content"])
        seg = merged["segments"][0]
        assert seg["start_time"] == "00:00:00,000"
        assert seg["end_time"] == "00:01:30,000"

    def test_malformed_reply_times_left_unchanged(self):
        reply = _segment("garbage", "00:01:30,000")

        result, _ = _run_generate_viral(
            _primary_topic_analysis(), reply_segment=reply
        )

        merged = json.loads(result["content"])
        seg = merged["segments"][0]
        assert seg["start_time"] == "garbage"
        assert seg["end_time"] == "00:01:30,000"


# ---------------------------------------------------------------------------
# Structured outputs (both LLM calls) + degradation
# ---------------------------------------------------------------------------


VALID_METADATA_REPLY = json.dumps(
    {"title": "T", "description": "D", "tags": "a, b"}
)


class TestStructuredOutputsGenerateMetadata:
    def test_response_format_passed(self):
        calls = []

        def _capture(**kwargs):
            calls.append(kwargs)
            response = MagicMock()
            response.choices[0].message.content = VALID_METADATA_REPLY
            return response

        openai_module = MagicMock()
        openai_module.OpenAI.return_value.chat.completions.create.side_effect = (
            _capture
        )

        with patch.object(reelsfy.openai, "OpenAI", openai_module.OpenAI):
            result = reelsfy.generate_metadata("transcript")

        assert json.loads(result["content"])["title"] == "T"
        assert len(calls) == 1
        assert calls[0].get("response_format") == {"type": "json_object"}

    def test_endpoint_rejecting_kwarg_retried_once_without_it(self):
        calls = []

        def _capture(**kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise TypeError("unexpected keyword argument 'response_format'")
            response = MagicMock()
            response.choices[0].message.content = VALID_METADATA_REPLY
            return response

        openai_module = MagicMock()
        openai_module.OpenAI.return_value.chat.completions.create.side_effect = (
            _capture
        )

        with patch.object(reelsfy.openai, "OpenAI", openai_module.OpenAI):
            result = reelsfy.generate_metadata("transcript")

        assert len(calls) == 2, "the rejected call must be retried once"
        assert "response_format" in calls[0]
        assert "response_format" not in calls[1]
        # Legacy parsing still produces the result.
        assert json.loads(result["content"])["title"] == "T"

    def test_endpoint_rejecting_kwarg_legacy_quoted_reply_repaired(self):
        """The legacy repair chain remains the last resort for the retry."""
        calls = []

        def _capture(**kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise TypeError("unsupported parameter")
            response = MagicMock()
            response.choices[0].message.content = "{'title': 'T2', 'description': 'D2'}"
            return response

        openai_module = MagicMock()
        openai_module.OpenAI.return_value.chat.completions.create.side_effect = (
            _capture
        )

        with patch.object(reelsfy.openai, "OpenAI", openai_module.OpenAI):
            result = reelsfy.generate_metadata("transcript")

        assert len(calls) == 2
        assert json.loads(result["content"])["title"] == "T2"


class TestStructuredOutputsGenerateViral:
    def test_response_format_passed(self):
        result, calls = _run_generate_viral(_primary_topic_analysis())

        assert result, "generate_viral must return the merged JSON"
        assert len(calls) == 1
        assert calls[0].get("response_format") == {"type": "json_object"}

    def test_endpoint_rejecting_kwarg_retried_once_without_it(self):
        reply = json.dumps(
            {"segments": [_segment("00:00:25,000", "00:01:00,000")]}
        )
        calls = []

        analyzer_instance = MagicMock()
        analyzer_instance.analyze_transcript.return_value = (
            _primary_topic_analysis()
        )
        fake_topic_analyzer = MagicMock(return_value=analyzer_instance)
        metadata = {"content": json.dumps({"title": "M", "description": "MD"})}

        def _capture(**kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise TypeError("unexpected keyword argument 'response_format'")
            response = MagicMock()
            response.choices[0].message.content = reply
            return response

        openai_module = MagicMock()
        openai_module.OpenAI.return_value.chat.completions.create.side_effect = (
            _capture
        )

        with patch.object(
            topic_analyzer, "TopicAnalyzer", fake_topic_analyzer
        ), patch.object(
            reelsfy, "generate_metadata", return_value=metadata
        ), patch.object(
            reelsfy, "resolve_topic_start_cue", return_value=(2, [])
        ), patch.object(
            reelsfy.openai, "OpenAI", openai_module.OpenAI
        ), patch.object(
            reelsfy, "check_audio_file", return_value=(False, "stubbed")
        ):
            result = reelsfy.generate_viral("srt content")

        assert len(calls) == 2, "the rejected call must be retried once"
        assert "response_format" in calls[0]
        assert "response_format" not in calls[1]
        # Legacy parsing still produces the merged result.
        merged = json.loads(result["content"])
        assert merged["segments"][0]["title"] == "T"
