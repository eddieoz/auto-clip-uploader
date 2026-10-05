"""
Tests for S2: LLM topic-start pass (trim previous-topic tail).

Plan: docs/VIRAL_SELECTION_IMPROVEMENT_PLAN.md, story S2 (gaps G1b, G2).

Replay-buffer captures (60-90 s) often start with the tail of the PREVIOUS
topic while the final topic concludes near the capture's end. This suite
covers:

1. ``resolve_topic_start_cue`` — the numbered cue-list prompt (full capture,
   no truncation), integer reply parsing, validation, the TOPIC_BOUNDARY_LLM
   gate, and total failure degradation to None.
2. ``trim_window_to_topic_start`` — window start moved to the resolved cue,
   duration recomputed, supporting segments filtered; a cue at or before the
   current window start leaves everything unchanged.
3. Wiring invariants in reelsfy.py: the pass runs on the primary path before
   ``segment_info`` is built, the gate defaults to "true", and the fallback
   path is untouched.

The OpenAI client is replaced with mocks — no network access happens.
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


REELSFY_SOURCE = (REELSFY_DIR / "reelsfy.py").read_text()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_segments():
    """Build a segment list shaped like TopicAnalyzer's to_dict() output."""
    return [
        {
            "start_time": 0.0,
            "end_time": 15.0,
            "start_time_srt": "00:00:00,000",
            "end_time_srt": "00:00:15,000",
            "text": "So last week we finished the old exporter discussion",
        },
        {
            "start_time": 15.0,
            "end_time": 20.0,
            "start_time_srt": "00:00:15,000",
            "end_time_srt": "00:00:20,000",
            "text": " wrapping up the previous topic tail",
        },
        {
            "start_time": 20.0,
            "end_time": 40.0,
            "start_time_srt": "00:00:20,000",
            "end_time_srt": "00:00:40,000",
            "text": "Today we start the new renderer deep dive",
        },
        {
            "start_time": 40.0,
            "end_time": 55.0,
            "start_time_srt": "00:00:40,000",
            "end_time_srt": "00:00:55,000",
            "text": "The renderer benchmark numbers look great",
        },
        {
            "start_time": 60.0,
            "end_time": 85.0,
            "start_time_srt": "00:01:00,000",
            "end_time_srt": "00:01:25,000",
            "text": "And that concludes the renderer topic, thanks for watching",
        },
    ]


def _make_optimized_segment():
    """Optimized window covering the whole capture (the untrimmed default)."""
    return {
        "start_time": "00:00:00,000",
        "end_time": "00:01:25,000",
        "duration": 85.0,
        "title": "Previous tail — Renderer deep dive",
    }


def _make_supporting_segments():
    """Two supporting segments: one before the trim point, one after."""
    return [
        {
            "start_time": 0.0,
            "end_time": 15.0,
            "start_time_srt": "00:00:00,000",
            "end_time_srt": "00:00:15,000",
            "text": "So last week we finished the old exporter discussion",
        },
        {
            "start_time": 30.0,
            "end_time": 40.0,
            "start_time_srt": "00:00:30,000",
            "end_time_srt": "00:00:40,000",
            "text": "Context for the renderer deep dive",
        },
    ]


def _llm_client_replying_with(text):
    """Build a patched OpenAI client whose reply is `text`."""
    mock_openai = MagicMock()
    mock_client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = text
    mock_client.chat.completions.create.return_value = response
    mock_openai.OpenAI.return_value = mock_client
    return mock_openai


def _capture_prompt(monkeypatch):
    """Return the user prompt built with TOPIC_BOUNDARY_LLM=true.

    Uses the same mock-capture pattern as the other prompt tests: the
    OpenAI client is replaced and the request's messages are recorded, so
    no network access happens.
    """
    captured = {}

    def _capture(model, messages, n, stop):
        captured["messages"] = messages
        response = MagicMock()
        response.choices[0].message.content = "3"
        return response

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.chat.completions.create.side_effect = _capture
    monkeypatch.setenv("TOPIC_BOUNDARY_LLM", "true")
    with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
        reelsfy.resolve_topic_start_cue("srt content", _make_segments())
    return captured["messages"][-1]["content"]


def _resolve_with_reply(monkeypatch, reply, segments=None):
    """Run resolve_topic_start_cue with the LLM pinned to `reply`.

    The gate is left unset so the default ("true") applies; tests that
    exercise the disabled gate set the variable before calling.
    """
    if segments is None:
        segments = _make_segments()
    monkeypatch.delenv("TOPIC_BOUNDARY_LLM", raising=False)
    mock_openai = _llm_client_replying_with(reply)
    with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
        result = reelsfy.resolve_topic_start_cue("srt content", segments)
    return result, mock_openai


# ---------------------------------------------------------------------------
# resolve_topic_start_cue
# ---------------------------------------------------------------------------


class TestResolveTopicStartCue:
    """Integer reply parsing, prompt coverage, and failure degradation."""

    @pytest.mark.parametrize("reply", ["3", "cue 3", " 3 ", "Cue 3"])
    def test_valid_reply_returns_cue_index(self, monkeypatch, reply):
        result, mock_openai = _resolve_with_reply(monkeypatch, reply)
        assert result == (3, None, None, [])
        mock_openai.OpenAI.assert_called_once()

    def test_prompt_contains_every_cue_text(self, monkeypatch):
        segments = _make_segments()
        captured = {}

        def _capture(model, messages, n, stop):
            captured["messages"] = messages
            response = MagicMock()
            response.choices[0].message.content = "3"
            return response

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = (
            _capture
        )
        monkeypatch.setenv("TOPIC_BOUNDARY_LLM", "true")
        with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
            reelsfy.resolve_topic_start_cue("srt content", segments)

        user_content = captured["messages"][-1]["content"]
        for index, segment in enumerate(segments, start=1):
            assert f"cue {index}:" in user_content
            assert segment["text"] in user_content
        # The final topic concludes at the end; the prompt must say so.
        assert "concludes" in user_content.lower()
        assert "begins" in user_content.lower()

    def test_exception_returns_none_without_raising(self, monkeypatch):
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = (
            RuntimeError("api down")
        )
        monkeypatch.setenv("TOPIC_BOUNDARY_LLM", "true")
        with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
            result = reelsfy.resolve_topic_start_cue(
                "srt content", _make_segments()
            )
        assert result == (None, None, None, [])

    @pytest.mark.parametrize(
        "reply", ["sorry, I cannot help with that", "", "none"]
    )
    def test_unparseable_reply_returns_none(self, monkeypatch, reply):
        result, _ = _resolve_with_reply(
            monkeypatch, reply, segments=_make_segments()
        )
        assert result == (None, None, None, [])

    @pytest.mark.parametrize("reply", ["0", "99", "9999"])
    def test_out_of_range_reply_returns_none(self, monkeypatch, reply):
        result, _ = _resolve_with_reply(
            monkeypatch, reply, segments=_make_segments()
        )
        assert result == (None, None, None, [])

    def test_gate_off_never_constructs_client(self, monkeypatch):
        segments = _make_segments()
        mock_openai = MagicMock()
        monkeypatch.setenv("TOPIC_BOUNDARY_LLM", "false")
        with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
            result = reelsfy.resolve_topic_start_cue("srt content", segments)
        assert result == (None, None, None, [])
        mock_openai.OpenAI.assert_not_called()

    def test_gate_defaults_to_enabled(self, monkeypatch):
        result, mock_openai = _resolve_with_reply(monkeypatch, "3")
        assert result == (3, None, None, [])
        mock_openai.OpenAI.assert_called_once()

    def test_empty_segments_returns_none_without_llm(self, monkeypatch):
        mock_openai = MagicMock()
        monkeypatch.setenv("TOPIC_BOUNDARY_LLM", "true")
        with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
            result = reelsfy.resolve_topic_start_cue("srt content", [])
        assert result == (None, None, None, [])
        mock_openai.OpenAI.assert_not_called()

    # --- Story A3: completeness signal (topic_complete / earlier_start_cue) ---

    def test_complete_reply_returns_none_earlier(self, monkeypatch):
        reply = (
            '{"topic_complete": true, "cue_number": 3, '
            '"earlier_start_cue": null, "hook_candidates": []}'
        )
        result = _resolve_with_reply(monkeypatch, reply)[0]
        assert result == (3, True, None, [])

    def test_incomplete_reply_reports_earlier_cue(self, monkeypatch):
        reply = (
            '{"topic_complete": false, "cue_number": 5, '
            '"earlier_start_cue": 2, "hook_candidates": []}'
        )
        result = _resolve_with_reply(monkeypatch, reply)[0]
        # earlier_start_cue is carried for the caller to consider.
        assert result == (5, False, 2, [])

    def test_incomplete_reply_without_earlier_cue_yields_none(self, monkeypatch):
        # topic_complete false but no usable earlier cue -> no extension.
        reply = (
            '{"topic_complete": false, "cue_number": 3, '
            '"earlier_start_cue": null, "hook_candidates": []}'
        )
        result = _resolve_with_reply(monkeypatch, reply)[0]
        assert result == (3, False, None, [])

    def test_earlier_cue_after_cue_number_is_rejected(self, monkeypatch):
        # Extension only moves the start earlier; a later index is ignored.
        reply = (
            '{"topic_complete": false, "cue_number": 2, '
            '"earlier_start_cue": 4, "hook_candidates": []}'
        )
        result = _resolve_with_reply(monkeypatch, reply)[0]
        assert result == (2, False, None, [])

    def test_earlier_cue_out_of_range_is_rejected(self, monkeypatch):
        reply = (
            '{"topic_complete": false, "cue_number": 3, '
            '"earlier_start_cue": 99, "hook_candidates": []}'
        )
        result = _resolve_with_reply(monkeypatch, reply)[0]
        assert result == (3, False, None, [])

    def test_topic_complete_missing_is_unknown_not_complete(self, monkeypatch):
        # A legacy reply carries no verdict: topic_complete stays None so no
        # caller reports a completeness the model never gave.
        reply = '{"cue_number": 3, "hook_candidates": []}'
        result = _resolve_with_reply(monkeypatch, reply)[0]
        assert result == (3, None, None, [])

    def test_incomplete_is_reported_even_when_earlier_cue_is_unusable(self, monkeypatch):
        # The model DID say "incomplete"; only the cue it offered was rejected.
        # topic_complete=False must survive, or the A5 observability log would
        # report this capture as complete and hide the signal it measures.
        reply = (
            '{"topic_complete": false, "cue_number": 3, '
            '"earlier_start_cue": 99, "hook_candidates": []}'
        )
        result = _resolve_with_reply(monkeypatch, reply)[0]
        assert result == (3, False, None, [])

    @pytest.mark.parametrize("flag", ["false", 0, 1, None])
    def test_non_bool_topic_complete_is_ignored(self, monkeypatch, flag):
        # Only a real JSON false means "incomplete"; a string/0/1 does not, so
        # the VERDICT stays None and the summary log never reports a
        # completeness the model did not give. The explicit, strictly-earlier
        # cue named alongside it is a separate, more specific signal and is
        # still honoured — dropping it is what left clips starting mid-topic.
        reply = json.dumps(
            {
                "topic_complete": flag,
                "cue_number": 5,
                "earlier_start_cue": 2,
                "hook_candidates": [],
            }
        )
        result = _resolve_with_reply(monkeypatch, reply)[0]
        assert result == (5, None, 2, [])

    def test_malformed_reply_falls_back_to_legacy(self, monkeypatch):
        # No JSON at all -> all-legacy shape.
        result = _resolve_with_reply(monkeypatch, "no json here")[0]
        assert result == (None, None, None, [])

    def test_prompt_contains_complete_and_steering_instructions(self, monkeypatch):
        segments = _make_segments()
        captured = {}

        def _capture(model, messages, n, stop):
            captured["messages"] = messages
            response = MagicMock()
            response.choices[0].message.content = '{"topic_complete": true, "cue_number": 3}'
            return response

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = _capture
        monkeypatch.setenv("TOPIC_BOUNDARY_LLM", "true")
        with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
            reelsfy.resolve_topic_start_cue("srt content", segments)

        user_content = captured["messages"][-1]["content"]
        assert "topic_complete" in user_content
        assert "earlier_start_cue" in user_content

    # --- Story B3: prompt steers hooks away from the first sentence ---

    def test_prompt_forbids_first_sentence_and_prefers_mid_topic(
        self, monkeypatch
    ):
        # The clip OPENS with the final topic's first sentence, so a hook
        # proposed there merely repeats the opening. The prompt must forbid
        # that and point the model at engaging mid-topic sentences.
        user_content = _capture_prompt(monkeypatch)
        assert "MUST NOT be the first sentence of the final topic" in user_content
        assert "mid-topic" in user_content

    def test_steering_sits_in_the_hook_candidates_bullet(self, monkeypatch):
        # One coherent rule set, not the constraint split off elsewhere.
        user_content = _capture_prompt(monkeypatch)
        bullet = next(
            line
            for line in user_content.splitlines()
            if line.startswith('- "hook_candidates"')
        )
        assert "MUST NOT be the first sentence of the final topic" in bullet
        assert "mid-topic" in bullet

    def test_gate_off_returns_legacy_none(self, monkeypatch):
        segments = _make_segments()
        mock_openai = MagicMock()
        monkeypatch.setenv("TOPIC_BOUNDARY_LLM", "false")
        with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
            result = reelsfy.resolve_topic_start_cue("srt content", segments)
        assert result == (None, None, None, [])
        mock_openai.OpenAI.assert_not_called()


# ---------------------------------------------------------------------------
# trim_window_to_topic_start
# ---------------------------------------------------------------------------


class TestTrimWindowToTopicStart:
    """Window start moves to the cue; end time and recency anchor untouched."""

    def test_valid_cue_moves_window_start(self):
        segments = _make_segments()
        optimized = _make_optimized_segment()
        supporting = _make_supporting_segments()

        new_optimized, new_supporting = reelsfy.trim_window_to_topic_start(
            3, optimized, supporting, segments
        )

        assert new_optimized["start_time"] == "00:00:20,000"
        # Duration recomputed from the new start to the untouched end time.
        assert new_optimized["duration"] == pytest.approx(85.0 - 20.0)
        assert new_optimized["end_time"] == "00:01:25,000"
        # The supporting segment that begins before the new start is dropped;
        # the later one is kept.
        assert [seg["text"] for seg in new_supporting] == [
            "Context for the renderer deep dive"
        ]

    def test_cue_at_window_start_leaves_window_unchanged(self):
        segments = _make_segments()
        optimized = _make_optimized_segment()
        supporting = _make_supporting_segments()

        new_optimized, new_supporting = reelsfy.trim_window_to_topic_start(
            1, optimized, supporting, segments
        )

        assert new_optimized["start_time"] == "00:00:00,000"
        assert new_optimized["duration"] == pytest.approx(85.0)
        assert len(new_supporting) == 2

    def test_out_of_range_cue_leaves_window_unchanged(self):
        segments = _make_segments()
        optimized = _make_optimized_segment()
        supporting = _make_supporting_segments()

        new_optimized, new_supporting = reelsfy.trim_window_to_topic_start(
            99, optimized, supporting, segments
        )

        assert new_optimized["start_time"] == "00:00:00,000"
        assert new_optimized["duration"] == pytest.approx(85.0)
        assert len(new_supporting) == 2

    def test_none_cue_leaves_window_unchanged(self):
        segments = _make_segments()
        optimized = _make_optimized_segment()
        supporting = _make_supporting_segments()

        new_optimized, new_supporting = reelsfy.trim_window_to_topic_start(
            None, optimized, supporting, segments
        )

        assert new_optimized["start_time"] == "00:00:00,000"
        assert new_optimized["duration"] == pytest.approx(85.0)
        assert len(new_supporting) == 2


# ---------------------------------------------------------------------------
# Wiring invariants (source-level, mirrors tests/test_reelsfy_bugfixes.py)
# ---------------------------------------------------------------------------


class TestWiringInvariants:
    """The pass must sit on the primary path, gated, before segment_info."""

    def test_gate_helper_defaults_to_true(self):
        assert "TOPIC_BOUNDARY_LLM" in REELSFY_SOURCE
        assert "os.getenv('TOPIC_BOUNDARY_LLM', 'true')" in REELSFY_SOURCE

    def test_primary_path_calls_resolve_before_window_is_consumed(self):
        """S3: resolve → trim → snap → the resolved window reaches the live
        prompt's RESOLVED TOPIC WINDOW section (the former segment_info /
        content-centric prompt build was dead code and has been removed)."""
        primary_marker = "Using Topic-Focused Content Generation"
        primary_start = REELSFY_SOURCE.index(primary_marker)
        resolve_call = REELSFY_SOURCE.find("resolve_topic_start_cue(", primary_start)
        trim_call = REELSFY_SOURCE.find("trim_window_to_topic_start(", primary_start)
        snap_call = REELSFY_SOURCE.find("snap_window_to_boundaries(", primary_start)
        resolved_section = REELSFY_SOURCE.find(
            "RESOLVED TOPIC WINDOW", primary_start
        )
        assert 0 < resolve_call < trim_call < snap_call < resolved_section

    def test_trim_applied_on_primary_path(self):
        primary_marker = "Using Topic-Focused Content Generation"
        primary_start = REELSFY_SOURCE.index(primary_marker)
        assert (
            REELSFY_SOURCE.find("trim_window_to_topic_start(", primary_start) > 0
        )

    def test_fallback_path_does_not_call_the_pass(self):
        fallback_marker = "Fallback to Traditional Analysis"
        fallback_start = REELSFY_SOURCE.index(fallback_marker)
        fallback_region = REELSFY_SOURCE[fallback_start:]
        assert "resolve_topic_start_cue(" not in fallback_region
