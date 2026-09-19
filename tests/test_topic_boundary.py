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
        assert result == 3
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
        assert result is None

    @pytest.mark.parametrize(
        "reply", ["sorry, I cannot help with that", "", "none"]
    )
    def test_unparseable_reply_returns_none(self, monkeypatch, reply):
        result, _ = _resolve_with_reply(
            monkeypatch, reply, segments=_make_segments()
        )
        assert result is None

    @pytest.mark.parametrize("reply", ["0", "99", "9999"])
    def test_out_of_range_reply_returns_none(self, monkeypatch, reply):
        result, _ = _resolve_with_reply(
            monkeypatch, reply, segments=_make_segments()
        )
        assert result is None

    def test_gate_off_never_constructs_client(self, monkeypatch):
        segments = _make_segments()
        mock_openai = MagicMock()
        monkeypatch.setenv("TOPIC_BOUNDARY_LLM", "false")
        with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
            result = reelsfy.resolve_topic_start_cue("srt content", segments)
        assert result is None
        mock_openai.OpenAI.assert_not_called()

    def test_gate_defaults_to_enabled(self, monkeypatch):
        result, mock_openai = _resolve_with_reply(monkeypatch, "3")
        assert result == 3
        mock_openai.OpenAI.assert_called_once()

    def test_empty_segments_returns_none_without_llm(self, monkeypatch):
        mock_openai = MagicMock()
        monkeypatch.setenv("TOPIC_BOUNDARY_LLM", "true")
        with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
            result = reelsfy.resolve_topic_start_cue("srt content", [])
        assert result is None
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
