"""
Tests for S3 (CORRECTED): boundary snapping + wiring the resolved topic
window into the LIVE traditional prompt.

Plan: docs/VIRAL_SELECTION_IMPROVEMENT_PLAN.md, story S3 (gaps G3, G4),
revised after the blocked dispatch proved the content-centric prompt is
dead code (overwritten by the traditional prompt, which is the one actually
sent). This suite covers:

1. ``snap_window_to_boundaries`` — pure, text-only refinement of the
   optimized window: start opens past the last in-cue sentence terminator
   (never earlier than the Task-13 topic-start cue), end trims a trailing
   partial sentence, and any parse failure returns the input unchanged.
2. ``refine_window_start_with_pause`` — prosodic START refinement against
   ``EnhancedAudioAnalyzer.detect_natural_boundaries`` within ±1.5 s of the
   text-snapped start; any audio failure degrades to text-only snapping
   without raising.
3. Wiring: the snapper runs on the primary path after the Task-13 trim and
   before the resolved window reaches the LIVE prompt via a "RESOLVED TOPIC
   WINDOW" instruction section; the dead content-centric prompt build is
   gone; the fallback path still builds its prompt with audio_context and
   still writes audio_analysis_summary.txt.

The OpenAI client and the heavy audio analyzer are replaced with mocks —
no network access and no real audio processing happens.
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


def _window(start_srt, end_srt, duration):
    return {
        "start_time": start_srt,
        "end_time": end_srt,
        "duration": duration,
        "title": "Window title",
    }


def _analyzer_cls(pauses, sr=22050):
    """A patched EnhancedAudioAnalyzer class whose detect_natural_boundaries
    returns `pauses` (seconds relative to the neighborhood start)."""
    cls = MagicMock()
    instance = MagicMock()
    instance.sr = sr
    instance.detect_natural_boundaries.return_value = list(pauses)
    cls.return_value = instance
    return cls


# ---------------------------------------------------------------------------
# snap_window_to_boundaries (pure helper)
# ---------------------------------------------------------------------------


class TestSnapWindowToBoundaries:
    """Pure, text-only window refinement."""

    def test_start_snaps_past_last_in_cue_terminator(self):
        cues = [_cue(0.0, 30.0, START_TEXT), _cue(30.0, 60.0, NO_PUNCT)]
        window = _window("00:00:00,000", "00:00:45,000", 45.0)

        result = reelsfy.snap_window_to_boundaries(window, cues)

        # '.' at index 10 of the 33-char cue spanning 30 s → 11/33 * 30 = 10 s
        assert result["start_time"] == _srt(10.0)
        assert result["end_time"] == "00:00:45,000"
        assert result["duration"] == pytest.approx(35.0)

    def test_window_already_past_sentence_start_unchanged(self):
        cues = [_cue(0.0, 30.0, START_TEXT), _cue(30.0, 60.0, NO_PUNCT)]
        window = _window("00:00:15,000", "00:00:45,000", 30.0)

        result = reelsfy.snap_window_to_boundaries(window, cues)

        # The only in-cue terminator maps to 10 s, before the window start —
        # the start never moves earlier, so the window is returned untouched.
        assert result is window
        assert result["start_time"] == "00:00:15,000"
        assert result["end_time"] == "00:00:45,000"
        assert result["duration"] == pytest.approx(30.0)

    def test_end_trimmed_to_last_terminator_in_end_cue(self):
        cues = [_cue(0.0, 30.0, NO_PUNCT), _cue(30.0, 60.0, END_TEXT)]
        window = _window("00:00:00,000", "00:00:45,000", 45.0)

        result = reelsfy.snap_window_to_boundaries(window, cues)

        # '.' at index 9 of the 30-char cue spanning 30 s → 30 + 10/30*30 = 40 s
        assert result["end_time"] == _srt(40.0)
        assert result["start_time"] == "00:00:00,000"
        assert result["duration"] == pytest.approx(40.0)

    def test_end_guard_binds_leaves_end_unchanged(self):
        cues = [_cue(0.0, 30.0, NO_PUNCT), _cue(30.0, 60.0, END_TEXT)]
        window = _window("00:00:00,000", "00:00:45,000", 45.0)

        result = reelsfy.snap_window_to_boundaries(window, cues, min_duration=50.0)

        # The trim would give 40 s < start + 50 → the edge is left unchanged.
        assert result["end_time"] == "00:00:45,000"
        assert result["duration"] == pytest.approx(45.0)

    def test_terminator_less_cues_unchanged(self):
        cues = [_cue(0.0, 30.0, NO_PUNCT), _cue(30.0, 60.0, NO_PUNCT)]
        window = _window("00:00:05,000", "00:00:45,000", 40.0)

        result = reelsfy.snap_window_to_boundaries(window, cues)

        assert result is window
        assert result["start_time"] == "00:00:05,000"
        assert result["end_time"] == "00:00:45,000"
        assert result["duration"] == pytest.approx(40.0)

    def test_min_start_floor_respected(self):
        cues = [_cue(0.0, 30.0, START_TEXT)]
        window = _window("00:00:02,000", "00:00:45,000", 43.0)

        result = reelsfy.snap_window_to_boundaries(
            window, cues, min_start_srt="00:00:12,000"
        )

        # The snap candidate (10 s) is earlier than the topic-start cue
        # floor (12 s) → the floor wins; the start is never moved before it.
        assert result["start_time"] == _srt(12.0)
        assert result["duration"] == pytest.approx(33.0)

    def test_malformed_start_timestamp_returns_input_unchanged(self):
        cues = [_cue(0.0, 30.0, START_TEXT)]
        window = {"start_time": "not-a-timestamp", "end_time": "00:00:45,000", "duration": 45.0}

        result = reelsfy.snap_window_to_boundaries(window, cues)

        assert result is window
        assert result["start_time"] == "not-a-timestamp"
        assert result["end_time"] == "00:00:45,000"

    def test_malformed_end_timestamp_returns_input_unchanged(self):
        cues = [_cue(0.0, 30.0, START_TEXT)]
        window = {"start_time": "00:00:00,000", "end_time": "garbage", "duration": 45.0}

        result = reelsfy.snap_window_to_boundaries(window, cues)

        assert result is window
        assert result["end_time"] == "garbage"

    def test_empty_segments_returned_unchanged(self):
        window = _window("00:00:00,000", "00:00:45,000", 45.0)

        result = reelsfy.snap_window_to_boundaries(window, [])

        assert result is window


# ---------------------------------------------------------------------------
# refine_window_start_with_pause (prosodic START refinement)
# ---------------------------------------------------------------------------


class TestRefineWindowStartWithPause:
    def test_start_moves_to_nearest_pause_in_neighborhood(self, tmp_path):
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"stub-audio")
        window = _window("00:00:20,000", "00:00:50,000", 30.0)
        cls = _analyzer_cls([1.0])

        with patch.object(
            sys.modules["enhanced_audio_analyzer"], "EnhancedAudioAnalyzer", cls
        ):
            result = reelsfy.refine_window_start_with_pause(
                window, str(audio_file), min_start_srt="00:00:18,000"
            )

        # The neighborhood is ±1.5 s → indices 18.5*sr .. 21.5*sr; the pause
        # is reported relative to the neighborhood start (18.5 + 1.0 = 19.5).
        cls.return_value.detect_natural_boundaries.assert_called_once_with(
            int(18.5 * 22050), int(21.5 * 22050)
        )
        assert result["start_time"] == _srt(19.5)
        assert result["duration"] == pytest.approx(30.5)

    def test_no_pauses_leaves_window_unchanged(self):
        window = _window("00:00:20,000", "00:00:50,000", 30.0)
        cls = _analyzer_cls([])

        with patch.object(
            sys.modules["enhanced_audio_analyzer"], "EnhancedAudioAnalyzer", cls
        ):
            result = reelsfy.refine_window_start_with_pause(window, "audio.wav")

        assert result is window
        assert result["start_time"] == "00:00:20,000"

    def test_missing_audio_path_never_builds_analyzer(self):
        window = _window("00:00:20,000", "00:00:50,000", 30.0)
        cls = _analyzer_cls([1.0])

        with patch.object(
            sys.modules["enhanced_audio_analyzer"], "EnhancedAudioAnalyzer", cls
        ):
            for audio_path in (None, "does-not-exist.wav"):
                result = reelsfy.refine_window_start_with_pause(window, audio_path)

        assert result is window
        cls.assert_not_called()

    def test_min_start_filters_earlier_pauses(self, tmp_path):
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"stub-audio")
        window = _window("00:00:20,000", "00:00:50,000", 30.0)
        cls = _analyzer_cls([0.5, 1.2])

        with patch.object(
            sys.modules["enhanced_audio_analyzer"], "EnhancedAudioAnalyzer", cls
        ):
            result = reelsfy.refine_window_start_with_pause(
                window, str(audio_file), min_start_srt="00:00:19,500"
            )

        # Pauses at 19.0 and 19.7 s; only 19.7 respects the 19.5 s floor.
        assert result["start_time"] == _srt(19.7)

    def test_all_pauses_below_min_start_leaves_window_unchanged(self):
        window = _window("00:00:20,000", "00:00:50,000", 30.0)
        cls = _analyzer_cls([0.5, 0.9])

        with patch.object(
            sys.modules["enhanced_audio_analyzer"], "EnhancedAudioAnalyzer", cls
        ):
            result = reelsfy.refine_window_start_with_pause(
                window, "audio.wav", min_start_srt="00:00:20,000"
            )

        assert result is window

    def test_analyzer_construction_failure_degrades_without_raising(self):
        window = _window("00:00:20,000", "00:00:50,000", 30.0)
        cls = MagicMock(side_effect=RuntimeError("audio load failed"))

        with patch.object(
            sys.modules["enhanced_audio_analyzer"], "EnhancedAudioAnalyzer", cls
        ):
            result = reelsfy.refine_window_start_with_pause(window, "audio.wav")

        assert result is window

    def test_detector_failure_degrades_without_raising(self):
        window = _window("00:00:20,000", "00:00:50,000", 30.0)
        cls = MagicMock()
        instance = MagicMock()
        instance.sr = 22050
        instance.detect_natural_boundaries.side_effect = RuntimeError("boom")
        cls.return_value = instance

        with patch.object(
            sys.modules["enhanced_audio_analyzer"], "EnhancedAudioAnalyzer", cls
        ):
            result = reelsfy.refine_window_start_with_pause(window, "audio.wav")

        assert result is window


# ---------------------------------------------------------------------------
# Wiring invariants (source-level, mirrors tests/test_topic_boundary.py)
# ---------------------------------------------------------------------------


class TestWiringInvariants:
    """The snapper sits after the Task-13 trim; the resolved window reaches
    the LIVE traditional prompt; the dead content-centric build is gone."""

    def test_snap_called_after_trim_before_resolved_section(self):
        primary_marker = "Using Topic-Focused Content Generation"
        primary_start = REELSFY_SOURCE.index(primary_marker)
        trim_call = REELSFY_SOURCE.find("trim_window_to_topic_start(", primary_start)
        snap_call = REELSFY_SOURCE.find("snap_window_to_boundaries(", primary_start)
        pause_call = REELSFY_SOURCE.find(
            "refine_window_start_with_pause(", primary_start
        )
        resolved_section = REELSFY_SOURCE.find("RESOLVED TOPIC WINDOW", primary_start)
        assert 0 < trim_call < snap_call < pause_call < resolved_section

    def test_dead_content_centric_prompt_no_longer_built(self):
        gen_source = inspect.getsource(reelsfy.generate_viral)
        assert "create_content_centric_prompt" not in gen_source
        assert "create_content_centric_system_prompt" not in gen_source
        assert "segment_info = {" not in gen_source

    def test_resolved_section_is_injected_into_the_live_prompt(self):
        gen_source = inspect.getsource(reelsfy.generate_viral)
        assert "RESOLVED TOPIC WINDOW" in gen_source
        assert "{resolved_window_section}{audio_context}" in gen_source

    def test_audio_todo_block_deleted_but_audio_pass_kept(self):
        assert "get_optimal_boundaries" not in REELSFY_SOURCE
        assert "# TODO: Use these boundaries" not in REELSFY_SOURCE
        assert "audio_analysis_summary.txt" in REELSFY_SOURCE
        assert "EnhancedAudioAnalyzer(audio_path)" in REELSFY_SOURCE

    def test_content_centric_functions_deleted_as_dead_code(self):
        # The content-centric prompt functions were production-dead (their
        # only call site in generate_viral was removed in Task 14) and have
        # been deleted entirely from reelsfy.py.
        assert "def create_content_centric_prompt" not in REELSFY_SOURCE
        assert "def create_content_centric_system_prompt" not in REELSFY_SOURCE


# ---------------------------------------------------------------------------
# Behavioral wiring: the resolved window reaches the LIVE prompt
# ---------------------------------------------------------------------------

# Primary-path cue layout:
#   cue 1: 0-15  (previous-topic tail, no terminators)
#   cue 2: 15-45 (final topic starts here; START_TEXT snaps start to 25.0)
#   cue 3: 45-60 (conclusion, no terminators)
PRIMARY_CUES = [
    _cue(0.0, 15.0, NO_PUNCT),
    _cue(15.0, 45.0, START_TEXT),
    _cue(45.0, 60.0, NO_PUNCT),
]

TRANSCRIPT_SRT = (
    "1\n00:00:00,000 --> 00:00:15,000\nFirst cue text here\n\n"
    "2\n00:00:15,000 --> 00:00:30,000\nSecond cue text here\n\n"
    "3\n00:00:30,000 --> 00:00:45,000\nThird cue text here\n"
)


def _topic_analysis_result():
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


def _run_generate_viral(
    topic_analysis, audio_path=None, output_dir=None, analyzer_cls=None
):
    """Run generate_viral with the LLM and heavy dependencies mocked.

    Returns (result, captured_messages, resolve_topic_start_cue mock).
    """
    analyzer_instance = MagicMock()
    analyzer_instance.analyze_transcript.return_value = topic_analysis
    fake_topic_analyzer = MagicMock(return_value=analyzer_instance)

    metadata = {"content": json.dumps({"title": "T", "description": "D"})}
    final_reply = json.dumps(
        {
            "segments": [
                {
                    "start_time": "00:00:25,000",
                    "end_time": "00:00:59,000",
                    "title": "T",
                    "description": "D",
                }
            ]
        }
    )

    captured = {}

    # **kwargs: since Task 15 the live call also carries
    # response_format={"type": "json_object"}; the stub accepts it so the
    # tests below exercise the primary (structured-output) path rather
    # than the degrade-and-retry fallback.
    def _capture(model, messages, n, stop, **kwargs):
        captured["messages"] = messages
        response = MagicMock()
        response.choices[0].message.content = final_reply
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
    ) as resolve_mock, patch.object(
        reelsfy.openai, "OpenAI", openai_module.OpenAI
    ), patch.object(
        reelsfy,
        "check_audio_file",
        return_value=(audio_path is not None, "stubbed"),
    ), patch.object(
        sys.modules["enhanced_audio_analyzer"],
        "EnhancedAudioAnalyzer",
        analyzer_cls or _analyzer_cls([]),
    ):
        result = reelsfy.generate_viral(
            TRANSCRIPT_SRT, audio_path=audio_path, output_dir=output_dir
        )

    return result, captured, resolve_mock


class TestGenerateViralResolvedWindowWiring:
    def test_primary_prompt_contains_resolved_window_with_snapped_values(self):
        result, captured, _ = _run_generate_viral(_topic_analysis_result())

        assert result, "generate_viral must return the merged JSON"
        user_content = captured["messages"][1]["content"]
        assert "RESOLVED TOPIC WINDOW" in user_content
        # Task-13 trim moves the start to cue 2 (15 s); the snapper then
        # opens past the in-cue terminator: 15 + 11/33 * 30 = 25 s.
        assert _srt(25.0) in user_content
        assert "00:01:00,000" in user_content
        assert "cue 2" in user_content
        # The LIVE (traditional) prompt is what was sent, not the dead
        # content-centric one.
        system_content = captured["messages"][0]["content"]
        assert system_content.startswith(
            "You are a Viral Segment Identifier and Instagram Content Creator"
        )

    def test_fallback_prompt_has_no_resolved_section_but_audio_pass_runs(
        self, tmp_path
    ):
        topic_analysis = {"error": "topic analysis exploded"}
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"stub-audio")
        analyzer_cls = _analyzer_cls([])

        result, captured, resolve_mock = _run_generate_viral(
            topic_analysis,
            audio_path=str(audio_file),
            output_dir=str(tmp_path),
            analyzer_cls=analyzer_cls,
        )

        assert result, "fallback path must still produce a result"
        resolve_mock.assert_not_called()
        user_content = captured["messages"][1]["content"]
        assert "RESOLVED TOPIC WINDOW" not in user_content
        assert "Audio Analysis Results" in user_content
        summary = tmp_path / "audio_analysis_summary.txt"
        assert summary.exists(), "fallback path must still write the summary file"
