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

import contextlib
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

    def test_max_duration_rejects_a_pause_earlier_than_the_cap(self, tmp_path):
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"stub-audio")
        # 60 s end with the start already sitting exactly on the 59 s cap
        # (60 - 59 = 1 s), which is where the earlier-cue clamp leaves a
        # window. The neighborhood is ±1.5 s, so the 0.2 s pause would pull the
        # start back to 0.2 s and make the window 59.8 s — over the cap.
        window = _window(_srt(1.0), _srt(60.0), 59.0)
        cls = _analyzer_cls([0.2])

        with patch.object(
            sys.modules["enhanced_audio_analyzer"], "EnhancedAudioAnalyzer", cls
        ):
            result = reelsfy.refine_window_start_with_pause(
                window, str(audio_file), max_duration=59.0
            )

        assert result is window
        assert result["start_time"] == _srt(1.0)
        assert result["duration"] == pytest.approx(59.0)

    def test_max_duration_leaves_a_window_under_the_cap_free_to_refine(
        self, tmp_path
    ):
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"stub-audio")
        # 20 s start on a 60 s end = 40 s, well under the 59 s cap, so the cap
        # must not switch the refinement off: the nearest pause still applies.
        window = _window(_srt(20.0), _srt(60.0), 40.0)
        cls = _analyzer_cls([1.0])

        with patch.object(
            sys.modules["enhanced_audio_analyzer"], "EnhancedAudioAnalyzer", cls
        ):
            result = reelsfy.refine_window_start_with_pause(
                window, str(audio_file), max_duration=59.0
            )

        assert result["start_time"] == _srt(19.5)
        assert result["duration"] == pytest.approx(40.5)

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
    topic_analysis,
    audio_path=None,
    output_dir=None,
    analyzer_cls=None,
    resolve_return=(2, None, None, []),
    extra_patches=(),
):
    """Run generate_viral with the LLM and heavy dependencies mocked.

    ``resolve_return`` is what the mocked ``resolve_topic_start_cue`` hands
    back, ``extra_patches`` lets a test spy on the window helpers while the
    real implementations still run.

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

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(topic_analyzer, "TopicAnalyzer", fake_topic_analyzer)
        )
        stack.enter_context(
            patch.object(reelsfy, "generate_metadata", return_value=metadata)
        )
        resolve_mock = stack.enter_context(
            patch.object(
                reelsfy, "resolve_topic_start_cue", return_value=resolve_return
            )
        )
        stack.enter_context(patch.object(reelsfy.openai, "OpenAI", openai_module.OpenAI))
        stack.enter_context(
            patch.object(
                reelsfy,
                "check_audio_file",
                return_value=(audio_path is not None, "stubbed"),
            )
        )
        stack.enter_context(
            patch.object(
                sys.modules["enhanced_audio_analyzer"],
                "EnhancedAudioAnalyzer",
                analyzer_cls or _analyzer_cls([]),
            )
        )
        for patcher in extra_patches:
            stack.enter_context(patcher)

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


# ---------------------------------------------------------------------------
# Story A4: extend_window_to_earlier_topic_start
# ---------------------------------------------------------------------------

# Cue layout for the extension tests:
#   cue 1:  0-10   cue 2: 10-20   cue 3: 20-40   cue 4: 40-55
EXT_CUES = [
    _cue(0.0, 10.0, NO_PUNCT),
    _cue(10.0, 20.0, NO_PUNCT),
    _cue(20.0, 40.0, START_TEXT),
    _cue(40.0, 55.0, NO_PUNCT),
]

UNPARSEABLE_SUPPORTING = {"text": "supporting segment with no usable timestamp"}


def _ext_window(start=20.0, end=55.0):
    return {
        "start_time": _srt(start),
        "end_time": _srt(end),
        "duration": end - start,
        "title": "Extended window title",
    }


def _ext_supporting():
    return [
        _cue(0.0, 8.0, "supporting context before the earlier cue"),
        _cue(12.0, 18.0, "supporting context between the cues"),
        _cue(25.0, 30.0, "supporting context after the new start"),
        dict(UNPARSEABLE_SUPPORTING),
    ]


def _extend(cue_index, window=None, supporting=None, max_duration=59.0):
    return reelsfy.extend_window_to_earlier_topic_start(
        cue_index,
        window if window is not None else _ext_window(),
        supporting if supporting is not None else _ext_supporting(),
        EXT_CUES,
        max_duration=max_duration,
    )


class TestExtendWindowToEarlierTopicStart:
    """The window start moves EARLIER to the reported cue; the end and the
    recency anchor never move."""

    def test_earlier_cue_moves_start_back_and_recomputes_duration(self):
        window = _ext_window()

        new_optimized, _ = _extend(2, window=window)

        assert new_optimized["start_time"] == _srt(10.0)
        assert new_optimized["duration"] == pytest.approx(45.0)

    def test_end_time_is_never_touched(self):
        window = _ext_window()

        new_optimized, _ = _extend(2, window=window)

        assert new_optimized["end_time"] == "00:00:55,000"
        assert new_optimized["end_time"] == window["end_time"]

    def test_never_extends_before_the_capture_start(self):
        new_optimized, _ = _extend(1, window=_ext_window())

        # Cue 1 is the first cue of the capture; the start lands on it
        # exactly and never goes negative / before 0.0 s.
        assert new_optimized["start_time"] == _srt(0.0)
        assert new_optimized["duration"] == pytest.approx(55.0)

    def test_cue_starting_before_the_capture_start_is_floored_at_zero(self):
        # The floor is enforced by the function, not by the data: a cue whose
        # start computes before 0.0 s (bad analyzer output) is clamped to the
        # capture start instead of producing a negative start timestamp.
        negative_cue = _cue(-8.0, 10.0, NO_PUNCT)
        cues = [negative_cue, EXT_CUES[2]]
        window = _ext_window(start=20.0, end=40.0)

        new_optimized, _ = reelsfy.extend_window_to_earlier_topic_start(
            1, window, [], cues
        )

        assert new_optimized["start_time"] == _srt(0.0)
        assert new_optimized["duration"] == pytest.approx(40.0)

    def test_earlier_cue_beyond_max_duration_is_clamped(self):
        # 80 s end - 0.0 s cue = 80 s, which is over the 59 s cap.
        new_optimized, _ = _extend(1, window=_ext_window(start=40.0, end=80.0))

        assert new_optimized["start_time"] == _srt(21.0)
        assert new_optimized["duration"] == pytest.approx(59.0)
        assert new_optimized["end_time"] == "00:01:20,000"

    def test_custom_max_duration_is_honored(self):
        new_optimized, _ = _extend(
            1, window=_ext_window(start=40.0, end=80.0), max_duration=50.0
        )

        assert new_optimized["start_time"] == _srt(30.0)
        assert new_optimized["duration"] == pytest.approx(50.0)

    def test_no_supporting_segment_is_dropped_by_the_extension(self):
        # The extension only GROWS the window (it moves the start earlier and
        # never touches the end), so every supporting segment that is still in
        # the list is inside the new, larger window and must survive — including
        # the ones that begin AFTER the new (earlier) start.
        supporting = _ext_supporting()

        _, new_supporting = _extend(2, supporting=supporting)

        assert new_supporting == supporting
        assert [seg.get("text") for seg in new_supporting] == [
            "supporting context before the earlier cue",
            "supporting context between the cues",
            "supporting context after the new start",
            "supporting segment with no usable timestamp",
        ]

    def test_unparseable_supporting_segment_is_kept(self):
        _, new_supporting = _extend(2)

        assert dict(UNPARSEABLE_SUPPORTING) in new_supporting

    def test_supporting_list_that_survived_the_trim_is_returned_intact(self):
        # Regression: the extension used to filter on
        # ``segment_seconds < new_start_seconds``, which is False for every
        # segment on the real call path (the trim already dropped everything
        # before the later start, so all remaining segments start at or after
        # it, and the new start is earlier than all of them). That emptied the
        # whole list. Trim to cue 3, then extend back to cue 1: whatever the
        # trim left behind must come back out unchanged, same length, same
        # contents.
        window = _ext_window()
        surviving = [
            _cue(20.0, 40.0, "context inside the trimmed window"),
            _cue(25.0, 30.0, "later context inside the trimmed window"),
        ]

        trimmed_window, trimmed_supporting = reelsfy.trim_window_to_topic_start(
            3, window, surviving, EXT_CUES
        )
        assert [seg.get("text") for seg in trimmed_supporting] == [
            "context inside the trimmed window",
            "later context inside the trimmed window",
        ]

        extended_window, extended_supporting = _extend(
            1, window=trimmed_window, supporting=trimmed_supporting
        )

        assert extended_window["start_time"] == _srt(0.0)
        assert len(extended_supporting) == 2
        assert extended_supporting == trimmed_supporting

    def test_no_earlier_cue_returns_the_window_unchanged(self):
        window = _ext_window()
        supporting = _ext_supporting()
        before = (dict(window), [dict(s) for s in supporting])

        new_optimized, new_supporting = _extend(None, window=window, supporting=supporting)

        assert new_optimized is window
        assert new_supporting is supporting
        assert (new_optimized, new_supporting) == before

    @pytest.mark.parametrize(
        "bad_cue", [None, "2", 2.0, True, False, 0, -1, 99], ids=repr
    )
    def test_unusable_cue_degrades_to_unchanged_window(self, bad_cue):
        window = _ext_window()
        supporting = _ext_supporting()

        new_optimized, new_supporting = _extend(
            bad_cue, window=window, supporting=supporting
        )

        assert new_optimized is window
        assert new_supporting is supporting
        assert new_optimized["start_time"] == _srt(20.0)
        assert new_optimized["duration"] == pytest.approx(35.0)

    @pytest.mark.parametrize("non_earlier_cue", [3, 4], ids=repr)
    def test_cue_at_or_after_the_window_start_leaves_it_unchanged(self, non_earlier_cue):
        window = _ext_window()
        supporting = _ext_supporting()

        new_optimized, new_supporting = _extend(
            non_earlier_cue, window=window, supporting=supporting
        )

        assert new_optimized is window
        assert new_supporting is supporting

    @pytest.mark.parametrize(
        "broken_field", ["start_time", "end_time"], ids=repr
    )
    def test_malformed_window_timestamp_leaves_window_unchanged(self, broken_field):
        window = _ext_window()
        window[broken_field] = "not-a-timestamp"
        supporting = _ext_supporting()

        new_optimized, new_supporting = _extend(
            2, window=window, supporting=supporting
        )

        assert new_optimized is window
        assert new_supporting is supporting

    def test_extension_never_raises(self):
        # Nonsense in, unchanged window out — no exception escapes.
        for bad_cue in (object(), [2], {"cue": 2}, float("nan")):
            new_optimized, _ = _extend(bad_cue)
            assert new_optimized["start_time"] == _srt(20.0)


# ---------------------------------------------------------------------------
# A4 wiring inside generate_viral
# ---------------------------------------------------------------------------

# The resolved-window section of the live prompt exactly as it was produced by
# the PRE-A4 code, captured from a worktree checked out at e3acda0 (the commit
# immediately before this story) by running this same `_run_generate_viral`
# helper with the reply the pre-A4 harness hard-codes: (2, None, None, []).
# A4 must reproduce it byte-for-byte when `earlier_start_cue` is None, which is
# what makes that path a no-op rather than merely "equivalent".
PRE_A4_RESOLVED_WINDOW_SECTION = (
    "RESOLVED TOPIC WINDOW:\n"
    "- Resolved start: 00:00:25,000\n"
    "- Resolved end: 00:01:00,000\n"
    "- The final topic begins at cue 2 of the numbered transcript cues above\n"
    "The returned segment MUST start at or after the resolved start time and end "
    "at or before the resolved end time. Prefer exactly the resolved boundaries; "
    "only a clear sentence boundary justifies a slight adjustment within them."
)


class TestExtendWindowWiring:
    """The extension runs on the primary path, after the trim and before the
    snap, and the snap floor is taken from the cue that owns the start."""

    def test_extension_runs_after_the_trim_and_before_the_snap(self):
        primary_marker = "Using Topic-Focused Content Generation"
        primary_start = REELSFY_SOURCE.index(primary_marker)
        trim_call = REELSFY_SOURCE.find("trim_window_to_topic_start(", primary_start)
        extend_call = REELSFY_SOURCE.find(
            "extend_window_to_earlier_topic_start(", primary_start
        )
        snap_call = REELSFY_SOURCE.find("snap_window_to_boundaries(", primary_start)
        assert 0 < trim_call < extend_call < snap_call

    def test_generate_viral_captures_topic_complete_and_uses_earlier_cue(self):
        extend_spy = MagicMock(wraps=reelsfy.extend_window_to_earlier_topic_start)

        result, _, resolve_mock = _run_generate_viral(
            _topic_analysis_result(),
            resolve_return=(2, False, 1, []),
            extra_patches=[
                patch.object(reelsfy, "extend_window_to_earlier_topic_start", extend_spy)
            ],
        )

        assert result, "generate_viral must return the merged JSON"
        resolve_mock.assert_called_once()
        assert extend_spy.call_count == 1
        # The extension is fed the earlier cue A3 reported, and A3's
        # topic_complete is now captured rather than discarded.
        assert extend_spy.call_args.args[0] == 1
        assert "topic_complete" in inspect.getsource(reelsfy.generate_viral)
        assert ", _topic_complete," not in inspect.getsource(reelsfy.generate_viral)

    def test_snap_floor_comes_from_the_earlier_cue_when_extended(self):
        snap_spy = MagicMock(wraps=reelsfy.snap_window_to_boundaries)

        _run_generate_viral(
            _topic_analysis_result(),
            resolve_return=(2, False, 1, []),
            extra_patches=[
                patch.object(reelsfy, "snap_window_to_boundaries", snap_spy)
            ],
        )

        # PRIMARY_CUES[0] (the earlier cue) starts at 0 s; PRIMARY_CUES[1]
        # (the trimmed cue) starts at 15 s. The floor must be the former,
        # otherwise the snap step trims the start forward again and silently
        # undoes the extension.
        assert snap_spy.call_args.kwargs["min_start_srt"] == _srt(0.0)
        assert snap_spy.call_args.kwargs["min_start_srt"] != _srt(15.0)

    def test_snap_floor_still_comes_from_the_trimmed_cue_without_extension(self):
        snap_spy = MagicMock(wraps=reelsfy.snap_window_to_boundaries)

        _run_generate_viral(
            _topic_analysis_result(),
            resolve_return=(2, None, None, []),
            extra_patches=[
                patch.object(reelsfy, "snap_window_to_boundaries", snap_spy)
            ],
        )

        assert snap_spy.call_args.kwargs["min_start_srt"] == _srt(15.0)

    def test_complete_topic_with_no_earlier_cue_is_handled_like_no_verdict_at_all(self):
        # A3's parser already yields earlier_start_cue=None when the topic is
        # complete, so both replies skip the extension. This pins that the
        # verdict itself changes nothing on the path — it is not a
        # byte-identity claim against the pre-A4 prompt (that is the golden
        # test below); it only fails if the wiring starts branching on
        # topic_complete.
        complete, complete_captured, _ = _run_generate_viral(
            _topic_analysis_result(), resolve_return=(2, True, None, [])
        )
        unknown, unknown_captured, _ = _run_generate_viral(
            _topic_analysis_result(), resolve_return=(2, None, None, [])
        )

        assert complete, "generate_viral must return the merged JSON"
        assert complete_captured["messages"] == unknown_captured["messages"]

    def test_no_earlier_cue_prompt_reproduces_the_pre_a4_prompt_byte_for_byte(self):
        # THE byte-identity claim: with earlier_start_cue=None the resolved
        # window section of the live prompt must be byte-for-byte what the
        # pre-A4 code (commit e3acda0, the commit immediately before this
        # story) produced for the same fixtures. The expected value below was
        # captured by running exactly this helper against a worktree checked
        # out at e3acda0 — the LLM reply (2, None, None, []) is hard-coded
        # there — so it is a golden, not a re-run of the current code.
        result, captured, _ = _run_generate_viral(
            _topic_analysis_result(), resolve_return=(2, None, None, [])
        )

        assert result, "generate_viral must return the merged JSON"
        user_content = captured["messages"][1]["content"]
        section_start = user_content.index("RESOLVED TOPIC WINDOW")
        section_end = user_content.index("\n\n\nBased on your analysis")
        assert user_content[section_start:section_end] == PRE_A4_RESOLVED_WINDOW_SECTION

    def test_extension_reaches_the_live_prompt(self):
        result, captured, _ = _run_generate_viral(
            _topic_analysis_result(), resolve_return=(2, False, 1, [])
        )

        assert result, "generate_viral must return the merged JSON"
        user_content = captured["messages"][1]["content"]
        # Trim lands the start on cue 2 (15 s); the extension pulls it back to
        # cue 1 (0 s) and the snapper cannot drag it forward again.
        assert "RESOLVED TOPIC WINDOW" in user_content
        assert _srt(0.0) in user_content
        assert "00:01:00,000" in user_content

    def test_cap_holds_through_extension_snap_and_prosodic_refinement(self, tmp_path):
        # The clamp can land the start exactly on `end - max_duration` (60 -
        # 59 = 1 s here). Every later step must not push it back before that
        # point: the snapper has no earlier terminator to move to, and the
        # prosodic refinement has a 0.2 s pause inside its ±1.5 s
        # neighborhood that would otherwise make the window 59.8 s.
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"stub-audio")
        topic_analysis = {
            "segments": [
                _cue(0.0, 20.0, NO_PUNCT),
                _cue(20.0, 45.0, NO_PUNCT),
                _cue(45.0, 60.0, NO_PUNCT),
            ],
            "concluding_topic": {
                "text": "The final topic text",
                "topic_keywords": ["final"],
                "topic_confidence": 0.9,
            },
            "supporting_segments": [],
            "optimized_segment": {
                "start_time": _srt(10.0),
                "end_time": _srt(60.0),
                "duration": 50.0,
                "title": "Final topic title",
            },
        }
        refine_spy = MagicMock(wraps=reelsfy.refine_window_start_with_pause)

        result, captured, _ = _run_generate_viral(
            topic_analysis,
            audio_path=str(audio_file),
            output_dir=str(tmp_path),
            analyzer_cls=_analyzer_cls([0.2]),
            resolve_return=(1, False, 1, []),
            extra_patches=[
                patch.object(reelsfy, "refine_window_start_with_pause", refine_spy)
            ],
        )

        assert result, "generate_viral must return the merged JSON"
        # The cap bound is handed to the prosodic step, and the resolved window
        # the prompt carries is the clamped 59 s one, not 59.8 s.
        assert refine_spy.call_args.kwargs["max_duration"] == 59.0
        user_content = captured["messages"][1]["content"]
        assert f"- Resolved start: {_srt(1.0)}\n" in user_content
        assert f"- Resolved end: {_srt(60.0)}\n" in user_content
