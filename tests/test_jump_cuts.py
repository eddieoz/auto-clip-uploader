"""
Story 5 (P1.3): Jump cuts — remove dead air from segments.

Covers:
1. build_jump_cut_plan - pure cut-plan builder:
   - no-pause passthrough (pieces == single full segment, words identical to
     today's collect_segment_words shift)
   - single pause cut with exact piece boundaries (end + padding / start - padding)
   - multiple pauses with cumulative remap
   - remap clamp >= 0
   - threshold not exceeded -> no cut (strictly greater than)
   - no words -> single piece + None remapped words
   - unsorted input sorted defensively
2. generate_segments wiring:
   - plan with >1 piece -> multi-piece ffmpeg command with per-piece trims,
     concat filter and the existing NVENC encode flags
   - no plan / single piece -> byte-identical legacy single-cut command
3. Env escape hatch:
   - JUMP_CUTS_ENABLED=false disables plan computation entirely
   - JUMP_CUT_PAUSE_THRESHOLD is parsed with a fallback warning
4. Caller wiring helpers:
   - compute_jump_cut_plans only stores plans for segments that were cut
   - resolve_subtitle_words returns remapped words when a plan exists and the
     plain collect_segment_words shift otherwise

All subprocess/ffmpeg usage is mocked - no real video is processed.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

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


from unittest.mock import MagicMock  # noqa: E402

for _name in (
    "pytube",
    "whisper",
    "audio_analyzer",
    "enhanced_audio_analyzer",
    "transcription_service",
):
    _ensure_stub(_name)

import reelsfy  # noqa: E402


SEG_START = 10.0
SEG_END = 40.0


def _clip_words(*timings):
    """Build clip-relative words (output of collect_segment_words).

    Each timing is a (start, end) tuple in clip-relative seconds.
    """
    return [
        {"word": f"w{i}", "start": float(s), "end": float(e)}
        for i, (s, e) in enumerate(timings)
    ]


class TestBuildJumpCutPlanNoPause:
    """No removable pause -> single piece + words identical to today."""

    def test_words_without_gap_passthrough(self):
        words = _clip_words((0.0, 2.0), (2.3, 4.0), (4.3, 6.0))

        pieces, remapped = reelsfy.build_jump_cut_plan(words, SEG_START, SEG_END)

        assert pieces == [(SEG_START, SEG_END)]
        # Zero removed duration: identical to the Story 4 shift output.
        assert remapped == words

    def test_gap_exactly_at_threshold_no_cut(self):
        # gap == pause_threshold (0.6) does NOT exceed it -> no cut
        words = _clip_words((0.0, 2.0), (2.6, 4.0))

        pieces, remapped = reelsfy.build_jump_cut_plan(words, SEG_START, SEG_END)

        assert pieces == [(SEG_START, SEG_END)]
        assert remapped == words

    def test_gap_below_threshold_no_cut(self):
        words = _clip_words((0.0, 2.0), (2.4, 4.0))

        pieces, remapped = reelsfy.build_jump_cut_plan(words, SEG_START, SEG_END)

        assert pieces == [(SEG_START, SEG_END)]
        assert remapped == words

    def test_leading_and_trailing_silence_kept(self):
        # Only intra-segment pauses are cut: leading material before the first
        # word's start and trailing material after the last word's end stay.
        words = _clip_words((2.0, 3.0), (3.3, 4.0))

        pieces, remapped = reelsfy.build_jump_cut_plan(words, SEG_START, SEG_END)

        assert pieces[0][0] == SEG_START
        assert pieces[-1][1] == SEG_END

    def test_no_words_single_piece_and_none_words(self):
        pieces, remapped = reelsfy.build_jump_cut_plan(None, SEG_START, SEG_END)

        assert pieces == [(SEG_START, SEG_END)]
        assert remapped is None

    def test_empty_words_single_piece_and_none_words(self):
        pieces, remapped = reelsfy.build_jump_cut_plan([], SEG_START, SEG_END)

        assert pieces == [(SEG_START, SEG_END)]
        assert remapped is None

    def test_unsorted_input_sorted_defensively(self):
        words = [
            {"word": "b", "start": 3.0, "end": 5.0},
            {"word": "a", "start": 0.0, "end": 2.0},
        ]

        pieces, remapped = reelsfy.build_jump_cut_plan(words, SEG_START, SEG_END)

        # The 1.0s gap between word a (end 2) and word b (start 3) is found
        # regardless of input order.
        assert len(pieces) == 2
        assert remapped[0]["word"] == "a"
        assert remapped[1]["word"] == "b"


class TestBuildJumpCutPlanSinglePause:
    """Single pause cut: exact piece boundaries and remap."""

    def test_piece_boundaries_exact(self):
        words = _clip_words((0.0, 2.0), (3.0, 5.0))
        # gap = 1.0 > 0.6 -> remove [2 + 0.12, 3 - 0.12] in clip time,
        # i.e. [12.12, 12.88] in source time.

        pieces, remapped = reelsfy.build_jump_cut_plan(words, SEG_START, SEG_END)

        assert len(pieces) == 2
        assert pieces[0] == (pytest.approx(10.0), pytest.approx(12.12))
        assert pieces[1] == (pytest.approx(12.88), pytest.approx(40.0))

    def test_word_remap_shifted_by_removed_duration(self):
        words = _clip_words((0.0, 2.0), (3.0, 5.0))

        pieces, remapped = reelsfy.build_jump_cut_plan(words, SEG_START, SEG_END)

        removed = pytest.approx(0.76)  # (3.0 - 0.12) - (2.0 + 0.12)
        assert remapped[0] == {"word": "w0", "start": pytest.approx(0.0), "end": pytest.approx(2.0)}
        assert remapped[1]["start"] == pytest.approx(3.0 - 0.76)
        assert remapped[1]["end"] == pytest.approx(5.0 - 0.76)
        assert remapped[1]["word"] == "w1"

    def test_cut_range_not_positive_no_cut(self):
        # Pause barely above threshold but padding eats the whole range:
        # cut range [2 + p, 2.65 - p] with p = 0.12 -> [2.12, 2.53] is fine;
        # force degeneracy with a large padding so the range is <= 0.
        words = _clip_words((0.0, 2.0), (2.65, 4.0))

        pieces, _ = reelsfy.build_jump_cut_plan(
            words, SEG_START, SEG_END, padding=0.4
        )

        # cut range would be [2.4, 2.25] -> invalid, no cut must be applied
        assert pieces == [(SEG_START, SEG_END)]

    def test_custom_threshold_and_padding(self):
        words = _clip_words((0.0, 2.0), (3.0, 5.0))

        pieces, _ = reelsfy.build_jump_cut_plan(
            words, SEG_START, SEG_END, pause_threshold=0.8, padding=0.3
        )

        assert pieces[0] == (pytest.approx(10.0), pytest.approx(12.3))
        assert pieces[1] == (pytest.approx(12.7), pytest.approx(40.0))


class TestBuildJumpCutPlanMultiplePauses:
    """Multiple pauses: cumulative remap across cuts."""

    def test_cumulative_remap(self):
        words = _clip_words((0.0, 2.0), (3.0, 5.0), (6.0, 7.0))
        # Two 1.0s pauses -> two 0.76s cuts.

        pieces, remapped = reelsfy.build_jump_cut_plan(words, SEG_START, SEG_END)

        assert len(pieces) == 3
        assert pieces[0] == (pytest.approx(10.0), pytest.approx(12.12))
        assert pieces[1] == (pytest.approx(12.88), pytest.approx(15.12))
        assert pieces[2] == (pytest.approx(15.88), pytest.approx(40.0))

        assert remapped[0]["start"] == pytest.approx(0.0)
        assert remapped[1]["start"] == pytest.approx(3.0 - 0.76)
        # Word 2 accumulates both cuts.
        assert remapped[2]["start"] == pytest.approx(6.0 - 1.52)
        assert remapped[2]["end"] == pytest.approx(7.0 - 1.52)

    def test_remapped_words_never_negative(self):
        # Degenerate overlap: force a cut whose removed duration exceeds the
        # following word's start; the remap must clamp to >= 0.
        words = _clip_words((0.0, 2.0), (3.0, 5.0))

        _, remapped = reelsfy.build_jump_cut_plan(
            words, SEG_START, SEG_END, padding=-1.0
        )

        for w in remapped:
            assert w["start"] >= 0.0
            assert w["end"] >= 0.0

    def test_one_piece_only_single_full_segment(self):
        # Even when words exist, zero removed duration means a single piece.
        words = _clip_words((0.0, 2.0), (2.2, 4.0))

        pieces, remapped = reelsfy.build_jump_cut_plan(words, SEG_START, SEG_END)

        assert pieces == [(SEG_START, SEG_END)]
        assert remapped == words


class TestGetJumpCutConfig:
    """Env escape hatch parsing."""

    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            enabled, threshold = reelsfy.get_jump_cut_config()

        assert enabled is True
        assert threshold == pytest.approx(0.6)

    def test_disabled_when_false(self):
        with patch.dict(os.environ, {"JUMP_CUTS_ENABLED": "false"}):
            enabled, _ = reelsfy.get_jump_cut_config()

        assert enabled is False

    def test_case_insensitive_false(self):
        with patch.dict(os.environ, {"JUMP_CUTS_ENABLED": "False"}):
            enabled, _ = reelsfy.get_jump_cut_config()

        assert enabled is False

    def test_invalid_threshold_falls_back(self, capsys):
        with patch.dict(os.environ, {"JUMP_CUT_PAUSE_THRESHOLD": "abc"}):
            _, threshold = reelsfy.get_jump_cut_config()

        assert threshold == pytest.approx(0.6)
        assert "JUMP_CUT_PAUSE_THRESHOLD" in capsys.readouterr().out

    def test_custom_threshold(self):
        with patch.dict(os.environ, {"JUMP_CUT_PAUSE_THRESHOLD": "1.5"}):
            _, threshold = reelsfy.get_jump_cut_config()

        assert threshold == pytest.approx(1.5)


class TestGenerateSegmentsAppliesPlan:
    """generate_segments applies the plan: multi-piece concat command."""

    def _segment(self):
        return {
            "start_time": "00:00:10,000",
            "end_time": "00:00:40,000",
            "title": "Test Segment",
        }

    def test_multi_piece_command_contains_concat_and_trims(self):
        plans = {0: ([(10.0, 12.0), (13.0, 40.0)], None)}

        with patch.object(reelsfy.subprocess, "call") as call_mock:
            reelsfy.generate_segments([self._segment()], jump_cut_plans=plans)

        command = call_mock.call_args[0][0]
        assert "concat" in command
        # One input per piece, each with its own trim.
        assert command.count("-i tmp/input_video.mp4") == 2
        assert command.count("-ss ") == 2
        # Existing NVENC encode settings are kept.
        assert "h264_nvenc" in command
        assert "-c:a aac -b:a 192k" in command

    def test_single_piece_plan_keeps_legacy_command(self):
        plans = {0: ([(10.0, 40.0)], None)}

        with patch.object(reelsfy.subprocess, "call") as call_mock:
            reelsfy.generate_segments([self._segment()], jump_cut_plans=plans)

        command = call_mock.call_args[0][0]
        # Same command shape as the no-plan single cut, now with the
        # Story 9 fast seek (-ss/-to before -i).
        assert command == (
            "ffmpeg -y -hwaccel cuda -ss 10.0 -to 40.0 -i tmp/input_video.mp4 "
            "-c:v h264_nvenc -profile:v high -rc:v vbr -qp 18 -b:v 10000k "
            "-maxrate:v 12000k -bufsize:v 15000k -c:a aac -b:a 192k "
            "tmp/Test_Segment000.mp4"
        )
        assert "concat" not in command

    def test_no_plan_legacy_command(self):
        with patch.object(reelsfy.subprocess, "call") as call_mock:
            reelsfy.generate_segments([self._segment()])

        command = call_mock.call_args[0][0]
        assert "concat" not in command
        assert "-ss 10.0 -to 40.0" in command

    def test_multi_piece_last_piece_uses_adjusted_end(self):
        # generate_segments extends a <30s segment's end_time; the trailing
        # piece must be extended to the adjusted end, preserving today's
        # short-segment behavior.
        segment = {
            "start_time": "00:00:10,000",
            "end_time": "00:00:15,000",
            "title": "Short Segment",
        }
        plans = {0: ([(10.0, 12.0), (13.0, 15.0)], None)}

        with patch.object(reelsfy.subprocess, "call") as call_mock:
            reelsfy.generate_segments([segment], jump_cut_plans=plans)

        command = call_mock.call_args[0][0]
        # 15 + (30 - 5) = 40 adjusted end lands in the final trim.
        assert "-to 40.0" in command


class TestCallerWiringHelpers:
    """compute_jump_cut_plans + resolve_subtitle_words wiring."""

    def _transcript_segments(self):
        # Master transcript in SOURCE time; a 1.0s pause between the words.
        return [
            {
                "words": [
                    {"word": "hello", "start": 10.0, "end": 12.0},
                    {"word": "world", "start": 13.0, "end": 15.0},
                ]
            }
        ]

    def _segment(self):
        return {"start_time": "00:00:10,000", "end_time": "00:00:40,000"}

    def test_compute_jump_cut_plans_stores_cut_segments(self):
        plans = reelsfy.compute_jump_cut_plans(
            [self._segment()], self._transcript_segments(), 0.6
        )

        assert 0 in plans
        pieces, remapped = plans[0]
        assert len(pieces) == 2
        assert remapped[0]["start"] == pytest.approx(0.0)
        assert remapped[1]["start"] == pytest.approx(2.24)

    def test_compute_jump_cut_plans_skips_uncut_segments(self):
        short_words = [
            {
                "words": [
                    {"word": "a", "start": 10.0, "end": 12.0},
                    {"word": "b", "start": 12.2, "end": 14.0},
                ]
            }
        ]

        plans = reelsfy.compute_jump_cut_plans([self._segment()], short_words, 0.6)

        assert plans == {}

    def test_compute_jump_cut_plans_skips_no_word_segments(self):
        plans = reelsfy.compute_jump_cut_plans([self._segment()], [{}], 0.6)

        assert plans == {}

    def test_compute_jump_cut_plans_empty_segments(self):
        assert reelsfy.compute_jump_cut_plans([], self._transcript_segments(), 0.6) == {}
        assert reelsfy.compute_jump_cut_plans(None, self._transcript_segments(), 0.6) == {}

    def test_resolve_subtitle_words_uses_remapped_words(self):
        remapped = [{"word": "hello", "start": 0.0, "end": 1.0}]
        plans = {0: ([(10.0, 12.0), (13.0, 40.0)], remapped)}

        words = reelsfy.resolve_subtitle_words(
            plans, 0, self._transcript_segments(), "00:00:10,000", "00:00:40,000"
        )

        assert words is remapped

    def test_resolve_subtitle_words_falls_back_to_collect(self):
        plans = {}

        words = reelsfy.resolve_subtitle_words(
            plans, 0, self._transcript_segments(), "00:00:10,000", "00:00:40,000"
        )

        assert words == [
            {"word": "hello", "start": pytest.approx(0.0), "end": pytest.approx(2.0)},
            {"word": "world", "start": pytest.approx(3.0), "end": pytest.approx(5.0)},
        ]

    def test_resolve_subtitle_words_none_plans(self):
        words = reelsfy.resolve_subtitle_words(
            None, 0, self._transcript_segments(), "00:00:10,000", "00:00:40,000"
        )

        assert words is not None
        assert words[0]["word"] == "hello"

    def test_remapped_words_reach_generate_subtitle(self):
        """End-to-end wiring: the plan's remapped words are what the caller
        passes to generate_subtitle (mocked)."""
        plans = reelsfy.compute_jump_cut_plans(
            [self._segment()], self._transcript_segments(), 0.6
        )

        with patch.object(reelsfy, "generate_subtitle") as sub_mock:
            subtitle_words = reelsfy.resolve_subtitle_words(
                plans, 0, self._transcript_segments(), "00:00:10,000", "00:00:40,000"
            )
            reelsfy.generate_subtitle(
                "final-output.mp4", "vid", "outdir", subtitle_words=subtitle_words
            )

        kwargs = sub_mock.call_args[1]
        assert kwargs["subtitle_words"][1]["start"] == pytest.approx(2.24)
