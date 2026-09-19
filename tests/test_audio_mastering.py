"""
Audio mastering tests for Story 2 (P1.1): loudness normalization + gentle
compression in reelsfy.py.

Covers:
1. The mastering filter chain has the exact brief parameters
   (loudnorm -14 LUFS -> acompressor) exposed as a module-level constant.
2. The final mux in generate_subtitle applies the chain and re-encodes to
   AAC (-c:a aac -b:a 192k) instead of -c:a copy - for both the with-subtitles
   and the no-subtitles/overlays-only variants - while video encoding is
   untouched.
3. Single-application invariant: the intermediate ffmpeg commands in
   generate_segments and generate_short's final merge do NOT contain the
   mastering chain (audio passes through those stages untouched).

All ffmpeg/subprocess calls are mocked - no real video or audio is processed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
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

# transcription_service may pull heavy deps; always stub it so the local
# `from transcription_service import get_transcriber` inside
# generate_subtitle resolves to a controllable mock.
transcription_service_stub = MagicMock()
sys.modules["transcription_service"] = transcription_service_stub

import cv2  # noqa: E402  (real cv2 is installed; used only for constants)

import reelsfy  # noqa: E402


EXPECTED_CHAIN = (
    "loudnorm=I=-14:LRA=11:tp=-1.5,acompressor=threshold=0.05:ratio=3:makeup=1.2"
)

SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": "Hello world"},
    {"start": 2.5, "end": 5.0, "text": "Second line"},
]


def _run_generate_subtitle(tmp_path, monkeypatch, segments):
    """
    Run generate_subtitle in an isolated tmp cwd with mocked transcription
    and subprocess. Returns (result, subprocess_call_mock).
    """
    workdir = tmp_path / "work"
    (workdir / "tmp").mkdir(parents=True)
    output_dir = workdir / "output"
    output_dir.mkdir()
    monkeypatch.chdir(workdir)

    # Fake extracted-audio file so the transcription step proceeds
    audio_path = workdir / "tmp" / "subtitle_audio_input.mp4.wav"
    audio_path.write_bytes(b"\x00" * 64)

    transcriber = MagicMock()
    transcriber.name = "stub-transcriber"
    transcriber.transcribe.return_value = segments

    with patch.object(
        transcription_service_stub, "get_transcriber", return_value=transcriber
    ), patch.object(reelsfy.subprocess, "call", return_value=0) as call_mock:
        result = reelsfy.generate_subtitle("input.mp4", "vid", str(output_dir))

    return result, call_mock


def _commands_called(call_mock):
    """Extract all shell command strings passed to subprocess.call."""
    return [args[0][0] for args in call_mock.call_args_list]


class TestFilterChainDefinition:
    """Requirement 3: testable construction of the exact filter parameters."""

    def test_constant_has_exact_parameters(self):
        assert reelsfy.AUDIO_MASTERING_FILTER == EXPECTED_CHAIN

    def test_build_audio_filter_args_contains_chain_and_aac(self):
        args = reelsfy.build_audio_filter_args()

        assert f'-af "{EXPECTED_CHAIN}"' in args
        assert "-c:a aac" in args
        assert "-b:a 192k" in args

    def test_build_audio_filter_args_drops_audio_copy(self):
        args = reelsfy.build_audio_filter_args()

        assert "-c:a copy" not in args


class TestFinalMuxMastersAudio:
    """Requirement 1: the final mux re-encodes mastered audio to AAC."""

    def test_with_subtitles_command_contains_chain(self, tmp_path, monkeypatch):
        result, call_mock = _run_generate_subtitle(
            tmp_path, monkeypatch, segments=SAMPLE_SEGMENTS
        )

        assert result is True
        final_cmd = _commands_called(call_mock)[-1]
        assert EXPECTED_CHAIN in final_cmd
        assert "-c:a aac" in final_cmd
        assert "-b:a 192k" in final_cmd

    def test_with_subtitles_no_audio_copy_anymore(self, tmp_path, monkeypatch):
        _, call_mock = _run_generate_subtitle(
            tmp_path, monkeypatch, segments=SAMPLE_SEGMENTS
        )

        final_cmd = _commands_called(call_mock)[-1]
        assert "-c:a copy" not in final_cmd

    def test_with_subtitles_video_encoding_untouched(self, tmp_path, monkeypatch):
        _, call_mock = _run_generate_subtitle(
            tmp_path, monkeypatch, segments=SAMPLE_SEGMENTS
        )

        final_cmd = _commands_called(call_mock)[-1]
        assert "-c:v h264_nvenc" in final_cmd
        assert "-qp 18" in final_cmd
        assert "-map 0:a" in final_cmd

    def test_no_subtitles_command_contains_chain(self, tmp_path, monkeypatch):
        # Empty transcriber result -> no-subtitles / overlays-only variant
        result, call_mock = _run_generate_subtitle(
            tmp_path, monkeypatch, segments=[]
        )

        assert result is True
        final_cmd = _commands_called(call_mock)[-1]
        assert EXPECTED_CHAIN in final_cmd
        assert "-c:a aac" in final_cmd
        assert "-b:a 192k" in final_cmd
        assert "subtitles=" not in final_cmd

    def test_no_subtitles_no_audio_copy_anymore(self, tmp_path, monkeypatch):
        _, call_mock = _run_generate_subtitle(
            tmp_path, monkeypatch, segments=[]
        )

        final_cmd = _commands_called(call_mock)[-1]
        assert "-c:a copy" not in final_cmd


class TestSingleApplicationInvariant:
    """Requirement 2: mastering happens ONLY in generate_subtitle's final mux."""

    def _generate_short_commands(self):
        """Run generate_short with mocked cv2/subprocess; return ffmpeg cmds."""
        detector = Mock()
        detector.static_frame_count = 20
        detector.get_best_talking_face_with_lookahead.return_value = 0
        detector.get_best_talking_face.return_value = 0
        detector.get_face_scores.return_value = {
            "movement": 0.9,
            "quality": 0.9,
            "combined": 0.9,
        }

        cap = MagicMock()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, frame)] + [(False, None)] * 50
        cap.get.side_effect = lambda attr: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_FPS: 30.0,
        }.get(attr, 0.0)

        cascade = MagicMock()
        cascade.detectMultiScale.return_value = np.array(
            [[100, 100, 200, 200]], dtype=np.int32
        )

        with patch.object(reelsfy.cv2, "VideoCapture", return_value=cap), patch.object(
            reelsfy.cv2, "VideoWriter"
        ), patch.object(reelsfy.cv2, "CascadeClassifier", return_value=cascade), patch.object(
            reelsfy.cv2, "waitKey"
        ), patch.object(
            reelsfy.cv2, "destroyAllWindows"
        ), patch.object(
            reelsfy.subprocess, "call"
        ) as call_mock, patch.object(
            reelsfy, "TalkingFaceDetector", return_value=detector
        ):
            reelsfy.generate_short("input.mp4", "output_reel.mp4", title="test")

        return _commands_called(call_mock)

    def test_generate_short_merge_has_no_loudnorm(self):
        commands = self._generate_short_commands()

        merge_commands = [cmd for cmd in commands if "final-" in cmd]
        assert merge_commands, "generate_short must issue its final merge command"
        for cmd in merge_commands:
            assert "loudnorm" not in cmd, (
                "generate_short merge must not master audio"
            )

    def test_generate_short_merge_keeps_audio_copy(self):
        commands = self._generate_short_commands()

        merge_commands = [cmd for cmd in commands if "final-" in cmd]
        for cmd in merge_commands:
            assert "-c:a copy" in cmd

    def test_generate_segments_has_no_loudnorm(self):
        segments = [
            {
                "start_time": "00:00:05,000",
                "end_time": "00:00:45,000",
                "title": "Segment One",
            }
        ]

        with patch.object(reelsfy.subprocess, "call") as call_mock:
            reelsfy.generate_segments(segments)

        for cmd in _commands_called(call_mock):
            assert "loudnorm" not in cmd, (
                "generate_segments must not master audio"
            )
