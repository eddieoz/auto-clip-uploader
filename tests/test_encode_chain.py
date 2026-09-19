"""
Story 9 (P2.4): encode chain fixes in reelsfy.py.

1. Fast seek: generate_segments places -ss/-to BEFORE each -i (single-piece
   command and every per-piece input of the multi-piece builder), so ffmpeg
   no longer decodes the entire source per cut.
2. Single encode: generate_short pipes rendered BGR frames into ONE ffmpeg
   pass (NVENC encode, audio muxed from the segment input via -map 1:a,
   output fps normalized to 30). The legacy mp4v VideoWriter + audio-merge
   chain survives only as the fallback for when the pipe process cannot
   start or exits nonzero.

All cv2/VideoCapture/subprocess usage is mocked - no real video is
processed. Stub pattern follows tests/test_reelsfy_bugfixes.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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

import cv2  # noqa: E402  (real cv2 is installed; used only for constants)

import reelsfy  # noqa: E402


def _make_detector():
    """Mocked TalkingFaceDetector that always selects face 0."""
    detector = MagicMock()
    detector.static_frame_count = 20
    detector.get_best_talking_face_with_lookahead.return_value = 0
    detector.get_best_talking_face.return_value = 0
    detector.get_face_scores.return_value = {
        "movement": 0.9,
        "quality": 0.9,
        "combined": 0.9,
    }
    return detector


def _success_proc():
    """A pipe ffmpeg process mock that starts fine and exits 0."""
    proc = MagicMock()
    proc.returncode = 0
    proc.wait.return_value = 0
    return proc


def _make_cap(frame):
    """Fresh VideoCapture mock yielding one 1080p frame with a face."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.side_effect = [(True, frame.copy())] + [(False, None)] * 60
    cap.get.side_effect = lambda attr: {
        cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
        cv2.CAP_PROP_FPS: 30.0,
    }.get(attr, 0.0)
    return cap


@pytest.fixture
def short_env():
    """
    Run generate_short with every collaborator mocked.

    Returns a dict of mocks: popen (subprocess.Popen), proc (pipe process),
    call (subprocess.call), writer_cls (cv2.VideoWriter), writer (its
    return value). Each generate_short invocation gets a fresh capture so
    fallback re-renders see unread frames.
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detector = _make_detector()
    cascade = MagicMock()
    cascade.detectMultiScale.return_value = np.array(
        [[100, 100, 200, 200]], dtype=np.int32
    )
    writer = MagicMock()
    proc = _success_proc()
    env = {}

    def capture_side_effect(*args, **kwargs):
        return _make_cap(frame)

    with patch.object(
        reelsfy.cv2, "VideoCapture", side_effect=capture_side_effect
    ), patch.object(
        reelsfy.cv2, "VideoWriter", return_value=writer
    ) as writer_cls, patch.object(
        reelsfy.cv2, "CascadeClassifier", return_value=cascade
    ), patch.object(
        reelsfy.cv2, "destroyAllWindows"
    ), patch.object(
        reelsfy.subprocess, "Popen", return_value=proc
    ) as popen_mock, patch.object(
        reelsfy.subprocess, "call"
    ) as call_mock, patch.object(
        reelsfy, "TalkingFaceDetector", return_value=detector
    ):
        env.update(
            {
                "popen": popen_mock,
                "proc": proc,
                "call": call_mock,
                "writer_cls": writer_cls,
                "writer": writer,
                "run": lambda **kwargs: reelsfy.generate_short(
                    "input.mp4", "output_reel.mp4", title="test", **kwargs
                ),
            }
        )
        yield env


def _pipe_command_of(env):
    assert env["popen"].call_args is not None, (
        "generate_short must start the pipe ffmpeg process"
    )
    return env["popen"].call_args[0][0]


def _legacy_commands_of(env):
    return [args[0][0] for args in env["call"].call_args_list]


class TestSegmentsFastSeek:
    """Requirement 1: -ss/-to BEFORE -i in every generate_segments cut."""

    def _segment(self):
        return {
            "start_time": "00:00:10,000",
            "end_time": "00:00:40,000",
            "title": "Test Segment",
        }

    def test_single_piece_command_seeks_before_input(self):
        with patch.object(reelsfy.subprocess, "call") as call_mock:
            reelsfy.generate_segments([self._segment()])

        command = call_mock.call_args[0][0]
        assert "-ss 10.0 -to 40.0 -i tmp/input_video.mp4" in command
        assert command.index("-ss 10.0") < command.index("-i tmp/input_video.mp4")
        # NVENC output settings unchanged.
        assert "h264_nvenc" in command
        assert "-c:a aac -b:a 192k" in command

    def test_single_piece_builder_is_fast_seek(self):
        command = reelsfy.build_single_piece_command(10.0, 40.0, "Test_Segment000.mp4")
        assert "-ss 10.0 -to 40.0 -i tmp/input_video.mp4" in command
        # The old shape trimmed AFTER the output settings; that must be gone.
        assert not command.rstrip().endswith("-to 40.0")

    def test_multi_piece_every_input_seeks_before_input(self):
        plans = {0: ([(10.0, 12.0), (13.0, 40.0)], None)}

        with patch.object(reelsfy.subprocess, "call") as call_mock:
            reelsfy.generate_segments([self._segment()], jump_cut_plans=plans)

        command = call_mock.call_args[0][0]
        assert command.count("-i tmp/input_video.mp4") == 2
        # Each "-ss ... -to ... -i ..." substring guarantees the trim
        # options precede that piece's input flag.
        assert "-ss 10.0 -to 12.0 -i tmp/input_video.mp4" in command
        assert "-ss 13.0 -to 40.0 -i tmp/input_video.mp4" in command


class TestPipeCommand:
    """Requirement 2: the single-encode pipe command shape."""

    def test_pipe_command_flags(self):
        command = reelsfy.build_pipe_ffmpeg_command(
            "input.mp4", "final-output.mp4", 25.0
        )

        for flag in (
            "-f rawvideo",
            "-pix_fmt bgr24",
            "-s 1080x1920",
            "-r 30",
            "-map 0:v",
            "-map 1:a",
            "h264_nvenc",
            "-profile:v high",
            "-rc:v vbr",
            "-qp 18",
            "-b:v 10000k",
            "-maxrate:v 12000k",
            "-bufsize:v 15000k",
            "-pix_fmt yuv420p",
            "-c:a aac -b:a 192k",
        ):
            assert flag in command, f"missing {flag} in pipe command"

        assert "-r 25.0 -i pipe:0" in command
        assert command.strip().endswith("tmp/final-output.mp4")
        assert "mp4v" not in command
        assert "output-audio" not in command


class TestGenerateShortSingleEncode:
    """Requirement 2: one ffmpeg pass; no mp4v intermediate, no merge step."""

    def test_primary_path_pipes_frames_and_skips_merge(self, short_env):
        result = short_env["run"]()

        pipe_command = _pipe_command_of(short_env)
        assert "-f rawvideo" in pipe_command
        assert "-pix_fmt bgr24" in pipe_command
        assert "-s 1080x1920" in pipe_command
        assert "-r 30" in pipe_command
        assert "-map 1:a" in pipe_command
        assert "h264_nvenc" in pipe_command
        assert pipe_command.strip().endswith("tmp/final-output_reel.mp4")

        # No legacy merge/extraction commands in the primary path.
        assert _legacy_commands_of(short_env) == []
        # No mp4v VideoWriter either.
        short_env["writer_cls"].assert_not_called()

        # Rendered BGR frames are piped as raw bytes, stdin is closed.
        assert short_env["proc"].stdin.write.called
        for call in short_env["proc"].stdin.write.call_args_list:
            blob = call.args[0]
            assert isinstance(blob, bytes)
            assert len(blob) == 1080 * 1920 * 3
        assert short_env["proc"].stdin.close.called

        assert result == "tmp/final-output_reel.mp4"

    def test_pipe_process_receives_bgr_frames_through_frame_writer(self, short_env):
        # Drive the writer class directly: one write abstraction for BOTH
        # render paths (screencast continue-branch and talking head).
        proc = _success_proc()
        writer = reelsfy._PipeFrameWriter(proc)
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

        writer.write(frame)
        writer.release()

        blob = proc.stdin.write.call_args[0][0]
        assert isinstance(blob, bytes)
        assert len(blob) == 1920 * 1080 * 3
        proc.stdin.close.assert_called_once()
        proc.wait.assert_called_once()


class TestGenerateShortFallback:
    """Requirement 2: broken pipe falls back to the legacy encode chain."""

    def test_popen_failure_uses_legacy_writer_and_merge(self, short_env):
        short_env["popen"].side_effect = Exception("no ffmpeg available")

        result = short_env["run"]()

        # Legacy mp4v writer took over the frame loop.
        short_env["writer_cls"].assert_called_once()
        fourcc = short_env["writer_cls"].call_args[0][1]
        assert fourcc == cv2.VideoWriter_fourcc(*"mp4v")
        assert short_env["writer"].write.call_count == 1

        commands = _legacy_commands_of(short_env)
        extraction = [c for c in commands if "output-audio.aac" in c]
        merges = [c for c in commands if "final-" in c]
        assert extraction, "legacy audio extraction command must run"
        assert merges, "legacy merge command must run"
        assert "-c:v h264_nvenc" in merges[0]
        assert "-c:a copy" in merges[0]
        assert merges[0].strip().endswith("tmp/final-output_reel.mp4")

        assert result == "tmp/final-output_reel.mp4"

    def test_pipe_nonzero_exit_falls_back_to_legacy(self, short_env):
        short_env["proc"].returncode = 1

        result = short_env["run"]()

        # The pipe was attempted exactly once; the fallback re-renders
        # through the legacy writer instead of piping again.
        assert short_env["popen"].call_count == 1
        short_env["writer_cls"].assert_called_once()

        commands = _legacy_commands_of(short_env)
        merges = [c for c in commands if "final-" in c]
        assert merges, "legacy merge command must run after pipe failure"
        assert merges[0].strip().endswith("tmp/final-output_reel.mp4")

        assert result == "tmp/final-output_reel.mp4"


class TestReturnValueContract:
    """Story 1 contract intact: success returns tmp/final-{output}."""

    def test_success_returns_final_path(self, short_env):
        assert short_env["run"]() == "tmp/final-output_reel.mp4"

    def test_fallback_returns_final_path(self, short_env):
        short_env["popen"].side_effect = Exception("no ffmpeg available")
        assert short_env["run"]() == "tmp/final-output_reel.mp4"
