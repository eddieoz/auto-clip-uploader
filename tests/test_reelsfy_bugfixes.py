"""
Bug-fix tests for Story 1: quick bug fixes and pipeline hygiene in reelsfy.py
(plan section 4).

Covers six defects:
1. generate_short must return the final output path on success so
   direct_process_file logs success correctly.
2. The face-box locals (x, y, w, h) must be explicit state, so a failure to
   select a face falls back to the center-crop branch instead of crashing on
   leaked/unset loop variables.
3. No top-level `import whisper` (it loads torch eagerly); local imports kept.
4. download_video uses yt-dlp instead of the unmaintained pytube.
5. No cv2.waitKey polling in the generate_short frame loop (headless runs).
6. The "Interval ..." random-mode print is gated behind debug_faces.

All cv2/VideoCapture usage is mocked - no real video is processed.
"""

import ast
import os
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

import cv2  # noqa: E402  (real cv2 is installed; used only for constants)

import reelsfy  # noqa: E402


REELSFY_SOURCE = (REELSFY_DIR / "reelsfy.py").read_text()


def _make_detector(lookahead_result=None, single_result=None):
    """Build a mocked TalkingFaceDetector with a configured selection result."""
    detector = Mock()
    detector.static_frame_count = 20
    detector.get_best_talking_face_with_lookahead.return_value = lookahead_result
    detector.get_best_talking_face.return_value = single_result
    detector.get_face_scores.return_value = {
        "movement": 0.9,
        "quality": 0.9,
        "combined": 0.9,
    }
    return detector


def _success_pipe_proc():
    """Pipe ffmpeg process mock that starts fine and exits 0 (Story 9)."""
    proc = MagicMock()
    proc.returncode = 0
    proc.wait.return_value = 0
    return proc


def _run_generate_short(debug_faces=False, detector=None):
    """
    Run generate_short against mocked cv2 and subprocess.

    Simulates a 1920x1080 @ 30fps video that yields exactly one frame with one
    detected face, then ends. Returns (result, writer_mock, waitkey_mock).

    Story 9: the primary encode path pipes frames into one ffmpeg pass, so
    the returned `writer` is the mocked _PipeFrameWriter sink receiving the
    rendered frames.

    Returns a tuple so tests can assert on the return value, on frames written,
    and on waitKey polling without touching real video files.
    """
    detector = detector or _make_detector(lookahead_result=0, single_result=0)

    cap = MagicMock()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cap.isOpened.return_value = True
    # One real frame, then end of video (the detector look-ahead also reads)
    cap.read.side_effect = [(True, frame)] + [(False, None)] * 50
    cap.get.side_effect = lambda attr: {
        cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
        cv2.CAP_PROP_FPS: 30.0,
    }.get(attr, 0.0)

    cascade = MagicMock()
    # detectMultiScale returns integer boxes in production
    cascade.detectMultiScale.return_value = np.array(
        [[100, 100, 200, 200]], dtype=np.int32
    )

    writer = MagicMock()

    with patch.object(reelsfy.cv2, "VideoCapture", return_value=cap), patch.object(
        reelsfy.cv2, "VideoWriter", return_value=writer
    ), patch.object(reelsfy.cv2, "CascadeClassifier", return_value=cascade), patch.object(
        reelsfy.cv2, "waitKey"
    ) as waitkey_mock, patch.object(
        reelsfy.cv2, "destroyAllWindows"
    ), patch.object(
        reelsfy.subprocess, "Popen", return_value=_success_pipe_proc()
    ), patch.object(
        reelsfy, "_PipeFrameWriter", return_value=writer
    ), patch.object(
        reelsfy.subprocess, "call"
    ), patch.object(
        reelsfy, "TalkingFaceDetector", return_value=detector
    ):
        result = reelsfy.generate_short(
            "input.mp4",
            "output_reel.mp4",
            title="test",
            debug_faces=debug_faces,
        )

    return result, writer, waitkey_mock


class TestGenerateShortReturnsPath:
    """Fix 1: generate_short must return the final output path on success."""

    def test_generate_short_returns_output_path_on_success(self):
        result, writer, _ = _run_generate_short()

        assert result, "generate_short must return a truthy path on success"
        assert result == "tmp/final-output_reel.mp4"

    def test_generate_short_returns_none_on_error(self):
        # Force an error inside the loop: face selection raises, hitting the
        # outer exception handler.
        detector = _make_detector(lookahead_result=0, single_result=0)
        detector.get_best_talking_face.side_effect = RuntimeError(
            "detector exploded"
        )

        result, _, _ = _run_generate_short(detector=detector)

        assert result is None

    def test_direct_process_file_logs_success(self, tmp_path, capsys):
        with patch.object(
            reelsfy, "generate_short", return_value="tmp/final-x.mp4"
        ):
            reelsfy.direct_process_file(
                str(tmp_path / "input.mp4"), str(tmp_path / "out")
            )

        captured = capsys.readouterr()
        assert "Successfully created reel at:" in captured.out
        assert "Error: Failed to create reel" not in captured.out

    def test_direct_process_file_logs_failure(self, tmp_path, capsys):
        with patch.object(reelsfy, "generate_short", return_value=None):
            reelsfy.direct_process_file(
                str(tmp_path / "input.mp4"), str(tmp_path / "out")
            )

        captured = capsys.readouterr()
        assert "Error: Failed to create reel" in captured.out


class TestStaleFaceBoxGuard:
    """
    Fix 2: face-box locals must be explicit state.

    Scenario: faces are detected, but the talking-face detector finds no
    suitable face while a previous face index exists. The old code kept
    face_positions non-empty and then crashed on uninitialized x/y/w/h
    ("Error during video cropping") instead of falling back to the
    center-crop branch.
    """

    def test_no_face_selected_falls_back_to_center_crop(self, capsys):
        detector = _make_detector(lookahead_result=None, single_result=None)

        result, writer, _ = _run_generate_short(detector=detector)

        captured = capsys.readouterr()
        assert result, "center-crop fallback must still produce a reel"
        assert "Error during video cropping" not in captured.out
        # One frame was written through the center-crop branch
        assert writer.write.call_count == 1


class TestNoTopLevelWhisperImport:
    """Fix 3: no top-level `import whisper`; local imports are kept."""

    def test_no_top_level_whisper_import(self):
        tree = ast.parse(REELSFY_SOURCE)

        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                assert "whisper" not in names, (
                    "whisper must not be imported at module top level"
                )
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "whisper", (
                    "whisper must not be imported at module top level"
                )

    def test_local_whisper_imports_are_kept(self):
        assert REELSFY_SOURCE.count("import whisper") >= 2, (
            "generate_transcript and generate_subtitle keep their local imports"
        )


class TestDownloadVideoUsesYtDlp:
    """Fix 4: download_video uses the yt_dlp Python API, not pytube."""

    def test_uses_ytdlp_with_tmp_outtmpl(self, tmp_path):
        with patch.object(reelsfy, "yt_dlp") as mock_ytdlp:
            reelsfy.download_video(
                "https://www.youtube.com/watch?v=abc", "input_video.mp4", str(tmp_path)
            )

        opts = mock_ytdlp.YoutubeDL.call_args[0][0]
        assert opts["outtmpl"] == os.path.join(str(tmp_path), "input_video.mp4")
        assert "mp4" in opts["format"]

        ydl = mock_ytdlp.YoutubeDL.return_value.__enter__.return_value
        ydl.download.assert_called_once_with(["https://www.youtube.com/watch?v=abc"])

    def test_tmp_dir_defaults_to_tmp(self):
        """The __main__ caller passes only (url, filename); tmp/ is implied."""
        with patch.object(reelsfy, "yt_dlp") as mock_ytdlp:
            reelsfy.download_video(
                "https://www.youtube.com/watch?v=abc", "input_video.mp4"
            )

        opts = mock_ytdlp.YoutubeDL.call_args[0][0]
        assert opts["outtmpl"] == os.path.join("tmp", "input_video.mp4")

    def test_pytube_no_longer_imported(self):
        tree = ast.parse(REELSFY_SOURCE)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert "pytube" not in [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "pytube"


class TestNoWaitKeyInFrameLoop:
    """Fix 5: cv2.waitKey polling is meaningless headless and must be gone."""

    def test_waitkey_not_called(self):
        _, _, waitkey_mock = _run_generate_short()

        waitkey_mock.assert_not_called()


class TestIntervalPrintGatedBehindDebugFaces:
    """Fix 6: the Interval/tracking-mode print only runs with debug_faces."""

    def test_interval_print_hidden_when_debug_faces_false(self, capsys):
        _run_generate_short(debug_faces=False)

        captured = capsys.readouterr()
        assert "tracking_mode=" not in captured.out

    def test_interval_print_shown_when_debug_faces_true(self, capsys):
        _run_generate_short(debug_faces=True)

        captured = capsys.readouterr()
        assert "tracking_mode=" in captured.out


class TestCollectSegmentWordsToleratesMalformedWords:
    """A malformed word entry must degrade to the documented fallback
    (re-transcription) instead of killing the run with KeyError/TypeError."""

    def test_missing_start_end_keys_returns_none(self):
        segments = [{"words": [{"word": "hello"}]}]

        assert reelsfy.collect_segment_words(segments, "00:00:00,000", "00:00:10,000") is None

    def test_non_numeric_start_end_values_returns_none(self):
        segments = [{"words": [{"word": "hello", "start": None, "end": 1.0}]}]

        assert reelsfy.collect_segment_words(segments, "00:00:00,000", "00:00:10,000") is None

    def test_malformed_entries_skipped_valid_words_kept(self):
        segments = [
            {
                "words": [
                    {"word": "bad", "start": "oops"},
                    {"word": "good", "start": 1.0, "end": 2.0},
                ]
            }
        ]

        words = reelsfy.collect_segment_words(segments, "00:00:00,000", "00:00:10,000")

        assert words is not None
        assert [w["word"] for w in words] == ["good"]
        assert words[0]["start"] == 1.0
        assert words[0]["end"] == 2.0
