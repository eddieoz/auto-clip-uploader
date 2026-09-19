"""
Polish-final tests for Story 11 (P3.4-P3.5) in reelsfy.py.

Covers:
1. compute_fallback_crop pure helper: with no face history it reproduces the
   existing centered-crop math exactly; with a last-known face center it
   positions the largest 9:16 window on that point, clamped to frame bounds.
2. Wiring: the center-crop fallback branch uses the last known face center
   when the clip HAS seen a face earlier, and keeps today's behavior when it
   has not. Screencast rendering (which has its own badge/zoom handling)
   never reaches the fallback crop.
3. Thumbnail: hook text is preferred over the title; smile candidates are
   collected across the clip (up to 5, no first-smile early exit) and the
   highest FaceQualityScorer score wins; scoring failure falls back to the
   FIRST candidate; no candidates -> no thumbnail.

All cv2/VideoCapture/cascade usage is mocked - no real video is processed.
Follows the sys.modules stub pattern from tests/test_reelsfy_bugfixes.py.
"""

import ast
import sys
from contextlib import ExitStack
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

# transcription_service is a local import inside generate_subtitle; stub it so
# importing reelsfy stays light.
sys.modules.setdefault("transcription_service", MagicMock())

import cv2  # noqa: E402  (real cv2 is installed; used only for constants)

import reelsfy  # noqa: E402


REELSFY_SOURCE = (REELSFY_DIR / "reelsfy.py").read_text()

# Talking-head-classification face box (centered, larger): center (1060, 600)
TH_FACE = (860, 400, 400, 400)


# ---------------------------------------------------------------------------
# Synthetic frame helpers
# ---------------------------------------------------------------------------

def make_face_frame():
    """1920x1080 frame with a bright 'face' block where TH_FACE sits."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    fx, fy, fw, fh = TH_FACE
    frame[fy : fy + fh, fx : fx + fw] = 200
    return frame


def make_plain_frame():
    """Uniform dark frame: no face content, no edges."""
    return np.full((1080, 1920, 3), 30, dtype=np.uint8)


# Face-content frames carrying a "how many faces" marker in brightness:
# 255 -> three face boxes, 200 -> one face box, none -> no faces.
def make_multi_face_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[400:800, 360:760] = 255
    frame[400:800, 860:1260] = 255
    frame[400:800, 1360:1760] = 255
    return frame


MULTI_FACES = [(360, 400, 400, 400), (860, 400, 400, 400), (1360, 400, 400, 400)]
MULTI_FACE_CENTER = (1560, 600)  # center of the third (selected) box


def _content_based_boxes(gray, **_kwargs):
    """detectMultiScale side effect: face boxes keyed on frame brightness."""
    if not gray.size:
        return np.empty((0, 4), dtype=np.int32)
    value = int(gray[600, 1060])
    if value > 200:
        return np.array([list(b) for b in MULTI_FACES], dtype=np.int32)
    if value > 100:
        return np.array([list(TH_FACE)], dtype=np.int32)
    return np.empty((0, 4), dtype=np.int32)


def _make_detector(lookahead_result=0, single_result=0):
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


def _run_generate_short(
    frames,
    fps=1.0,
    thumb=False,
    thumb_text=None,
    thumb_dir="",
    debug_faces=False,
    detector=None,
    extra_patches=None,
):
    """
    Run generate_short against mocked cv2 and subprocess.

    `frames` is the list of BGR frames the frame loop reads (the layout
    sampler reads copies via grab/retrieve). Face detection is keyed on
    frame content, so a sequence of face frames followed by plain frames
    exercises "face seen earlier, then lost".

    Returns (result, writer, cascade_detect_spy).
    """
    detector = detector or _make_detector(lookahead_result=0, single_result=0)

    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.side_effect = [(True, f.copy()) for f in frames] + [(False, None)] * 60
    cap.grab.return_value = True
    cap.retrieve.return_value = (True, frames[0].copy())
    cap.get.side_effect = lambda attr: {
        cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_COUNT: float(len(frames)),
    }.get(attr, 0.0)

    cascade = MagicMock()
    cascade.detectMultiScale.side_effect = _content_based_boxes

    writer = MagicMock()

    patches = [
        patch.object(reelsfy.cv2, "VideoCapture", return_value=cap),
        patch.object(reelsfy.cv2, "VideoWriter", return_value=writer),
        patch.object(reelsfy.cv2, "CascadeClassifier", return_value=cascade),
        patch.object(reelsfy.cv2, "destroyAllWindows"),
        patch.object(reelsfy.subprocess, "Popen", return_value=_success_pipe_proc()),
        patch.object(reelsfy, "_PipeFrameWriter", return_value=writer),
        patch.object(reelsfy.subprocess, "call"),
        patch.object(reelsfy, "TalkingFaceDetector", return_value=detector),
    ]
    patches.extend(extra_patches or [])

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = reelsfy.generate_short(
            "input.mp4",
            "output_reel.mp4",
            title="test",
            thumb=thumb,
            thumb_dir=thumb_dir,
            thumb_text=thumb_text,
            debug_faces=debug_faces,
        )

    return result, writer, cascade


def _fallback_spy():
    """Spy wrapping the real compute_fallback_crop."""
    return patch.object(
        reelsfy, "compute_fallback_crop", wraps=reelsfy.compute_fallback_crop
    )


# ---------------------------------------------------------------------------
# 1. compute_fallback_crop pure helper
# ---------------------------------------------------------------------------


class TestComputeFallbackCrop:
    """P3.4: pure fallback-crop math."""

    def test_no_history_matches_current_centered_math_1920x1080(self):
        # Current inline math: max_h = 1920*16/9 overflows -> clamp to 1080,
        # max_w = 1080*9/16 = 607. Centered: x=(1920-607)//2, y=0.
        assert reelsfy.compute_fallback_crop(1920, 1080, None) == (656, 0, 607, 1080)

    def test_no_history_matches_current_centered_math_1280x720(self):
        assert reelsfy.compute_fallback_crop(1280, 720, None) == (437, 0, 405, 720)

    def test_no_history_matches_current_centered_math_portrait(self):
        # Portrait 1080x1920: the full frame already is 9:16.
        assert reelsfy.compute_fallback_crop(1080, 1920, None) == (0, 0, 1080, 1920)

    def test_off_center_face_window_contains_point_within_bounds(self):
        x, y, w, h = reelsfy.compute_fallback_crop(1920, 1080, (1500, 540))
        assert (x, y, w, h) == (1197, 0, 607, 1080)
        assert 0 <= x and x + w <= 1920
        assert 0 <= y and y + h <= 1080
        assert x <= 1500 < x + w
        assert y <= 540 < y + h

    def test_window_clamped_when_face_near_left_top(self):
        x, y, w, h = reelsfy.compute_fallback_crop(1920, 1080, (50, 20))
        assert (x, y, w, h) == (0, 0, 607, 1080)

    def test_window_clamped_when_face_near_right_bottom(self):
        x, y, w, h = reelsfy.compute_fallback_crop(1920, 1080, (1919, 1079))
        assert (x, y, w, h) == (1313, 0, 607, 1080)
        assert x + w <= 1920 and y + h <= 1080

    def test_crop_ratio_scales_the_window(self):
        # Half ratio: half the max window, centered on the face point.
        x, y, w, h = reelsfy.compute_fallback_crop(1920, 1080, (1500, 540), 0.5)
        assert (w, h) == (303, 540)
        assert x <= 1500 < x + w and y <= 540 < y + h


# ---------------------------------------------------------------------------
# 2. Fallback-crop wiring in generate_short
# ---------------------------------------------------------------------------


class TestFallbackCropWiring:
    """P3.4: the center-crop fallback branch uses face history when present."""

    @staticmethod
    def _selection_failing_detector():
        """
        Detector that selects the third face in the first interval, then
        fails selection later (its index is no longer valid) -> the loop
        clears face_positions and takes the center-crop fallback WHILE the
        clip has face history (prev_face_center).
        """
        detector = _make_detector(lookahead_result=None, single_result=None)
        detector.get_best_talking_face_with_lookahead.side_effect = [2] + [None] * 20
        detector.get_best_talking_face.side_effect = [None] * 20
        return detector

    def test_fallback_uses_last_face_center_after_face_lost(self):
        frames = [make_multi_face_frame() for _ in range(5)] + [
            make_face_frame() for _ in range(30)
        ]
        # fps=1.0 -> switch_interval=5: faces detected every interval, but
        # the talking-face selection fails from interval 1 on, so the loop
        # falls back while prev_face_center (the EMA-smoothed center) lives on.
        with _fallback_spy() as spy:
            result, writer, _ = _run_generate_short(
                frames, fps=1.0, detector=self._selection_failing_detector()
            )

        assert result, "fallback path must still produce a reel"
        assert spy.call_count > 0, "fallback branch must route through the helper"
        for call in spy.call_args_list:
            assert call.args[2] == MULTI_FACE_CENTER

    def test_fallback_output_matches_helper_window(self):
        frames = [make_multi_face_frame() for _ in range(5)] + [
            make_face_frame() for _ in range(30)
        ]
        with _fallback_spy():
            result, writer, _ = _run_generate_short(
                frames, fps=1.0, detector=self._selection_failing_detector()
            )

        face = make_face_frame()
        x, y, w, h = reelsfy.compute_fallback_crop(1920, 1080, MULTI_FACE_CENTER)
        expected = cv2.resize(
            face[y : y + h, x : x + w], (1080, 1920), interpolation=cv2.INTER_CUBIC
        )
        written = [call.args[0] for call in writer.write.call_args_list]
        assert written, "frames must be written"
        assert written[-1].shape == (1920, 1080, 3)
        assert np.array_equal(written[-1], expected), (
            "fallback frames must come from the last-known-face-centered window"
        )

    def test_no_face_history_keeps_centered_behavior(self):
        frames = [make_plain_frame() for _ in range(40)]
        with _fallback_spy() as spy:
            result, writer, _ = _run_generate_short(frames, fps=1.0)

        assert result
        assert spy.call_count > 0
        for call in spy.call_args_list:
            assert call.args[2] is None, (
                "no face history must keep the exact centered crop"
            )

        plain = frames[0]
        x, y, w, h = reelsfy.compute_fallback_crop(1920, 1080, None)
        expected = cv2.resize(
            plain[y : y + h, x : x + w], (1080, 1920), interpolation=cv2.INTER_CUBIC
        )
        written = [call.args[0] for call in writer.write.call_args_list]
        assert np.array_equal(written[0], expected)

    def test_screencast_path_never_uses_fallback_crop(self):
        frames = [make_face_frame() for _ in range(6)]
        with _fallback_spy() as spy, patch.dict(
            "os.environ", {"FRAMING_MODE": "pip"}
        ):
            result, writer, _ = _run_generate_short(frames)

        assert result
        assert spy.call_count == 0, (
            "screencast rendering handles framing itself (badge/zoom)"
        )
        assert writer.write.call_count > 0


# ---------------------------------------------------------------------------
# 3. Thumbnail: hook text + best face-quality smile frame
# ---------------------------------------------------------------------------


def _thumb_frame(tag):
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[:, :] = tag
    return frame


def _cycling_smile(candidates):
    """find_smile replacement returning candidates in order, then the last."""
    calls = {"n": 0}

    def fake_smile(frame, output_file):
        idx = min(calls["n"], len(candidates) - 1)
        calls["n"] += 1
        return True, candidates[idx]

    return fake_smile, calls


def _scorer_patch(scoring):
    """Patch reelsfy.FaceQualityScorer with a Mock class whose instance scores."""
    scorer_cls = MagicMock()
    scorer_cls.return_value.score_face_quality.side_effect = scoring
    return patch.object(reelsfy, "FaceQualityScorer", scorer_cls), scorer_cls


class TestThumbnailHookText:
    """P3.5: generate_thumbnail receives hook_text or title."""

    def test_hook_text_preferred_over_title(self, tmp_path):
        cand = _thumb_frame(10)
        fake_smile, _ = _cycling_smile([cand])
        with patch.object(reelsfy, "find_smile", side_effect=fake_smile), patch.object(
            reelsfy, "generate_thumbnail"
        ) as thumb_mock:
            result, _, _ = _run_generate_short(
                [make_face_frame()] * 30,
                thumb=True,
                thumb_text="STOP DOING X",
                thumb_dir=str(tmp_path),
            )

        assert result
        thumb_mock.assert_called_once()
        args = thumb_mock.call_args.args
        assert args[1] == "STOP DOING X"
        assert args[2] == str(tmp_path)

    def test_no_hook_keeps_title_text(self, tmp_path):
        cand = _thumb_frame(10)
        fake_smile, _ = _cycling_smile([cand])
        with patch.object(reelsfy, "find_smile", side_effect=fake_smile), patch.object(
            reelsfy, "generate_thumbnail"
        ) as thumb_mock:
            result, _, _ = _run_generate_short(
                [make_face_frame()] * 30, thumb=True, thumb_dir=str(tmp_path)
            )

        assert result
        assert thumb_mock.call_args.args[1] == "test"

    def test_screencast_pip_collects_candidates_with_hook_text(self, tmp_path):
        cand = _thumb_frame(10)
        fake_smile, _ = _cycling_smile([cand])
        with patch.dict("os.environ", {"FRAMING_MODE": "pip"}), patch.object(
            reelsfy, "find_smile", side_effect=fake_smile
        ) as smile_spy, patch.object(
            reelsfy, "generate_thumbnail"
        ) as thumb_mock:
            result, _, _ = _run_generate_short(
                [make_face_frame()] * 4,
                thumb=True,
                thumb_text="HOOK LINE",
                thumb_dir=str(tmp_path),
            )

        assert result
        assert smile_spy.call_count >= 1
        thumb_mock.assert_called_once()
        assert thumb_mock.call_args.args[1] == "HOOK LINE"


class TestBestSmileFrame:
    """P3.5: highest FaceQualityScorer score wins; failure -> first candidate."""

    def _three_candidates(self):
        return _thumb_frame(10), _thumb_frame(20), _thumb_frame(30)

    def test_highest_score_wins(self):
        a, b, c = self._three_candidates()
        score_by_id = {id(a): 0.1, id(b): 0.9, id(c): 0.5}
        scorer_patch, scorer_cls = _scorer_patch(
            lambda frame, bbox: score_by_id[id(frame)]
        )
        with scorer_patch:
            best = reelsfy.select_best_smile_frame([a, b, c])

        assert best is b
        assert scorer_cls.return_value.score_face_quality.call_count == 3

    def test_scoring_failure_falls_back_to_first_candidate(self):
        a, b, c = self._three_candidates()
        scorer_patch, _ = _scorer_patch(Mock(side_effect=RuntimeError("boom")))
        with scorer_patch:
            best = reelsfy.select_best_smile_frame([a, b, c])

        assert best is a

    def test_empty_scores_fall_back_to_first_candidate(self):
        a, b, c = self._three_candidates()
        scorer_patch, _ = _scorer_patch(lambda frame, bbox: None)
        with scorer_patch:
            best = reelsfy.select_best_smile_frame([a, b, c])

        assert best is a

    def test_single_candidate_needs_no_scorer(self):
        a = _thumb_frame(10)
        with patch.object(reelsfy, "FaceQualityScorer") as scorer_cls:
            best = reelsfy.select_best_smile_frame([a])

        assert best is a
        scorer_cls.assert_not_called()

    def test_no_candidates_no_thumbnail(self, tmp_path):
        with patch.object(
            reelsfy, "find_smile", return_value=(False, "")
        ) as smile_spy, patch.object(
            reelsfy, "generate_thumbnail"
        ) as thumb_mock:
            result, _, _ = _run_generate_short(
                [make_face_frame()] * 6, thumb=True, thumb_dir=str(tmp_path)
            )

        assert result
        assert smile_spy.call_count > 0, "detection must still be attempted"
        thumb_mock.assert_not_called()

    def test_candidates_capped_at_five(self):
        cand = _thumb_frame(10)
        fake_smile, calls = _cycling_smile([cand])
        with patch.object(reelsfy, "find_smile", side_effect=fake_smile), patch.object(
            reelsfy, "generate_thumbnail"
        ):
            result, _, _ = _run_generate_short(
                [make_face_frame()] * 30, thumb=True
            )

        assert result
        assert calls["n"] == 5, (
            "candidate collection must stop after the limit (no first-smile "
            "early exit, but also no unbounded collection)"
        )

    def test_best_candidate_wins_end_to_end(self, tmp_path):
        a, b, c = self._three_candidates()
        fake_smile, _ = _cycling_smile([a, b, c])
        score_by_id = {id(a): 0.1, id(b): 0.9, id(c): 0.5}
        scorer_patch, _ = _scorer_patch(
            lambda frame, bbox: score_by_id[id(frame)]
        )
        with patch.object(
            reelsfy, "find_smile", side_effect=fake_smile
        ), patch.object(reelsfy, "generate_thumbnail") as thumb_mock, scorer_patch:
            result, _, _ = _run_generate_short(
                [make_face_frame()] * 30, thumb=True, thumb_dir=str(tmp_path)
            )

        assert result
        assert thumb_mock.call_args.args[0] is b


# ---------------------------------------------------------------------------
# 4. Segment-loop wiring: hook fields flow into generate_short's thumb text
# ---------------------------------------------------------------------------


class TestSegmentLoopHookThumbTextWiring:
    """The segment loop parses hook fields and passes the text down."""

    def _segment_loop_source(self):
        tree = ast.parse(REELSFY_SOURCE)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__main__":
                return ast.get_source_segment(REELSFY_SOURCE, node)
        raise AssertionError("__main__ not found")

    def test_generate_short_receives_thumb_text_in_segment_loop(self):
        source = self._segment_loop_source()
        assert "thumb_text=" in source, (
            "the segment loop must pass thumb_text down to generate_short"
        )

    def test_hook_parsed_before_generate_short_in_segment_loop(self):
        source = self._segment_loop_source()
        hook_pos = source.find("parse_hook_fields(")
        short_pos = source.find("generate_short(")
        assert hook_pos != -1 and short_pos != -1
        assert hook_pos < short_pos, (
            "hook fields must be parsed before generate_short so the hook "
            "text can be passed as the thumbnail text"
        )

    def test_generate_short_signature_has_thumb_text(self):
        tree = ast.parse(REELSFY_SOURCE)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "generate_short":
                defaults = {
                    arg.arg: default
                    for arg, default in zip(
                        node.args.args[-len(node.args.defaults):],
                        node.args.defaults,
                    )
                }
                assert "thumb_text" in defaults
                return
        raise AssertionError("generate_short not found")
