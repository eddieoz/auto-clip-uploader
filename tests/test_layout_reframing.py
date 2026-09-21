"""
Layout-aware reframing tests for Story 8 (P2.1): screencast-first reframing
in reelsfy.py.

Covers:
1. detect_layout classification table (screencast vs talking_head vs forced
   FRAMING_MODE override).
2. PiP renderer pure function: 1080x1920 output, circular face-cam badge in
   the top-right with a white ring, background crop respects the screen
   region, push-in scale eases 100% -> 96%.
3. Screen-zoom crop centers on the synthetic high-motion point and EMA
   smooths (with the existing dead-zone threshold).
4. Deterministic variety schedule (5 s boundaries, pip first, alternating).
5. Talking-head behavior contract preserved (random modes still reachable,
   center-crop fallback intact).
6. Badge hold-on-detection-miss behavior.
7. Badge stabilization helpers (strong EMA, dead-zone, re-crop on real
   movement only, size smoothing).

All cv2.VideoCapture/VideoWriter and Haar usage is mocked - no real video is
processed. Follows the sys.modules stub pattern from
tests/test_reelsfy_bugfixes.py.
"""

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

# transcription_service is a local import inside generate_subtitle; stub it so
# importing reelsfy stays light.
sys.modules.setdefault("transcription_service", MagicMock())

import cv2  # noqa: E402  (real cv2 is installed)

import reelsfy  # noqa: E402


REELSFY_SOURCE = (REELSFY_DIR / "reelsfy.py").read_text()


# ---------------------------------------------------------------------------
# Synthetic frame helpers
# ---------------------------------------------------------------------------

def make_screencast_frame():
    """
    1920x1080 screencast-style synthetic frame:
    - a dark "face" block in the top third (center-y ~16%, height ~22%),
    - striped (edgy) "screen content" below the face block.
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[60:300, 800:1040] = 40  # face block: center (920, 180)
    cols = np.arange(1920)
    stripes = np.where((cols // 10) % 2 == 0, 160, 200).astype(np.uint8)
    frame[300:1080, :, :] = stripes[None, :, None]
    return frame


def make_talking_head_frame():
    """Uniform frame -> no meaningful edge density below any face box."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# Screencast-classification face box (top third, small)
SC_FACE = (800, 60, 240, 240)
# Talking-head-classification face box (centered, larger)
TH_FACE = (860, 400, 400, 400)

FRAME_SHAPE = (1080, 1920)


def _many_faces(box, samples=30):
    """Observation list where a face is present in >= 30% of samples."""
    return [box] * samples


def _few_faces(box, present=2, total=30):
    boxes = [None] * total
    for i in range(present):
        boxes[i] = box
    return boxes


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
    frame,
    cascade_boxes,
    total_frames=90,
    loop_frames=3,
    debug_faces=False,
    fps=30.0,
):
    """
    Run generate_short against mocked cv2 and subprocess.

    Simulates a 1920x1080 @ fps video of `total_frames` frames. The sampler
    sees `cascade_boxes` for every sampled frame; the frame loop sees the
    same for every read frame. Returns (result, writer_mock, capsys capture).

    Story 9: the primary encode path pipes frames into one ffmpeg pass, so
    the returned `writer` is the mocked _PipeFrameWriter sink receiving the
    rendered frames.
    """
    detector = _make_detector(lookahead_result=0, single_result=0)

    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.side_effect = [(True, frame.copy()) for _ in range(loop_frames)] + [
        (False, None)
    ] * 50
    cap.grab.return_value = True
    cap.retrieve.return_value = (True, frame.copy())
    cap.get.side_effect = lambda attr: {
        cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_COUNT: float(total_frames),
    }.get(attr, 0.0)

    cascade = MagicMock()
    cascade.detectMultiScale.return_value = np.array(cascade_boxes, dtype=np.int32)

    writer = MagicMock()

    with patch.object(reelsfy.cv2, "VideoCapture", return_value=cap), patch.object(
        reelsfy.cv2, "VideoWriter", return_value=writer
    ), patch.object(reelsfy.cv2, "CascadeClassifier", return_value=cascade), patch.object(
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

    return result, writer


# ---------------------------------------------------------------------------
# 1. Layout detection
# ---------------------------------------------------------------------------


class TestDetectLayout:
    """detect_layout classification table."""

    def test_screencast_when_face_top_small_and_screen_below(self):
        layout = reelsfy.detect_layout(
            _many_faces(SC_FACE), FRAME_SHAPE, 0.10, forced_mode="auto"
        )
        assert layout == reelsfy.LAYOUT_SCREENCAST

    def test_talking_head_when_face_in_bottom_half(self):
        layout = reelsfy.detect_layout(
            _many_faces(TH_FACE), FRAME_SHAPE, 0.10, forced_mode="auto"
        )
        assert layout == reelsfy.LAYOUT_TALKING_HEAD

    def test_talking_head_when_face_too_large(self):
        big = (800, 60, 500, 500)  # height 500/1080 = 46% > 35%
        layout = reelsfy.detect_layout(
            _many_faces(big), FRAME_SHAPE, 0.10, forced_mode="auto"
        )
        assert layout == reelsfy.LAYOUT_TALKING_HEAD

    def test_talking_head_when_too_few_face_samples(self):
        layout = reelsfy.detect_layout(
            _few_faces(SC_FACE, present=2, total=30), FRAME_SHAPE, 0.10,
            forced_mode="auto",
        )
        assert layout == reelsfy.LAYOUT_TALKING_HEAD

    def test_talking_head_when_edge_density_too_low(self):
        layout = reelsfy.detect_layout(
            _many_faces(SC_FACE), FRAME_SHAPE, 0.01, forced_mode="auto"
        )
        assert layout == reelsfy.LAYOUT_TALKING_HEAD

    def test_no_faces_at_all_is_talking_head(self):
        layout = reelsfy.detect_layout(
            [None] * 30, FRAME_SHAPE, 0.0, forced_mode="auto"
        )
        assert layout == reelsfy.LAYOUT_TALKING_HEAD

    def test_forced_pip_override_returns_screencast(self):
        # Even talking-head-looking observations classify as screencast.
        layout = reelsfy.detect_layout(
            _many_faces(TH_FACE), FRAME_SHAPE, 0.0, forced_mode="pip"
        )
        assert layout == reelsfy.LAYOUT_SCREENCAST

    def test_forced_screen_zoom_override_returns_screencast(self):
        layout = reelsfy.detect_layout(
            _many_faces(TH_FACE), FRAME_SHAPE, 0.0, forced_mode="screen_zoom"
        )
        assert layout == reelsfy.LAYOUT_SCREENCAST

    def test_forced_closeup_override_returns_talking_head(self):
        layout = reelsfy.detect_layout(
            _many_faces(SC_FACE), FRAME_SHAPE, 0.20, forced_mode="closeup"
        )
        assert layout == reelsfy.LAYOUT_TALKING_HEAD


class TestFramingModeOverride:
    """FRAMING_MODE env override: auto (default) | pip | screen_zoom | closeup."""

    def test_default_is_auto(self, monkeypatch):
        monkeypatch.delenv("FRAMING_MODE", raising=False)
        assert reelsfy.resolve_framing_override() == "auto"

    def test_pip_override(self, monkeypatch):
        monkeypatch.setenv("FRAMING_MODE", "pip")
        assert reelsfy.resolve_framing_override() == "pip"

    def test_screen_zoom_override(self, monkeypatch):
        monkeypatch.setenv("FRAMING_MODE", "screen_zoom")
        assert reelsfy.resolve_framing_override() == "screen_zoom"

    def test_closeup_override(self, monkeypatch):
        monkeypatch.setenv("FRAMING_MODE", "closeup")
        assert reelsfy.resolve_framing_override() == "closeup"

    def test_invalid_value_falls_back_to_auto(self, monkeypatch, capsys):
        monkeypatch.setenv("FRAMING_MODE", "bananas")
        assert reelsfy.resolve_framing_override() == "auto"


class TestMedianFaceBoxAndScreenRegion:
    def test_median_of_face_boxes(self):
        boxes = [(0, 0, 100, 100), (0, 50, 200, 200), (0, 100, 300, 300)]
        med = reelsfy.median_face_box(boxes)
        assert med == (0, 50, 200, 200)

    def test_median_ignores_none_samples(self):
        boxes = [None, (0, 0, 100, 100), None, (0, 20, 120, 120), None]
        med = reelsfy.median_face_box(boxes)
        assert med == (0, 10, 110, 110)

    def test_median_of_all_none_is_none(self):
        assert reelsfy.median_face_box([None, None]) is None

    def test_screen_region_below_median_face(self):
        region = reelsfy.screen_region_from_median_box((800, 60, 240, 240), FRAME_SHAPE)
        sx, sy, sw, sh = region
        assert (sx, sw) == (0, 1920)
        assert 300 <= sy <= 330  # face bottom (300) + small margin
        assert sy + sh == 1080

    def test_screen_region_full_frame_when_no_face(self):
        region = reelsfy.screen_region_from_median_box(None, FRAME_SHAPE)
        assert region == (0, 0, 1920, 1080)


# ---------------------------------------------------------------------------
# 2. PiP renderer
# ---------------------------------------------------------------------------


class TestRenderPipFrame:
    def test_output_is_1080x1920(self):
        frame = make_screencast_frame()
        region = reelsfy.screen_region_from_median_box(SC_FACE, FRAME_SHAPE)
        out, badge = reelsfy.render_pip_frame(
            frame, region, SC_FACE, (920, 180), push_scale=1.0, last_badge=None
        )
        assert out.shape == (1920, 1080, 3)

    def test_badge_present_at_top_right_with_ring(self):
        frame = make_screencast_frame()
        region = reelsfy.screen_region_from_median_box(SC_FACE, FRAME_SHAPE)
        out, badge = reelsfy.render_pip_frame(
            frame, region, SC_FACE, (920, 180), push_scale=1.0, last_badge=None
        )

        out_w, out_h = 1080, 1920
        d = int(out_w * reelsfy.PIP_BADGE_DIAMETER_RATIO)
        margin = max(10, int(out_w * 0.03))
        bcx = out_w - d // 2 - margin
        bcy = d // 2 + margin
        ring_w = max(2, d // 30)

        # Ring pixel (top of the badge circle) is white
        ring_pixel = out[bcy, bcx + d // 2 - ring_w // 2]
        assert ring_pixel.min() > 230, f"ring pixel not white: {ring_pixel}"

        # Badge interior is the dark face crop, not the striped background
        center_pixel = out[bcy, bcx]
        assert center_pixel.max() < 100, f"badge center not dark face: {center_pixel}"

        assert badge is not None

    def test_background_comes_from_screen_region(self):
        frame = make_screencast_frame()
        region = reelsfy.screen_region_from_median_box(SC_FACE, FRAME_SHAPE)
        out, _ = reelsfy.render_pip_frame(
            frame, region, SC_FACE, (920, 180), push_scale=1.0, last_badge=None
        )

        # Far from the badge, pixels must come from the striped screen region
        patch = out[1200:1400, 100:300]
        vals = patch.reshape(-1, 3)
        assert vals.max() <= 210
        assert vals.min() >= 140  # stripe values 160/200, never face-dark (40)
        assert not np.any(vals < 60), "face/top-of-frame content leaked into background"

    def test_push_in_scale_eases_100_to_96_percent(self):
        assert abs(reelsfy.compute_push_in_scale(0.0) - 1.0) < 1e-9
        end = reelsfy.compute_push_in_scale(1.0)
        assert 0.955 < end <= 0.965
        mid = reelsfy.compute_push_in_scale(0.5)
        assert 0.975 < mid < 0.995  # smoothstep midpoint

    def test_push_in_changes_background(self):
        frame = make_screencast_frame()
        region = reelsfy.screen_region_from_median_box(SC_FACE, FRAME_SHAPE)
        out_start, _ = reelsfy.render_pip_frame(
            frame, region, SC_FACE, (920, 180), push_scale=1.0, last_badge=None
        )
        out_end, _ = reelsfy.render_pip_frame(
            frame, region, SC_FACE, (920, 180),
            push_scale=reelsfy.compute_push_in_scale(1.0), last_badge=None,
        )
        assert not np.array_equal(out_start, out_end)

    def test_badge_held_on_detection_miss(self):
        frame = make_screencast_frame()
        region = reelsfy.screen_region_from_median_box(SC_FACE, FRAME_SHAPE)

        # First frame: face present -> badge crop returned
        _, badge = reelsfy.render_pip_frame(
            frame, region, SC_FACE, (920, 180), push_scale=1.0, last_badge=None
        )
        assert badge is not None

        # Next frame: detection miss -> badge held from previous frame
        out, held = reelsfy.render_pip_frame(
            frame, region, None, None, push_scale=1.0, last_badge=badge
        )
        assert held is badge

        out_w = 1080
        d = int(out_w * reelsfy.PIP_BADGE_DIAMETER_RATIO)
        margin = max(10, int(out_w * 0.03))
        bcx = out_w - d // 2 - margin
        bcy = d // 2 + margin
        ring_w = max(2, d // 30)
        ring_pixel = out[bcy, bcx + d // 2 - ring_w // 2]
        assert ring_pixel.min() > 230, "held badge ring missing on detection miss"


# ---------------------------------------------------------------------------
# 3. Screen-zoom
# ---------------------------------------------------------------------------


class TestScreenZoom:
    def test_motion_center_found_on_synthetic_blob(self):
        prev_gray = np.zeros((1080, 1920), dtype=np.uint8)
        gray = np.zeros((1080, 1920), dtype=np.uint8)
        gray[100:200, 300:400] = 255  # motion blob centered at (350, 150)

        center = reelsfy.compute_motion_center(prev_gray, gray, FRAME_SHAPE)
        assert center is not None
        cx, cy = center
        cell_w = 1920 / 16
        cell_h = 1080 / 16
        assert abs(cx - 350) <= cell_w
        assert abs(cy - 150) <= cell_h

    def test_no_motion_returns_none(self):
        gray = np.zeros((1080, 1920), dtype=np.uint8)
        assert reelsfy.compute_motion_center(gray.copy(), gray, FRAME_SHAPE) is None

    def test_no_previous_frame_returns_none(self):
        gray = np.zeros((1080, 1920), dtype=np.uint8)
        assert reelsfy.compute_motion_center(None, gray, FRAME_SHAPE) is None

    def test_crop_centers_on_motion_point(self):
        region = reelsfy.screen_region_from_median_box(SC_FACE, FRAME_SHAPE)
        crop = reelsfy.compute_screen_zoom_crop(region, (350, 700), FRAME_SHAPE)
        cx, cy, cw, ch = crop

        assert abs(cx + cw // 2 - 350) <= 1  # centered on the motion x
        assert ch == 1080  # crop extends to the full frame height
        assert 0 <= cx and cx + cw <= 1920
        assert 0 <= cy and cy + ch <= 1080

    def test_smooth_center_ema_blend(self):
        prev = (100, 100)
        result = reelsfy.smooth_center(prev, (120, 100), smoothing_factor=0.7)
        assert result == (106, 100)

    def test_smooth_center_dead_zone_keeps_previous(self):
        prev = (100, 100)
        result = reelsfy.smooth_center(prev, (103, 100), smoothing_factor=0.7,
                                       movement_threshold=5)
        assert result == (100, 100)

    def test_smooth_center_first_frame_uses_raw(self):
        assert reelsfy.smooth_center(None, (42, 24)) == (42, 24)

    def test_smooth_center_holds_previous_on_miss(self):
        assert reelsfy.smooth_center((10, 20), None) == (10, 20)


# ---------------------------------------------------------------------------
# 3. Badge stabilization (user feedback: badge content shook every frame)
# ---------------------------------------------------------------------------


class TestBadgeStabilization:
    def test_smooth_badge_side_first_value_taken_as_is(self):
        assert reelsfy.smooth_badge_side(None, 320) == 320

    def test_smooth_badge_side_blends_with_prev(self):
        # 90% previous / 10% new with the PIP_BADGE_SMOOTHING_FACTOR default
        assert reelsfy.smooth_badge_side(300, 400) == int(300 * 0.9 + 400 * 0.1)

    def test_smooth_badge_side_keeps_prev_on_miss(self):
        assert reelsfy.smooth_badge_side(310, None) == 310

    def test_should_recenter_true_when_nothing_held(self):
        assert reelsfy.should_recenter_badge((500, 200), None, 320)

    def test_should_recenter_false_for_small_jitter(self):
        # Jitter below 12% of the badge size is frozen out
        assert not reelsfy.should_recenter_badge((505, 203), (500, 200), 320)

    def test_should_recenter_true_on_real_movement(self):
        # 100 px drift is far beyond 12% of a 320 px badge
        assert reelsfy.should_recenter_badge((600, 200), (500, 200), 320)

    def test_should_recenter_false_when_center_none(self):
        assert not reelsfy.should_recenter_badge(None, (500, 200), 320)

    def test_render_pip_frame_face_side_override(self):
        frame = make_screencast_frame()
        region = reelsfy.screen_region_from_median_box(SC_FACE, FRAME_SHAPE)
        out_a, badge_a = reelsfy.render_pip_frame(
            frame, region, SC_FACE, (920, 180), push_scale=1.0,
            last_badge=None, face_side=400,
        )
        # Explicit face_side drives the crop size (EMA-smoothed in the loop)
        assert badge_a.shape[0] == 400
        # Same call with a DIFFERENT explicit side produces a different crop
        _, badge_b = reelsfy.render_pip_frame(
            frame, region, SC_FACE, (920, 180), push_scale=1.0,
            last_badge=None, face_side=500,
        )
        assert badge_b.shape[0] == 500

    def test_stabilization_constants_binding(self):
        assert reelsfy.PIP_BADGE_SMOOTHING_FACTOR == 0.9
        assert reelsfy.PIP_BADGE_MOVEMENT_THRESHOLD == 15
        assert 0.0 < reelsfy.PIP_BADGE_RECENTER_RATIO < 0.5


# ---------------------------------------------------------------------------
# 4. Deterministic variety schedule
# ---------------------------------------------------------------------------


class TestVarietySchedule:
    def test_first_segment_is_pip(self):
        assert reelsfy.screencast_mode_for_frame(0, 30.0) == "pip"

    def test_within_5s_is_pip(self):
        assert reelsfy.screencast_mode_for_frame(149, 30.0) == "pip"

    def test_second_segment_is_screen_zoom(self):
        assert reelsfy.screencast_mode_for_frame(150, 30.0) == "screen_zoom"
        assert reelsfy.screencast_mode_for_frame(299, 30.0) == "screen_zoom"

    def test_pattern_alternates_every_5s(self):
        assert reelsfy.screencast_mode_for_frame(299, 30.0) == "screen_zoom"
        assert reelsfy.screencast_mode_for_frame(300, 30.0) == "pip"
        assert reelsfy.screencast_mode_for_frame(449, 30.0) == "pip"
        assert reelsfy.screencast_mode_for_frame(450, 30.0) == "screen_zoom"

    def test_segment_seconds_is_5(self):
        assert reelsfy.SCREENCAST_SEGMENT_SECONDS == 5

    def test_forced_pip_overrides_schedule(self):
        for fc in (0, 300, 600, 900):
            assert reelsfy.resolve_screencast_mode(fc, 30.0, forced="pip") == "pip"

    def test_forced_screen_zoom_overrides_schedule(self):
        for fc in (0, 300, 600, 900):
            assert (
                reelsfy.resolve_screencast_mode(fc, 30.0, forced="screen_zoom")
                == "screen_zoom"
            )

    def test_auto_uses_schedule(self):
        assert reelsfy.resolve_screencast_mode(0, 30.0, forced="auto") == "pip"
        assert (
            reelsfy.resolve_screencast_mode(150, 30.0, forced="auto")
            == "screen_zoom"
        )


# ---------------------------------------------------------------------------
# 5. Talking-head contract preserved
# ---------------------------------------------------------------------------


class TestTalkingHeadContract:
    """The talking-head path keeps its random modes and center-crop fallback."""

    def test_random_mode_constants_still_present(self):
        assert "TRACKING_STATIC_PROBABILITY" in REELSFY_SOURCE
        assert "ZOOM_CLOSEUP_PROBABILITY" in REELSFY_SOURCE
        assert "ZOOM_IN_PROBABILITY" in REELSFY_SOURCE
        assert "random.random() < TRACKING_STATIC_PROBABILITY" in REELSFY_SOURCE

    def test_center_crop_fallback_branch_still_present(self):
        assert "using center crop fallback" in REELSFY_SOURCE
        assert "No faces detected, using center crop fallback" in REELSFY_SOURCE

    def test_uniform_frame_runs_talking_head_path(self, capsys):
        """A uniform (no-edge-density) frame classifies as talking_head and the
        existing random tracking/zoom machinery still drives the crop."""
        frame = make_talking_head_frame()
        result, writer = _run_generate_short(
            frame,
            cascade_boxes=[[860, 400, 400, 400]],
            debug_faces=True,
        )

        captured = capsys.readouterr()
        assert result == "tmp/final-output_reel.mp4"
        assert "Layout:" in captured.out
        assert "talking_head" in captured.out
        # Existing per-interval random mode selection still runs
        assert "tracking_mode=" in captured.out
        # The 5 s look-ahead consumed the rest of this tiny clip, but the
        # talking-head crop path still wrote its frame.
        assert writer.write.call_count >= 1
        assert writer.write.call_args_list[0].args[0].shape == (1920, 1080, 3)

    def test_closeup_env_override_forces_talking_head_path(self, monkeypatch, capsys):
        monkeypatch.setenv("FRAMING_MODE", "closeup")
        frame = make_screencast_frame()
        result, writer = _run_generate_short(
            frame,
            cascade_boxes=[list(SC_FACE)],
            debug_faces=True,
        )

        captured = capsys.readouterr()
        assert result == "tmp/final-output_reel.mp4"
        assert "closeup override" in captured.out
        # Random talking-head modes still drive the crop
        assert "tracking_mode=" in captured.out
        assert writer.write.call_count >= 1


# ---------------------------------------------------------------------------
# 6. Screencast end-to-end
# ---------------------------------------------------------------------------


class TestScreencastEndToEnd:
    def test_screencast_frames_render_as_pip(self, capsys):
        frame = make_screencast_frame()
        result, writer = _run_generate_short(
            frame, cascade_boxes=[list(SC_FACE)], debug_faces=True
        )

        captured = capsys.readouterr()
        assert result == "tmp/final-output_reel.mp4"
        assert "screencast" in captured.out
        # Within the first 10 s everything is pip; no random talking-head modes
        assert "tracking_mode=" not in captured.out
        assert "Screencast mode switch" in captured.out

        assert writer.write.call_count == 3
        for call in writer.write.call_args_list:
            written = call.args[0]
            assert written.shape == (1920, 1080, 3)

        # Background comes from the striped screen region
        first = writer.write.call_args_list[0].args[0]
        vals = first[1200:1400, 100:300].reshape(-1, 3)
        assert vals.min() >= 140
        assert not np.any(vals < 60)

    def test_screencast_frames_differ_between_scheduled_modes(self, capsys):
        """Over a >10 s clip the schedule switches pip -> screen_zoom."""
        frame = make_screencast_frame()
        # 12 s clip at 30 fps: first 10 s pip, last 2 s screen_zoom
        result, writer = _run_generate_short(
            frame,
            cascade_boxes=[list(SC_FACE)],
            total_frames=360,
            loop_frames=360,
            fps=30.0,
            debug_faces=True,
        )

        assert result == "tmp/final-output_reel.mp4"
        assert writer.write.call_count == 360
        captured = capsys.readouterr()
        assert "Screencast mode switch" in captured.out

    def test_screencast_missing_face_holds_badge(self, capsys):
        """After frames run out of detections the badge is held, not dropped."""
        frame = make_screencast_frame()
        # Sampler sees a face (classification works), loop loses it after frame 1
        cap_detector_boxes = {"calls": 0}

        detector = _make_detector(lookahead_result=0, single_result=0)
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, frame.copy()) for _ in range(3)] + [
            (False, None)
        ] * 50
        cap.grab.return_value = True
        cap.retrieve.return_value = (True, frame.copy())
        cap.get.side_effect = lambda attr: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 90.0,
        }.get(attr, 0.0)

        def detect(gray, *args, **kwargs):
            # 30 sampler calls see the face (classification works); loop
            # frame 0 sees it too, frames 1-2 lose it -> badge must hold.
            cap_detector_boxes["calls"] += 1
            if cap_detector_boxes["calls"] <= 31:
                return np.array([list(SC_FACE)], dtype=np.int32)
            return np.array([], dtype=np.int32).reshape(0, 4)

        cascade = MagicMock()
        cascade.detectMultiScale.side_effect = detect
        writer = MagicMock()

        with patch.object(reelsfy.cv2, "VideoCapture", return_value=cap), patch.object(
            reelsfy.cv2, "VideoWriter", return_value=writer
        ), patch.object(reelsfy.cv2, "CascadeClassifier", return_value=cascade), patch.object(
            reelsfy.cv2, "destroyAllWindows"
        ), patch.object(
            reelsfy.subprocess, "Popen", return_value=_success_pipe_proc()
        ), patch.object(
            reelsfy, "_PipeFrameWriter", return_value=writer
        ), patch.object(reelsfy.subprocess, "call"), patch.object(
            reelsfy, "TalkingFaceDetector", return_value=detector
        ):
            result = reelsfy.generate_short("input.mp4", "output_reel.mp4")

        assert result == "tmp/final-output_reel.mp4"
        assert writer.write.call_count == 3
        for call in writer.write.call_args_list:
            written = call.args[0]
            assert written.shape == (1920, 1080, 3)
            # Badge ring must be present in every written frame (held badge)
            d = int(1080 * reelsfy.PIP_BADGE_DIAMETER_RATIO)
            margin = max(10, int(1080 * 0.03))
            bcx = 1080 - d // 2 - margin
            bcy = d // 2 + margin
            ring_w = max(2, d // 30)
            ring_pixel = written[bcy, bcx + d // 2 - ring_w // 2]
            assert ring_pixel.min() > 230
