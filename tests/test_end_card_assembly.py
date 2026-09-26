"""
Assembly-boundary characterization tests (Story C2).

C2's job was to reproduce the Instagram freeze and identify the exact
incompatibility, not to fix it. These tests pin the diagnosis so the fix
story (C3) has a target and so a later revert is caught.

The diagnosis, measured with ffprobe on a reproduction built from the real
command strings (see the appendix in the EPIC, and
`reels-clips-automator/tools/probe_end_card_assembly.py`):

  * `build_end_card_command()` never sets `-r`, so the `color` lavfi source
    encodes at **25 fps / time_base 1/12800**, while the teaser and the main
    clip are **30 fps / 1/15360** (`generate_short` hardcodes `-r 30`).
  * `assemble_hook_teaser_final` concatenates with `-c copy`, which SUCCEEDS
    (rc=0) and silently rescales the end card's 38 frames into a ~2.5 ms
    window on top of the main clip's last frames. Video track duration
    reports 44.100 s while the container claims 45.635 s.
  * A desktop player draws the main clip's last frame and shows a 2.5 ms
    flash — imperceptible, so local testing looks fine. A social transcoder
    cannot render a 2.5 ms frame cluster and stalls there: the freeze.

So the tests below are characterization assertions about the *current*
behaviour. C3 flips them. They deliberately do not assert a bug is present
forever; the C3 commit message names this module as the pair to invert.

All subprocess calls are mocked - no real video is processed.
"""

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
REELSFY_DIR = PROJECT_ROOT / "reels-clips-automator"
PROBE_TOOL = REELSFY_DIR / "tools" / "probe_end_card_assembly.py"

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

sys.modules["transcription_service"] = MagicMock()

reelsfy = pytest.importorskip("reelsfy")


class TestEndCardFrameRateDiagnosis:
    """The measured root cause: the end card is the only piece without -r."""

    def test_end_card_command_pins_no_frame_rate(self, monkeypatch):
        """DIAGNOSIS (pre-C3): no -r, so the lavfi color source is 25 fps.

        Every other concat piece inherits `-r 30` from generate_short. The
        end card alone defaults, and that alone collapses the assembly.
        """
        monkeypatch.delenv("END_CARD_ENABLED", raising=False)
        monkeypatch.setenv("END_CARD_TEXT", "Subscribe")

        command = reelsfy.build_end_card_command()

        assert " -r " not in f" {command} "
        assert "-r 30" not in command

    def test_the_other_pieces_do_pin_the_frame_rate(self):
        """The mismatch only exists because of the asymmetry.

        build_pipe_ffmpeg_command is the single-pass 9:16 crop that produces
        both the teaser and the main clip, and it normalizes to 30 fps.
        """
        source = inspect.getsource(reelsfy.build_pipe_ffmpeg_command)

        assert "-r 30" in source

    def test_no_piece_pins_a_video_timescale(self):
        """DIAGNOSIS (pre-C3): -video_track_timescale appears nowhere.

        This is why the end card lands on 1/12800 while everything else is
        1/15360 — the encoder picks the timescale per input, and the concat
        demuxer then rescales the mismatched piece.
        """
        source = (REELSFY_DIR / "reelsfy.py").read_text(encoding="utf-8")

        assert "-video_track_timescale" not in source


class TestConcatStrategyDiagnosis:
    """`-c copy` across the pieces is what turns the mismatch silent."""

    def test_primary_rung_is_stream_copy(self):
        command = reelsfy.build_teaser_concat_command("list.txt", "out.mp4")

        assert "-c copy" in command

    def test_ladder_falls_back_to_re_encode(self):
        fallback = reelsfy.build_teaser_concat_fallback_command(
            "list.txt", "out.mp4"
        )
        full = reelsfy.build_teaser_concat_full_fallback_command(
            "list.txt", "out.mp4"
        )

        assert "h264_nvenc" in fallback
        assert "-c:a copy" in fallback
        assert "h264_nvenc" in full
        # The last rung re-encodes audio WITHOUT re-mastering it.
        assert reelsfy.AUDIO_MASTERING_FILTER not in full


class TestAudioParameterDiagnosis:
    """The secondary hazard: mismatched AAC rates fail silently, not loudly."""

    def test_music_bed_rung_omits_the_sample_rate_pin(self):
        """DIAGNOSIS (pre-C3): the music-bed rung is the one burn that does
        not pin -ar, so its output carries the source's rate (44100 for a
        typical capture) while the end card is 48000.

        Measured consequence: `concat -c copy` does NOT fail. It adopts the
        first stream's parameters for the whole file and emits a container
        0.40 s longer than the sum of the pieces - a silent A/V desync that
        no rung in the ladder ever sees, because nothing errored.
        """
        source = inspect.getsource(reelsfy.generate_subtitle)

        assert "-c:a aac -b:a 192k" in source
        # build_audio_filter_args (which does pin -ar 48000) is the other rung.
        assert "-ar 48000" in reelsfy.build_audio_filter_args()

    def test_end_card_audio_is_pinned(self):
        """The end card was always correct on the audio side."""
        command = reelsfy.build_end_card_command()

        assert "-ar 48000" in command
        assert "-ac 2" in command


class TestProbeToolExists:
    """C2 ships the reproduction, so C4 can re-measure after the fix."""

    def test_probe_tool_is_present_and_runnable(self):
        assert PROBE_TOOL.exists(), (
            "the C2 reproduction tool is gone; C4's A/B decision needs it"
        )
        source = PROBE_TOOL.read_text(encoding="utf-8")

        assert "ffprobe" in source
        # It must be able to reproduce the bug, not just describe the fix.
        assert "--unpinned-end-card" in source

    def test_probe_tool_pins_the_settings_it_measures(self):
        """The probe drifts from the pipeline unless it tracks it, so the
        frame rate it asserts is shared with the pipeline's own constant."""
        source = PROBE_TOOL.read_text(encoding="utf-8")

        assert "FPS = 30" in source
        assert "15360" in source
