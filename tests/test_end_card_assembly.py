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


class TestEndCardFrameRateFix:
    """C3: every assembly piece carries the same frame rate and timescale.

    The freeze came from one piece not declaring what the others inherited.
    All three now pin both, from constants that cannot drift apart.
    """

    def test_end_card_command_pins_the_shared_frame_rate(self, monkeypatch):
        monkeypatch.delenv("END_CARD_ENABLED", raising=False)
        monkeypatch.setenv("END_CARD_TEXT", "Subscribe")

        command = reelsfy.build_end_card_command()

        assert f"-r {reelsfy.ASSEMBLY_FPS}" in command
        assert (
            f"-video_track_timescale {reelsfy.ASSEMBLY_VIDEO_TIMESCALE}"
            in command
        )

    def test_teaser_burn_pins_the_shared_timescale(self):
        command = reelsfy.build_teaser_burn_command("hook text")

        assert (
            f"-video_track_timescale {reelsfy.ASSEMBLY_VIDEO_TIMESCALE}"
            in command
        )

    def test_the_crop_pins_the_shared_frame_rate(self):
        """generate_short's crop is where the 30 fps normalization lives.

        This is not only an intermediate: with no overlays, generate_subtitle
        stream-copies it straight into the published path, so its parameters
        would otherwise reach the assembly verbatim.
        """
        command = reelsfy.build_pipe_ffmpeg_command("in.mp4", "out.mp4", 60)

        assert f"-r {reelsfy.ASSEMBLY_FPS}" in command
        assert (
            f"-video_track_timescale {reelsfy.ASSEMBLY_VIDEO_TIMESCALE}"
            in command
        )
        assert f"-ar {reelsfy.ASSEMBLY_AUDIO_RATE}" in command

    def test_every_burn_pins_the_shared_timescale(self):
        """No assembly piece may be left to pick its own timescale.

        Checked on the BUILT commands, so this fails when a pin is actually
        removed rather than when the source is re-wrapped.
        """
        pin = f"-video_track_timescale {reelsfy.ASSEMBLY_VIDEO_TIMESCALE}"

        assert pin in reelsfy.build_end_card_command()
        assert pin in reelsfy.build_teaser_burn_command("hook")
        assert pin in reelsfy.build_pipe_ffmpeg_command("i.mp4", "o.mp4", 60)

    def test_the_legacy_merge_path_also_declares_the_shared_shape(self):
        """The fallback chain is the one a reviewer almost missed.

        When the single-pass pipe encoder fails, generate_short falls back to
        a cv2 mp4v write plus this merge, and the result is the SAME
        tmp/final-* file both burns read. So its parameters reach the concat
        whenever the no-overlay `-c copy` rung is taken — it must pin the
        frame rate, timescale, pixel format and audio rate like every other
        piece, or the original freeze comes back on the fallback path only.
        """
        source = inspect.getsource(reelsfy.generate_short)

        assert f"-r {{ASSEMBLY_FPS}}" in source
        assert f"-video_track_timescale {{ASSEMBLY_VIDEO_TIMESCALE}}" in source
        assert "-pix_fmt yuv420p" in source
        assert f"-ar {{ASSEMBLY_AUDIO_RATE}}" in source
        assert f"-ac {{ASSEMBLY_AUDIO_CHANNELS}}" in source

    def test_teaser_burn_pins_the_frame_rate_too(self):
        """A timescale alone is not enough: the teaser must not inherit a
        60 fps input from the legacy crop path."""
        command = reelsfy.build_teaser_burn_command("hook")

        assert f"-r {reelsfy.ASSEMBLY_FPS}" in command

    def test_shared_constants_are_defined_once(self):
        """The B1 lesson applies here: one definition, shared, so a retune
        cannot leave two pieces disagreeing again. Asserted on the VALUE,
        never a literal, so this test does not become a second source of
        truth for 30 and 15360.
        """
        source = (REELSFY_DIR / "reelsfy.py").read_text(encoding="utf-8")

        for name in ("ASSEMBLY_FPS", "ASSEMBLY_VIDEO_TIMESCALE",
                     "ASSEMBLY_AUDIO_RATE", "ASSEMBLY_AUDIO_CHANNELS"):
            assert source.count(f"\n{name} = ") == 1, f"{name} defined twice"
        # 15360 = 30 * 512, which is what NVENC picks for 30 fps; pin it so a
        # change of ASSEMBLY_FPS cannot silently leave the timescale behind.
        assert reelsfy.ASSEMBLY_VIDEO_TIMESCALE == reelsfy.ASSEMBLY_FPS * 512


class TestConcatStrategyUnchanged:
    """C3 deliberately keeps the -c copy concat and the ladder.

    The fix is in the pieces, not the concat. Re-encoding the whole clip per
    publish would cost a full NVENC pass and put the boundaries mid-GOP.
    """

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


class TestAudioParameterFix:
    """C3: no piece may inherit its rate from the source.

    The music-bed rung was the one burn that omitted -ar, so its output was
    44100 against the end card's 48000. `concat -c copy` does not fail on
    that - it adopts the first stream's parameters and emits a container
    longer than the sum of the pieces. A silent desync.
    """

    def test_music_bed_rung_pins_the_shared_audio_rate(self):
        """The one burn that used to emit bare `-c:a aac -b:a 192k`.

        The command is built inline, so the invariant is checked on the
        source: the rung must interpolate the shared constants instead of
        carrying a rate literal of its own. Written as a word-level check so
        re-wrapping the f-string does not break a passing build.
        """
        source = inspect.getsource(reelsfy.generate_subtitle)

        assert "-c:a aac -b:a 192k" in source
        assert "ASSEMBLY_AUDIO_RATE" in source
        assert "ASSEMBLY_AUDIO_CHANNELS" in source
        # no rate literal anywhere in the rung that could drift from the
        # shared constant
        assert "-ar 48000" not in source
        assert "-ar 44100" not in source

    def test_build_audio_filter_args_uses_the_shared_rate(self):
        args = reelsfy.build_audio_filter_args()

        assert f"-ar {reelsfy.ASSEMBLY_AUDIO_RATE}" in args
        assert f"-ac {reelsfy.ASSEMBLY_AUDIO_CHANNELS}" in args

    def test_end_card_audio_matches(self):
        command = reelsfy.build_end_card_command()

        assert f"-ar {reelsfy.ASSEMBLY_AUDIO_RATE}" in command
        assert f"-ac {reelsfy.ASSEMBLY_AUDIO_CHANNELS}" in command

    def test_mastering_is_still_applied_exactly_once(self):
        """The invariant the whole design protects: the assembly must never
        re-master, and the end card is silent-sourced so nothing may."""
        assert reelsfy.AUDIO_MASTERING_FILTER not in reelsfy.build_end_card_command()
        assert (
            reelsfy.AUDIO_MASTERING_FILTER
            not in reelsfy.build_teaser_concat_fallback_command("l", "o")
        )
        assert (
            reelsfy.AUDIO_MASTERING_FILTER
            not in reelsfy.build_teaser_concat_full_fallback_command("l", "o")
        )


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

    def test_probe_tool_reads_the_pipelines_constants(self):
        """The probe drifts from the pipeline unless it tracks it.

        Asserted by running its extractor, not by grepping for a name: a
        probe that hardcoded fps=25 would satisfy a substring check but fail
        here, which is the point.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "probe_end_card_assembly", PROBE_TOOL
        )
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)

        assert probe.CONST["fps"] == reelsfy.ASSEMBLY_FPS
        assert probe.CONST["timescale"] == reelsfy.ASSEMBLY_VIDEO_TIMESCALE
        assert probe.CONST["audio_rate"] == reelsfy.ASSEMBLY_AUDIO_RATE
        assert probe.CONST["audio_channels"] == reelsfy.ASSEMBLY_AUDIO_CHANNELS
        assert probe.CONST["mastering"] == reelsfy.AUDIO_MASTERING_FILTER
        # and it must pin the timescale on the pieces it compares, or a
        # retune of that constant would still report "healthy"
        assert probe.TIMESCALE_ARGS == [
            "-video_track_timescale", str(reelsfy.ASSEMBLY_VIDEO_TIMESCALE)
        ]
