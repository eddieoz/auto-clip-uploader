"""
Polish tests for Story 10 (P3.1-P3.3): visual grade, end card CTA, music bed.

Covers:
1. P3.1 visual grade: VIDEO_GRADE defaults on with the exact
   eq=saturation=1.06:contrast=1.02,unsharp=5:5:0.6 filter prepended as the
   FIRST filter of generate_subtitle's video chain (subtitles/overlays burn
   ON TOP of the graded image); VIDEO_GRADE=false removes it;
   VIDEO_LUT replaces the pair with lut3d=<path>.
2. P3.2 end card: builder command (solid color source, centered drawtext
   fontsize=64, fade-in, silent anullsrc AAC), concat list gains the
   end-card entry, END_CARD_ENABLED=false keeps the legacy byte-identical
   assembly, and the fallback ladder never re-masters audio.
3. P3.3 music bed: MUSIC_BED unset keeps the legacy -af audio args; set,
   the bed is added with -stream_loop -1, mixed UNDER the speech
   (volume -24dB, amix duration=first) and the mastering chain is still
   applied exactly once; the video chain is unchanged.

All subprocess calls are mocked - no real video/audio is processed.
"""

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

# transcription_service may pull heavy deps; always stub it so the local
# `from transcription_service import get_transcriber` inside generate_subtitle
# resolves to a controllable mock.
transcription_service_stub = MagicMock()
sys.modules["transcription_service"] = transcription_service_stub

import reelsfy  # noqa: E402


GRADE_FILTER = "eq=saturation=1.06:contrast=1.02,unsharp=5:5:0.6"

SAMPLE_WORDS = [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "world", "start": 0.5, "end": 1.0},
]

POLISH_ENV_VARS = (
    "VIDEO_GRADE",
    "VIDEO_LUT",
    "MUSIC_BED",
    "END_CARD_ENABLED",
    "END_CARD_COLOR",
    "END_CARD_TEXT",
)


def _clear_polish_env(monkeypatch):
    """Remove every Story 10 env var so tests exercise true defaults."""
    for name in POLISH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _run_generate_subtitle(tmp_path, monkeypatch, main_output_path=None):
    """
    Run generate_subtitle in an isolated tmp cwd with mocked subprocess
    (karaoke path, no transcription). Returns (result, commands).
    """
    workdir = tmp_path / "work"
    (workdir / "tmp").mkdir(parents=True, exist_ok=True)
    output_dir = workdir / "output"
    output_dir.mkdir(exist_ok=True)
    monkeypatch.chdir(workdir)

    with patch.object(reelsfy.subprocess, "call", return_value=0) as call_mock:
        result = reelsfy.generate_subtitle(
            "final-test_cropped000.mp4",
            "vid",
            str(output_dir),
            subtitle_words=list(SAMPLE_WORDS),
            main_output_path=main_output_path,
        )

    commands = [args[0][0] for args in call_mock.call_args_list]
    return result, commands


def _final_mux_command(commands, output_dir):
    """The single encode of the burned/published main clip."""
    finals = [c for c in commands if str(output_dir) in c or "hook-main.mp4" in c]
    assert finals, "no final mux command found"
    return finals[0]


class TestResolveVideoGrade:
    """P3.1: pure env resolution for the visual grade."""

    def test_default_enables_eq_unsharp_grade(self, monkeypatch):
        _clear_polish_env(monkeypatch)

        assert reelsfy.resolve_video_grade() == GRADE_FILTER

    def test_video_grade_false_disables_grade(self, monkeypatch):
        monkeypatch.setenv("VIDEO_GRADE", "false")

        assert reelsfy.resolve_video_grade() is None

    def test_video_lut_replaces_eq_unsharp_pair(self, monkeypatch):
        _clear_polish_env(monkeypatch)
        monkeypatch.setenv("VIDEO_LUT", "luts/warm.cube")

        result = reelsfy.resolve_video_grade()

        assert "lut3d=" in result
        assert "luts/warm.cube" in result
        assert "eq=" not in result
        assert "unsharp" not in result


class TestGradeInGenerateSubtitle:
    """P3.1: the grade is the FIRST video filter of the final mux."""

    def test_grade_prepended_before_subtitles(self, tmp_path, monkeypatch):
        _clear_polish_env(monkeypatch)
        _, commands = _run_generate_subtitle(tmp_path, monkeypatch)
        output_dir = tmp_path / "work" / "output"

        command = _final_mux_command(commands, output_dir)

        assert GRADE_FILTER in command
        grade_idx = command.index(GRADE_FILTER)
        subtitles_idx = command.index("subtitles=")
        assert grade_idx < subtitles_idx, (
            "grade must be burned before the captions, not after"
        )
        # The grade opens the filter_complex and hands [v0] to the captions.
        assert f'-filter_complex "{GRADE_FILTER}[v0];[v0]subtitles=' in command

    def test_grade_disabled_leaves_legacy_chain(self, tmp_path, monkeypatch):
        _clear_polish_env(monkeypatch)
        monkeypatch.setenv("VIDEO_GRADE", "false")
        _, commands = _run_generate_subtitle(tmp_path, monkeypatch)
        output_dir = tmp_path / "work" / "output"

        command = _final_mux_command(commands, output_dir)

        assert GRADE_FILTER not in command
        assert "lut3d=" not in command
        # Legacy audio mastering is untouched
        assert f'-af "{reelsfy.AUDIO_MASTERING_FILTER}"' in command

    def test_video_lut_used_instead_of_eq_unsharp(self, tmp_path, monkeypatch):
        _clear_polish_env(monkeypatch)
        monkeypatch.setenv("VIDEO_LUT", "luts/warm.cube")
        _, commands = _run_generate_subtitle(tmp_path, monkeypatch)
        output_dir = tmp_path / "work" / "output"

        command = _final_mux_command(commands, output_dir)

        assert "lut3d='luts/warm.cube'" in command
        assert GRADE_FILTER not in command
        # Captions still burn on top of the LUT
        assert command.index("lut3d=") < command.index("subtitles=")

    def test_video_lut_filter_specials_escaped(self, monkeypatch):
        # Final-fix D: the user-set LUT path goes into lut3d='...'; spaces
        # survive inside the quotes, but ' and : must be escaped so the
        # filtergraph parser cannot split the filename. Expected form
        # verified against ffmpeg 6.1 with the LUT file actually on disk.
        _clear_polish_env(monkeypatch)
        monkeypatch.setenv("VIDEO_LUT", "luts/warm grade's:sun.cube")

        result = reelsfy.resolve_video_grade()

        expected = "lut3d='" + r"luts/warm grade\\'\''s\\:sun.cube" + "'"
        assert expected in result
        assert "eq=" not in result


class TestResolveMusicBed:
    """P3.3: pure env resolution for the music bed path."""

    def test_unset_music_bed_is_off(self, monkeypatch):
        _clear_polish_env(monkeypatch)

        assert reelsfy.resolve_music_bed() is None

    def test_set_music_bed_returns_path(self, monkeypatch):
        monkeypatch.setenv("MUSIC_BED", "music/bed-loop.mp3")

        assert reelsfy.resolve_music_bed() == "music/bed-loop.mp3"


class TestMusicBedAudioFilters:
    """P3.3: pure builder for the -filter_complex audio half."""

    def test_mixes_bed_under_speech_then_masters_once(self):
        filters = reelsfy.build_music_bed_audio_filters()

        assert "[1:a]volume=-24dB[m]" in filters
        assert "[0:a][m]amix=inputs=2:duration=first[aout]" in filters
        # The SAME mastering chain ends the complex (applied exactly once)
        assert filters.count(reelsfy.AUDIO_MASTERING_FILTER) == 1
        assert filters.endswith(f"{reelsfy.AUDIO_MASTERING_FILTER}[aout2]")


class TestMusicBedInGenerateSubtitle:
    """P3.3: final mux integration."""

    def test_unset_keeps_legacy_audio_args(self, tmp_path, monkeypatch):
        _clear_polish_env(monkeypatch)
        _, commands = _run_generate_subtitle(tmp_path, monkeypatch)
        output_dir = tmp_path / "work" / "output"

        command = _final_mux_command(commands, output_dir)

        assert f'-af "{reelsfy.AUDIO_MASTERING_FILTER}"' in command
        assert "amix" not in command
        assert "-stream_loop" not in command

    def test_set_mixes_bed_and_masters_once(self, tmp_path, monkeypatch):
        _clear_polish_env(monkeypatch)
        monkeypatch.setenv("MUSIC_BED", "music/bed-loop.mp3")
        _, commands = _run_generate_subtitle(tmp_path, monkeypatch)
        output_dir = tmp_path / "work" / "output"

        command = _final_mux_command(commands, output_dir)

        # Bed added as a looped extra input
        assert "-stream_loop -1" in command
        assert '-i "music/bed-loop.mp3"' in command
        # Mixed UNDER the speech, then mastered exactly once
        assert "[1:a]volume=-24dB[m]" in command
        assert "[0:a][m]amix=inputs=2:duration=first[aout]" in command
        assert command.count(reelsfy.AUDIO_MASTERING_FILTER) == 1
        assert '-af "' not in command
        assert '-map "[aout2]"' in command
        # Video chain unchanged: captions and overlays still burn, NVENC still
        # encodes
        assert "subtitles=" in command
        assert "h264_nvenc" in command


class TestEndCardCommand:
    """P3.2: pure builder for the end-card clip command."""

    def test_default_command_shape(self, monkeypatch):
        _clear_polish_env(monkeypatch)

        command = reelsfy.build_end_card_command()

        # Solid 1080x1920 color source, 1.5 s
        assert "color=c=0x14213D:s=1080x1920:d=1.5" in command
        # Centered CTA drawtext: fontsize 64, white, black outline
        assert "drawtext=text='Segue para mais!'" in command
        assert f"fontfile={reelsfy.OVERLAY_FONTFILE}" in command
        assert "fontsize=64" in command
        assert "fontcolor=white" in command
        assert "bordercolor=black" in command
        assert "x=(w-tw)/2" in command
        assert "y=(h-th)/2" in command
        # Fade-in
        assert "fade=t=in:st=0:d=0.4" in command
        # Silent AAC audio, pinned to the same rate as the mastering args
        # (anullsrc source stays at 44100; ffmpeg resamples to the -ar rate)
        assert "anullsrc=r=44100:cl=stereo" in command
        assert "-c:a aac" in command
        assert "-ar 48000" in command
        assert "-ac 2" in command
        # Same NVENC settings as the other concat pieces
        assert "h264_nvenc" in command
        assert command.rstrip().endswith("tmp/end-card.mp4")
        # The end card is silent-sourced: no mastering may touch it
        assert reelsfy.AUDIO_MASTERING_FILTER not in command

    def test_env_overrides_color_and_text(self, monkeypatch):
        monkeypatch.setenv("END_CARD_COLOR", "0xFFFFFF")
        monkeypatch.setenv("END_CARD_TEXT", "Subscribe now")

        command = reelsfy.build_end_card_command()

        assert "color=c=0xFFFFFF" in command
        assert "drawtext=text='Subscribe now'" in command

    def test_end_card_audio_rate_matches_mastering_args(self, monkeypatch):
        # Final-fix A: the concat rungs only line up when both sides of the
        # assembly encode the same AAC sample rate.
        _clear_polish_env(monkeypatch)

        command = reelsfy.build_end_card_command()
        mastering_args = reelsfy.build_audio_filter_args()

        assert "-ar 48000" in command
        assert "-ar 48000" in mastering_args

    def test_env_text_with_filter_specials_is_escaped(self, monkeypatch):
        # Final-fix D: user-set text goes into a drawtext text='...'
        # filter value; the filtergraph specials must be escaped or the
        # command breaks (or injects extra filter options). Expected
        # forms verified byte-exact against ffmpeg 6.1: backslash and
        # colon doubled, quote as close-escape-reopen; '%' verbatim via
        # expansion=none.
        monkeypatch.setenv("END_CARD_TEXT", "It's a: 100% test")

        command = reelsfy.build_end_card_command()

        expected = "drawtext=text='" + r"It\\'\''s a\\: 100% test" + "'"
        assert expected in command
        assert "expansion=none" in command

    def test_env_text_escaping_runs_backslash_first(self, monkeypatch):
        # A literal backslash must be doubled before quote/colon escaping,
        # otherwise the introduced escapes get double-escaped.
        monkeypatch.setenv("END_CARD_TEXT", "a\\:b")

        command = reelsfy.build_end_card_command()

        expected = "drawtext=text='a" + ("\\" * 6) + ":b'"
        assert expected in command

    def test_env_text_with_double_quote_is_shell_escaped(self, monkeypatch):
        # The -vf argument is shell double-quoted, so a literal double
        # quote in the CTA text must be backslash-escaped for the shell.
        monkeypatch.setenv("END_CARD_TEXT", 'Vem "agora"')

        command = reelsfy.build_end_card_command()

        expected = "drawtext=text='" + 'Vem \\"agora\\"' + "'"
        assert expected in command


class TestGenerateEndCard:
    """P3.2: build-once-per-segment orchestration."""

    def test_success_returns_end_card_path(self):
        with patch.object(reelsfy.subprocess, "call", return_value=0) as call_mock, patch(
            "os.path.exists", return_value=True
        ), patch("os.path.getsize", return_value=64):
            result = reelsfy.generate_end_card()

        assert result == "tmp/end-card.mp4"
        assert "tmp/end-card.mp4" in call_mock.call_args[0][0]

    def test_failure_returns_none(self):
        with patch.object(reelsfy.subprocess, "call", return_value=1), patch(
            "os.path.exists", return_value=True
        ), patch("os.path.getsize", return_value=64):
            result = reelsfy.generate_end_card()

        assert result is None


class TestEndCardConcatFullFallback:
    """P3.2: last ladder rung for end-card assemblies only."""

    def test_full_fallback_reencodes_video_and_audio_without_mastering(self):
        command = reelsfy.build_teaser_concat_full_fallback_command(
            "tmp/hook-concat.txt", "/out/final-x.mp4"
        )

        assert "-f concat" in command
        assert "-safe 0" in command
        assert "h264_nvenc" in command
        assert '-c:a aac -b:a 192k "/out/final-x.mp4"' in command
        # Never re-master audio over the concatenation
        assert reelsfy.AUDIO_MASTERING_FILTER not in command


class TestAssembleHookTeaserFinalEndCard:
    """P3.2: the concat list gains the end card; ladder is reused."""

    def _run(self, tmp_path, subprocess_side_effect, end_card_path=None):
        concat_list = tmp_path / "hook-concat.txt"
        published = tmp_path / "out" / "final-test_cropped000.mp4"
        (tmp_path / "out").mkdir(exist_ok=True)
        published.write_bytes(b"\x00" * 16)

        call_mock = MagicMock(side_effect=subprocess_side_effect)
        with patch.object(reelsfy.subprocess, "call", call_mock):
            result = reelsfy.assemble_hook_teaser_final(
                "tmp/hook-teaser.mp4",
                "tmp/hook-main.mp4",
                str(published),
                concat_list_path=str(concat_list),
                end_card_path=end_card_path,
            )

        return result, call_mock, concat_list

    def _concat_entries(self, concat_list):
        return [
            line[len("file '") : -len("'")]
            for line in concat_list.read_text().splitlines()
            if line.startswith("file '")
        ]

    def test_end_card_appended_after_main(self, tmp_path):
        result, call_mock, concat_list = self._run(tmp_path, [0], end_card_path="tmp/end-card.mp4")

        assert result is True
        entries = self._concat_entries(concat_list)
        assert len(entries) == 3
        assert entries[0].endswith("tmp/hook-teaser.mp4")
        assert entries[1].endswith("tmp/hook-main.mp4")
        assert entries[2].endswith("tmp/end-card.mp4")
        # First ladder rung is unchanged: -c copy
        assert "-c copy" in call_mock.call_args_list[0][0][0]

    def test_no_end_card_keeps_legacy_two_entry_list(self, tmp_path):
        result, _, concat_list = self._run(tmp_path, [0])

        assert result is True
        entries = self._concat_entries(concat_list)
        assert len(entries) == 2
        assert entries[1].endswith("tmp/hook-main.mp4")

    def test_end_card_ladder_video_reencode_then_full_reencode(self, tmp_path):
        result, call_mock, _ = self._run(tmp_path, [1, 1, 0], end_card_path="tmp/end-card.mp4")

        assert result is True
        first, second, third = [c[0][0] for c in call_mock.call_args_list]
        assert "-c copy" in first
        assert "h264_nvenc" in second
        assert "-c:a copy" in second
        assert reelsfy.AUDIO_MASTERING_FILTER not in second
        assert "h264_nvenc" in third
        assert "-c:a aac -b:a 192k" in third
        assert reelsfy.AUDIO_MASTERING_FILTER not in third

    def test_end_card_ladder_all_fail_returns_false(self, tmp_path):
        result, call_mock, _ = self._run(tmp_path, [1, 1, 1], end_card_path="tmp/end-card.mp4")

        assert result is False
        assert call_mock.call_count == 3

    def test_no_end_card_keeps_two_rung_ladder(self, tmp_path):
        # Legacy (byte-identical) assembly: no third attempt exists.
        result, call_mock, _ = self._run(tmp_path, [1, 1])

        assert result is False
        assert call_mock.call_count == 2


class TestPublishSegmentEndCard:
    """P3.2: end card default ON; disabled path is byte-identical legacy."""

    def _run(
        self,
        monkeypatch,
        hook=None,
        end_card_enabled=True,
        end_card_path="tmp/end-card.mp4",
        assemble_return=True,
        generate_subtitle_side_effect=None,
    ):
        _clear_polish_env(monkeypatch)
        if not end_card_enabled:
            monkeypatch.setenv("END_CARD_ENABLED", "false")

        generate_subtitle_mock = MagicMock(
            side_effect=generate_subtitle_side_effect or [True, True]
        )
        assemble_mock = MagicMock(return_value=assemble_return)
        end_card_mock = MagicMock(return_value=end_card_path)

        with patch.object(
            reelsfy, "generate_subtitle", generate_subtitle_mock
        ), patch.object(
            reelsfy, "generate_hook_teaser", return_value="tmp/hook-teaser.mp4"
        ), patch.object(
            reelsfy, "generate_end_card", end_card_mock
        ), patch.object(
            reelsfy, "assemble_hook_teaser_final", assemble_mock
        ), patch(
            "os.path.exists", return_value=True
        ), patch(
            "os.path.getsize", return_value=64
        ):
            reelsfy.publish_segment_with_optional_teaser(
                "final-test_cropped000.mp4",
                "vid",
                "/out",
                video_title="Title",
                subtitle_words=list(SAMPLE_WORDS),
                hook=hook,
            )

        return generate_subtitle_mock, assemble_mock, end_card_mock

    def test_end_card_enabled_no_hook_assembles_main_plus_end_card(self, monkeypatch):
        generate_subtitle_mock, assemble_mock, end_card_mock = self._run(
            monkeypatch, hook=None
        )

        end_card_mock.assert_called_once()
        assert generate_subtitle_mock.call_count == 1
        call = generate_subtitle_mock.call_args
        assert call.kwargs.get("main_output_path") == "tmp/hook-main.mp4"
        assemble_mock.assert_called_once()
        args = assemble_mock.call_args
        assert args[0][0] is None  # no teaser
        assert args[0][1] == "tmp/hook-main.mp4"
        assert args[1].get("end_card_path") == "tmp/end-card.mp4"

    def test_end_card_disabled_no_hook_is_direct_publish(self, monkeypatch):
        generate_subtitle_mock, assemble_mock, end_card_mock = self._run(
            monkeypatch, hook=None, end_card_enabled=False
        )

        end_card_mock.assert_not_called()
        assemble_mock.assert_not_called()
        assert generate_subtitle_mock.call_count == 1
        call = generate_subtitle_mock.call_args
        # Direct publish: identical to the pre-end-card path
        assert call.args[:3] == ("final-test_cropped000.mp4", "vid", "/out")
        assert not call.kwargs.get("main_output_path")

    def test_end_card_disabled_with_hook_keeps_legacy_assembly(self, monkeypatch):
        generate_subtitle_mock, assemble_mock, end_card_mock = self._run(
            monkeypatch,
            hook=(10.5, 14.0, "THIS CHANGED EVERYTHING"),
            end_card_enabled=False,
        )

        end_card_mock.assert_not_called()
        assemble_mock.assert_called_once()
        assert assemble_mock.call_args[1].get("end_card_path") is None
        assert generate_subtitle_mock.call_args.kwargs.get(
            "main_output_path"
        ) == "tmp/hook-main.mp4"

    def test_end_card_build_failure_falls_back_to_direct_publish(self, monkeypatch):
        generate_subtitle_mock, assemble_mock, end_card_mock = self._run(
            monkeypatch, hook=None, end_card_path=None
        )

        end_card_mock.assert_called_once()
        assemble_mock.assert_not_called()
        assert generate_subtitle_mock.call_count == 1
        call = generate_subtitle_mock.call_args
        assert call.args[:3] == ("final-test_cropped000.mp4", "vid", "/out")
        assert not call.kwargs.get("main_output_path")

    def test_assembly_failure_still_publishes_directly(self, monkeypatch):
        generate_subtitle_mock, assemble_mock, _ = self._run(
            monkeypatch, hook=None, assemble_return=False
        )

        assert assemble_mock.called
        assert generate_subtitle_mock.call_count == 2
        second_call = generate_subtitle_mock.call_args_list[1]
        assert second_call.args[:3] == ("final-test_cropped000.mp4", "vid", "/out")
        assert not second_call.kwargs.get("main_output_path")
