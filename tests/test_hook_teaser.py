"""
Hook teaser tests for Story 7 (P2.2): cold-open hook teaser in reelsfy.py.

Covers:
1. parse_hook_fields validation: all three fields required, SRT timestamps
   via parse_srt_time, the hook teaser duration band, inside the segment, end > start.
2. generate_viral prompt/json template paths mention the hook fields
   (string assertions only - no live LLM call).
3. Teaser pipeline: cut command timestamps (NVENC style), hook drawtext
   (fontsize=64, centered, white with black border), AUDIO_MASTERING_FILTER
   applied to the teaser stream exactly once, concat-demuxer `-c copy`
   assembly with a video-re-encode fallback that never re-masters audio.
4. generate_subtitle main_output_path passthrough; default path unchanged
   (byte-identical no-teaser behavior) and published `final-*` naming intact.

All subprocess/cv2/LLM calls are mocked - no real video is processed.
"""

import inspect
import json
import os
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


REELSFY_SOURCE = (REELSFY_DIR / "reelsfy.py").read_text()

SAMPLE_WORDS = [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "world", "start": 0.5, "end": 1.0},
]


def _hook_segment(
    hook_start="00:00:05,000",
    hook_end="00:00:08,000",
    hook_text="This changed everything",
):
    return {
        "start_time": "00:00:00,000",
        "end_time": "00:00:35,000",
        "title": "Test Segment",
        "hook_start_time": hook_start,
        "hook_end_time": hook_end,
        "hook_text": hook_text,
    }


def _run_generate_subtitle(tmp_path, monkeypatch, main_output_path=None, subtitle_words=None):
    """
    Run generate_subtitle in an isolated tmp cwd with mocked subprocess.
    Uses the karaoke path (subtitle_words) so no transcription is needed.
    Returns (result, subprocess_call_mock).
    """
    workdir = tmp_path / "work"
    (workdir / "tmp").mkdir(parents=True)
    output_dir = workdir / "output"
    output_dir.mkdir()
    monkeypatch.chdir(workdir)

    with patch.object(
        reelsfy.subprocess, "call", return_value=0
    ) as call_mock:
        result = reelsfy.generate_subtitle(
            "final-test_cropped000.mp4",
            "vid",
            str(output_dir),
            subtitle_words=subtitle_words or SAMPLE_WORDS,
            main_output_path=main_output_path,
        )

    return result, call_mock


def _commands_called(call_mock):
    """Extract all shell command strings passed to subprocess.call."""
    return [args[0][0] for args in call_mock.call_args_list]


class TestParseHookFields:
    """Requirement 2: pure hook parsing/validation helper."""

    def test_valid_hook_fields_return_tuple(self):
        result = reelsfy.parse_hook_fields(_hook_segment(), 0.0, 35.0)

        assert result is not None
        hook_start, hook_end, hook_text = result
        assert hook_start == 5.0
        assert hook_end == 8.0
        assert hook_text == "This changed everything"

    def test_missing_hook_start_returns_none(self):
        segment = _hook_segment()
        del segment["hook_start_time"]

        assert reelsfy.parse_hook_fields(segment, 0.0, 35.0) is None

    def test_missing_hook_end_returns_none(self):
        segment = _hook_segment()
        del segment["hook_end_time"]

        assert reelsfy.parse_hook_fields(segment, 0.0, 35.0) is None

    def test_missing_hook_text_returns_none(self):
        segment = _hook_segment()
        del segment["hook_text"]

        assert reelsfy.parse_hook_fields(segment, 0.0, 35.0) is None

    def test_empty_hook_text_returns_none(self):
        segment = _hook_segment(hook_text="   ")

        assert reelsfy.parse_hook_fields(segment, 0.0, 35.0) is None

    def test_range_shorter_than_1_5s_returns_none(self):
        # 1.0 s span: below the 2.5 s minimum.
        segment = _hook_segment(
            hook_start="00:00:05,000", hook_end="00:00:06,000"
        )

        assert reelsfy.parse_hook_fields(segment, 0.0, 35.0) is None

    def test_range_longer_than_9s_returns_none(self):
        # 9.5 s span: above the 9 s maximum.
        segment = _hook_segment(
            hook_start="00:00:05,000", hook_end="00:00:14,500"
        )

        assert reelsfy.parse_hook_fields(segment, 0.0, 35.0) is None

    def test_end_le_start_returns_none(self):
        segment = _hook_segment(
            hook_start="00:00:08,000", hook_end="00:00:08,000"
        )

        assert reelsfy.parse_hook_fields(segment, 0.0, 35.0) is None

    def test_hook_start_before_segment_returns_none(self):
        segment = _hook_segment(
            hook_start="00:00:00,500", hook_end="00:00:03,500"
        )

        assert reelsfy.parse_hook_fields(segment, 1.0, 35.0) is None

    def test_hook_end_after_segment_returns_none(self):
        segment = _hook_segment(
            hook_start="00:00:34,000", hook_end="00:00:36,000"
        )

        assert reelsfy.parse_hook_fields(segment, 0.0, 35.0) is None

    def test_invalid_timestamp_returns_none(self):
        segment = _hook_segment(hook_start="not-a-timestamp")

        assert reelsfy.parse_hook_fields(segment, 0.0, 35.0) is None

    def test_non_dict_segment_returns_none(self):
        assert reelsfy.parse_hook_fields(None, 0.0, 35.0) is None


class TestPromptHookFields:
    """Requirement 1: hook fields are requested in BOTH GPT prompt paths."""

    def test_json_template_mentions_hook_fields(self):
        template = reelsfy.build_segment_json_template()

        assert "hook_start_time" in template
        assert "hook_end_time" in template
        assert "hook_text" in template

    def test_json_template_is_valid_json(self):
        template = reelsfy.build_segment_json_template()

        parsed = json.loads(template)
        assert "segments" in parsed
        assert "hook_start_time" in parsed["segments"][0]

    def test_json_template_values_have_no_parentheticals(self):
        # Example values must be plain: a parenthetical like
        # "00:00:05,000 (optional; ...)" echoed by GPT would be rejected by
        # parse_srt_time and the teaser would silently vanish.
        template = reelsfy.build_segment_json_template()

        assert "(" not in template
        assert ")" not in template

    def test_fallback_viral_prompt_mentions_hook_fields(self):
        # The fallback prompt is built inline in generate_viral; verify the
        # instruction lines exist there without making an LLM call.
        source = inspect.getsource(reelsfy.generate_viral)

        assert "hook_start_time" in source
        assert "hook_end_time" in source
        assert "hook_text" in source

    def test_generate_viral_uses_shared_json_template(self):
        source = inspect.getsource(reelsfy.generate_viral)

        assert "build_segment_json_template()" in source


class TestTeaserCommandBuilders:
    """Requirement 3: pure builders for the teaser ffmpeg shell commands."""

    def test_cut_command_uses_hook_timestamps(self):
        command = reelsfy.build_teaser_cut_command(10.5, 14.0)

        assert "-ss 10.5" in command
        assert "-to 14.0" in command
        assert "tmp/input_video.mp4" in command
        assert command.rstrip().endswith("tmp/hook-teaser-raw.mp4")

    def test_cut_command_uses_nvenc_settings(self):
        command = reelsfy.build_teaser_cut_command(10.5, 14.0)

        assert "h264_nvenc" in command
        assert "-c:a aac -b:a 192k" in command

    def test_cut_command_does_not_master_audio(self):
        # Mastering invariant: the mastering chain only applies in the final
        # burn, never in the intermediate cut.
        command = reelsfy.build_teaser_cut_command(10.5, 14.0)

        assert reelsfy.AUDIO_MASTERING_FILTER not in command

    def test_drawtext_filter_styles_hook_text(self):
        hook_filter = reelsfy.build_hook_drawtext_filter("This changed everything")

        assert "fontsize=64" in hook_filter
        assert "fontcolor=white" in hook_filter
        # One centered drawtext per wrapped line
        assert hook_filter.count("x=(w-tw)/2") == 2
        assert f"fontfile={reelsfy.OVERLAY_FONTFILE}" in hook_filter
        # Black border/outline around the text
        assert "bordercolor=black" in hook_filter
        # Text is uppercased and wrapped so each line fits the frame
        assert "THIS CHANGED" in hook_filter
        assert "EVERYTHING" in hook_filter

    def test_wrap_hook_text_splits_long_sentences(self):
        lines = reelsfy.wrap_hook_text(
            "THIS ONE SMALL CHANGE MADE EVERYTHING DIFFERENT"
        )

        assert all(len(line) <= reelsfy.HOOK_TEXT_MAX_LINE_CHARS for line in lines)
        assert len(lines) >= 3

    def test_wrap_hook_text_never_returns_empty(self):
        assert reelsfy.wrap_hook_text("SINGLE") == ["SINGLE"]
        assert reelsfy.wrap_hook_text("") == []

    def test_drawtext_lines_fit_frame_width(self):
        # Worst case: every line at max chars must stay inside 1080 px
        # (fontsize 64 uppercase ≈ 40 px/char → 18 chars ≈ 720 px).
        hook_filter = reelsfy.build_hook_drawtext_filter(
            "this is a very long hook sentence that keeps going on"
        )
        drawtext_texts = [
            part for part in hook_filter.split("drawtext=text='")
            if part
        ]
        for part in drawtext_texts:
            line = part[: part.index("'")]
            assert len(line) <= reelsfy.HOOK_TEXT_MAX_LINE_CHARS

    def test_sanitize_hook_text_strips_apostrophes(self):
        sanitized = reelsfy.sanitize_hook_text("Don't stop")

        assert "'" not in sanitized
        assert "DONT STOP" == sanitized

    def test_sanitize_hook_text_escapes_colon_and_percent(self):
        sanitized = reelsfy.sanitize_hook_text("50:50 at 100%")

        assert "\\:" in sanitized
        assert "\\%" in sanitized

    def test_burn_command_applies_drawtext_and_mastering(self):
        command = reelsfy.build_teaser_burn_command("This changed everything")

        assert "tmp/final-hook-teaser.mp4" in command
        assert command.rstrip().endswith("tmp/hook-teaser.mp4")
        # Hook text burned with the drawtext filter
        assert "drawtext=" in command
        assert "fontsize=64" in command
        # Mastering applied exactly once for the teaser stream
        assert f"-af \"{reelsfy.AUDIO_MASTERING_FILTER}\"" in command
        assert "-c:a aac -b:a 192k" in command
        assert "h264_nvenc" in command

    def test_concat_command_uses_copy_codec(self):
        command = reelsfy.build_teaser_concat_command(
            "tmp/hook-concat.txt", "/out/final-test_cropped000.mp4"
        )

        assert "-f concat" in command
        assert "-safe 0" in command
        assert "-c copy" in command
        assert '"/out/final-test_cropped000.mp4"' in command

    def test_concat_fallback_reencodes_video_copies_audio(self):
        command = reelsfy.build_teaser_concat_fallback_command(
            "tmp/hook-concat.txt", "/out/final-test_cropped000.mp4"
        )

        assert "h264_nvenc" in command
        # Audio is never re-mastered over the concatenation
        assert "-c:a copy" in command
        assert reelsfy.AUDIO_MASTERING_FILTER not in command
        assert "-c copy" not in command
        assert '"/out/final-test_cropped000.mp4"' in command


class TestGenerateHookTeaser:
    """Requirement 3: cut -> crop (generate_short) -> burn orchestration."""

    def _run(self, subprocess_side_effect=None, generate_short_return=None):
        """Run generate_hook_teaser with mocked subprocess and generate_short."""
        call_mock = MagicMock()
        if subprocess_side_effect is not None:
            call_mock.side_effect = subprocess_side_effect
        else:
            call_mock.return_value = 0

        exists_mock = MagicMock(return_value=True)
        getsize_mock = MagicMock(return_value=64)

        with patch.object(reelsfy.subprocess, "call", call_mock), patch.object(
            reelsfy, "generate_short", return_value=generate_short_return
        ) as short_mock, patch(
            "os.path.exists", exists_mock
        ), patch(
            "os.path.getsize", getsize_mock
        ):
            result = reelsfy.generate_hook_teaser(
                10.5, 14.0, "This changed everything"
            )

        return result, short_mock, call_mock

    def test_success_returns_teaser_path(self):
        result, short_mock, _ = self._run(
            generate_short_return="tmp/final-hook-teaser.mp4"
        )

        assert result == "tmp/hook-teaser.mp4"
        # The cropped 9:16 teaser comes from generate_short on the raw cut
        short_mock.assert_called_once_with("hook-teaser-raw.mp4", "hook-teaser.mp4")

    def test_cut_failure_skips_teaser(self):
        result, short_mock, _ = self._run(
            subprocess_side_effect=[1], generate_short_return="tmp/final-hook-teaser.mp4"
        )

        assert result is None
        short_mock.assert_not_called()

    def test_crop_failure_skips_teaser(self):
        result, short_mock, call_mock = self._run(
            generate_short_return=None
        )

        assert result is None
        # Only the cut command ran; the burn step never runs
        assert call_mock.call_count == 1

    def test_burn_failure_skips_teaser(self):
        result, _, call_mock = self._run(
            subprocess_side_effect=[0, 1],
            generate_short_return="tmp/final-hook-teaser.mp4",
        )

        assert result is None
        assert call_mock.call_count == 2


class TestGenerateSubtitleMainOutputPath:
    """Requirement 3.5: main_output_path passthrough with unchanged default."""

    def test_main_output_path_used_when_provided(self, tmp_path, monkeypatch):
        _, call_mock = _run_generate_subtitle(
            tmp_path, monkeypatch, main_output_path="tmp/hook-main.mp4"
        )

        commands = _commands_called(call_mock)
        final_cmds = [c for c in commands if "final-test_cropped000.mp4" in c or "hook-main.mp4" in c]
        assert final_cmds
        assert any('tmp/hook-main.mp4"' in c for c in final_cmds)
        # The burned main file is NOT written to the published output dir
        assert not any(
            f"{tmp_path / 'work' / 'output'}" in c and "hook-main" in c
            for c in commands
        )

    def test_default_output_path_unchanged(self, tmp_path, monkeypatch):
        _, call_mock = _run_generate_subtitle(tmp_path, monkeypatch)

        commands = _commands_called(call_mock)
        published = [
            c for c in commands if f"{tmp_path / 'work' / 'output'}" in c
        ]
        assert published
        assert any(
            f'"{tmp_path / "work" / "output"}/final-test_cropped000.mp4"' in c
            for c in published
        )

    def test_main_output_path_is_optional_keyword(self):
        signature = inspect.signature(reelsfy.generate_subtitle)

        assert "main_output_path" in signature.parameters
        assert (
            signature.parameters["main_output_path"].default is None
        )


class TestPublishedNaming:
    """Requirement 3.6: published filename keeps matching publisher.py."""

    def test_published_path_matches_publisher_globs(self):
        path = reelsfy.compute_published_output_path(
            "/out", "final-test_cropped000.mp4"
        )

        assert path == "/out/final-test_cropped000.mp4"
        assert path.endswith(".mp4")
        # Matches publisher.py's final_video_*.mp4 / final-*.mp4 fallback glob
        assert Path(path).name.startswith("final-")

    def test_published_path_sanitizes_titles(self):
        path = reelsfy.compute_published_output_path(
            "/out", "final-my: title?_cropped000.mp4"
        )

        assert ":" not in Path(path).name
        assert "?" not in Path(path).name

    def test_generate_subtitle_uses_shared_published_path_helper(self):
        # generate_subtitle and the loop must derive the SAME published name.
        assert "compute_published_output_path" in inspect.getsource(
            reelsfy.generate_subtitle
        )


class TestAssembleHookTeaserFinal:
    """Requirement 3.6: concat assembly with re-encode fallback."""

    def _run(self, tmp_path, subprocess_side_effect):
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
            )

        return result, call_mock, concat_list

    def test_concat_copy_success(self, tmp_path):
        result, call_mock, concat_list = self._run(tmp_path, [0])

        assert result is True
        list_content = concat_list.read_text()
        entries = [
            line[len("file '") : -len("'")]
            for line in list_content.splitlines()
            if line.startswith("file '")
        ]
        # ffmpeg's concat demuxer resolves entry paths against the list
        # file's directory, so entries MUST be absolute.
        assert len(entries) == 2
        for entry in entries:
            assert os.path.isabs(entry), entry
        assert entries[0].endswith("tmp/hook-teaser.mp4")
        assert entries[1].endswith("tmp/hook-main.mp4")
        command = call_mock.call_args[0][0]
        assert "-f concat" in command
        assert "-c copy" in command

    def test_concat_failure_falls_back_to_reencode(self, tmp_path):
        result, call_mock, _ = self._run(tmp_path, [1, 0])

        assert result is True
        fallback_command = call_mock.call_args_list[1][0][0]
        assert "h264_nvenc" in fallback_command
        assert "-c:a copy" in fallback_command
        assert reelsfy.AUDIO_MASTERING_FILTER not in fallback_command

    def test_both_steps_fail_returns_false(self, tmp_path):
        result, _, _ = self._run(tmp_path, [1, 1])

        assert result is False


class TestPublishSegmentWithOptionalTeaser:
    """Requirement 3.7: every teaser failure mode still publishes the video.

    The published path must always be derived exactly like the no-teaser
    case (generate_subtitle direct write), so a failed assembly never loses
    the segment's output file.
    """

    HOOK = (10.5, 14.0, "THIS CHANGED EVERYTHING")

    def _run(
        self,
        hook=HOOK,
        generate_subtitle_side_effect=None,
        teaser_path="tmp/hook-teaser.mp4",
        assemble_return=False,
        env=None,
    ):
        """Run publish_segment_with_optional_teaser with the teaser helpers
        mocked. Returns (generate_subtitle_mock, assemble_mock)."""
        generate_subtitle_mock = MagicMock(
            side_effect=generate_subtitle_side_effect or [True, True]
        )
        assemble_mock = MagicMock(return_value=assemble_return)

        with patch.object(
            reelsfy, "generate_subtitle", generate_subtitle_mock
        ), patch.object(
            reelsfy, "generate_hook_teaser", return_value=teaser_path
        ), patch.object(
            reelsfy, "assemble_hook_teaser_final", assemble_mock
        ), patch.object(
            # Story 10: keep the default-ON end card hermetic (no real ffmpeg)
            reelsfy, "generate_end_card", MagicMock(return_value="tmp/end-card.mp4")
        ), patch.dict(
            os.environ, env or {}
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
                subtitle_words=SAMPLE_WORDS,
                hook=hook,
            )

        return generate_subtitle_mock, assemble_mock

    def _is_direct_publish_call(self, call):
        """A direct-publish call: no main_output_path (no-teaser behavior)."""
        args = call.args
        kwargs = call.kwargs
        return (
            args[:3] == ("final-test_cropped000.mp4", "vid", "/out")
            and not kwargs.get("main_output_path")
            and kwargs.get("video_title") == "Title"
            and kwargs.get("subtitle_words") == SAMPLE_WORDS
        )

    def test_no_hook_publishes_directly(self):
        # No hook fields AND the end card disabled: exactly the legacy
        # no-teaser behavior (Story 10's end card defaults ON and is covered
        # by tests/test_polish.py).
        generate_subtitle_mock, assemble_mock = self._run(
            hook=None, env={"END_CARD_ENABLED": "false"}
        )

        assert generate_subtitle_mock.call_count == 1
        assert self._is_direct_publish_call(generate_subtitle_mock.call_args)
        assemble_mock.assert_not_called()

    def test_teaser_success_single_main_write_and_assembly(self):
        generate_subtitle_mock, assemble_mock = self._run(
            assemble_return=True
        )

        assert generate_subtitle_mock.call_count == 1
        call = generate_subtitle_mock.call_args
        assert call.kwargs.get("main_output_path") == "tmp/hook-main.mp4"
        assemble_mock.assert_called_once()

    def test_assembly_failure_falls_back_to_direct_publish(self):
        # Both concat attempts fail -> the segment must still be published
        # through the direct (no-teaser) generate_subtitle path.
        generate_subtitle_mock, assemble_mock = self._run(assemble_return=False)

        assert generate_subtitle_mock.call_count == 2
        first_call, second_call = generate_subtitle_mock.call_args_list
        # First call: burned main written to tmp for the assembly
        assert first_call.kwargs.get("main_output_path") == "tmp/hook-main.mp4"
        # Second call: direct publish, identical to the no-teaser path
        assert self._is_direct_publish_call(second_call)

    def test_missing_burned_main_falls_back_to_direct_publish(self):
        generate_subtitle_mock, assemble_mock = self._run(
            generate_subtitle_side_effect=[False, True], assemble_return=True
        )

        assert generate_subtitle_mock.call_count == 2
        first_call, second_call = generate_subtitle_mock.call_args_list
        assert first_call.kwargs.get("main_output_path") == "tmp/hook-main.mp4"
        assert self._is_direct_publish_call(second_call)
        # No assembly was attempted without a burned main file
        assemble_mock.assert_not_called()

    def test_teaser_generation_failure_publishes_directly(self):
        # Legacy degradation (end card disabled): a failed teaser still
        # publishes the video through the direct generate_subtitle path.
        generate_subtitle_mock, assemble_mock = self._run(
            teaser_path=None, env={"END_CARD_ENABLED": "false"}
        )

        assert generate_subtitle_mock.call_count == 1
        assert self._is_direct_publish_call(generate_subtitle_mock.call_args)
        assemble_mock.assert_not_called()

    def test_direct_publish_uses_published_path_derivation(self):
        # The loop no longer derives the published path for concat only:
        # the helper must target the same file publisher.py globs.
        assert (
            "compute_published_output_path"
            in inspect.getsource(reelsfy.publish_segment_with_optional_teaser)
        )


class TestTeaserDuplicatesClipOpeningHelper:
    """B4: the pure overlap helper (reuses HOOK_DUPLICATE_GUARD_SECONDS)."""

    def test_hook_at_the_clip_opening_is_a_duplicate(self):
        assert reelsfy.teaser_duplicates_clip_opening(0.0, 0.0, 3.0) is True

    def test_hook_entirely_before_the_clip_is_not_a_duplicate(self):
        # A teaser that ends before the clip even starts cannot replay it.
        assert reelsfy.teaser_duplicates_clip_opening(10.0, 4.0, 9.5) is False

    def test_hook_far_inside_the_clip_is_not_a_duplicate(self):
        assert reelsfy.teaser_duplicates_clip_opening(0.0, 20.0, 23.0) is False

    def test_hook_starting_exactly_at_the_guard_edge_is_not_a_duplicate(self):
        # clip_start + HOOK_DUPLICATE_GUARD_SECONDS is just OUTSIDE the guard.
        assert reelsfy.teaser_duplicates_clip_opening(0.0, 5.0, 8.0) is False

    def test_hook_straddling_the_guard_edge_is_a_duplicate(self):
        assert reelsfy.teaser_duplicates_clip_opening(0.0, 3.0, 9.0) is True

    def test_guard_argument_overrides_the_default(self):
        # guard=2 s keeps the 0-3 s hook inside the opening...
        assert reelsfy.teaser_duplicates_clip_opening(0.0, 0.0, 3.0, guard=2.0) is True
        # ...and a 12-15 s hook only clears a 10 s opening.
        assert reelsfy.teaser_duplicates_clip_opening(0.0, 12.0, 15.0, guard=10.0) is False

    def test_none_clip_start_means_no_overlap_check(self):
        # The inert case: no clip start -> never a duplicate (legacy callers).
        assert reelsfy.teaser_duplicates_clip_opening(None, 0.0, 3.0) is False

    def test_unparseable_times_are_not_a_duplicate(self):
        assert reelsfy.teaser_duplicates_clip_opening("nope", 0.0, 3.0) is False
        assert reelsfy.teaser_duplicates_clip_opening(0.0, "nope", None) is False


class TestTeaserRepeatsFirstCaptionHelper:
    """B4: the tolerant caption-sentence comparison (second net)."""

    WORDS = [
        {"word": "This", "start": 0.0, "end": 0.4},
        {"word": "changed", "start": 0.4, "end": 0.9},
        {"word": "everything.", "start": 0.9, "end": 1.6},
        {"word": "Then", "start": 1.6, "end": 1.8},
        {"word": "it", "start": 1.8, "end": 2.0},
        {"word": "broke.", "start": 2.0, "end": 2.4},
    ]

    def test_matching_text_is_a_repeat(self):
        assert (
            reelsfy.teaser_repeats_first_caption("This changed everything.", self.WORDS)
            is True
        )

    def test_casing_and_trailing_punctuation_are_normalized(self):
        assert (
            reelsfy.teaser_repeats_first_caption("this CHANGED everything!!!", self.WORDS)
            is True
        )

    def test_different_text_is_not_a_repeat(self):
        assert (
            reelsfy.teaser_repeats_first_caption("A totally different line", self.WORDS)
            is False
        )

    def test_no_caption_words_is_inert(self):
        assert reelsfy.teaser_repeats_first_caption("This changed everything.", []) is False
        assert reelsfy.teaser_repeats_first_caption("This changed everything.", None) is False

    def test_empty_hook_text_is_inert(self):
        assert reelsfy.teaser_repeats_first_caption("", self.WORDS) is False
        assert reelsfy.teaser_repeats_first_caption(None, self.WORDS) is False

    def test_sentence_after_the_opening_is_not_a_repeat(self):
        # The later sentence is INSIDE the 5 s window but does not open it, so
        # the match is anchored at the start of the window, not anywhere in it.
        assert reelsfy.teaser_repeats_first_caption("Then it broke.", self.WORDS) is False

    def test_hook_repeating_the_opening_words_in_pieces_is_a_repeat(self):
        # Real screencast captions rarely carry a terminator on the opening
        # line ("we set version 2." -> fragment), so a terminator-delimited
        # sentence can never equal a hook that restates the opening mid-clip.
        # The net is time-based: any hook sharing the opening's first words
        # counts, no terminal punctuation required on either side.
        punct_less = [
            {"word": "we", "start": 0.0, "end": 0.2},
            {"word": "set", "start": 0.2, "end": 0.5},
            {"word": "the", "start": 0.5, "end": 0.7},
            {"word": "version", "start": 0.7, "end": 1.1},
            {"word": "number", "start": 1.1, "end": 1.6},
            {"word": "two", "start": 1.6, "end": 1.9},
            {"word": "and", "start": 30.0, "end": 30.2},
            {"word": "moved", "start": 30.2, "end": 30.6},
        ]
        assert (
            reelsfy.teaser_repeats_first_caption("We set the version number", punct_less)
            is True
        )
        # A different mid-clip teaser that shares nothing with the opening stays.
        assert reelsfy.teaser_repeats_first_caption("A totally different line", punct_less) is False

    def test_one_word_overlap_is_not_enough(self):
        # Conservative floor: a single shared leading word is coincidental
        # phrasing, and a false match costs a teaser that was fine.
        short = [
            {"word": "we", "start": 0.0, "end": 0.2},
            {"word": "set", "start": 0.2, "end": 0.5},
            {"word": "the", "start": 0.5, "end": 0.7},
            {"word": "version", "start": 0.7, "end": 1.1},
        ]
        assert reelsfy.teaser_repeats_first_caption("We need", short) is False
        assert reelsfy.teaser_repeats_first_caption("We set", short) is True

    def test_formulaic_opening_bigram_is_not_a_repeat(self):
        # Screencast captions open with formulaic bigrams ("so today", "in
        # this video"). A genuinely different teaser that happens to open with
        # the same two words must NOT be dropped: the shared prefix has to
        # cover a substantial fraction of the shorter side.
        words = [
            {"word": "So", "start": 0.0, "end": 0.3},
            {"word": "today", "start": 0.3, "end": 0.6},
            {"word": "we", "start": 0.6, "end": 0.7},
            {"word": "ship", "start": 0.7, "end": 1.0},
            {"word": "the", "start": 1.0, "end": 1.1},
            {"word": "pipeline.", "start": 1.1, "end": 1.6},
        ]
        assert (
            reelsfy.teaser_repeats_first_caption(
                "So today the retry limit shipped", words
            )
            is False
        )
        # A teaser that really does restate the opening still trips.
        assert (
            reelsfy.teaser_repeats_first_caption("So today we ship the", words)
            is True
        )

    def test_punctuation_only_opening_does_not_fake_a_match(self):
        # "…" and "--" normalize to empty tokens; two of them aligned at
        # positions 0 and 1 must not count as two shared words.
        punct_only = [
            {"word": "…", "start": 0.0, "end": 0.2},
            {"word": "--", "start": 0.2, "end": 0.5},
        ]
        assert reelsfy.teaser_repeats_first_caption("… --", punct_only) is False
        assert (
            reelsfy.teaser_repeats_first_caption("… -- but the build broke", punct_only)
            is False
        )


class TestPublishHookDedupGuard:
    """B4: the publish step refuses a teaser that repeats the clip opening."""

    def _run(
        self,
        hook,
        clip_start_time,
        subtitle_words=None,
        assemble_return=True,
        end_card_enabled="false",
    ):
        """Run publish_segment_with_optional_teaser with the teaser helpers
        mocked. Returns (teaser_mock, generate_subtitle_mock, assemble_mock)."""
        teaser_mock = MagicMock(return_value="tmp/hook-teaser.mp4")
        generate_subtitle_mock = MagicMock(side_effect=[True, True])
        assemble_mock = MagicMock(return_value=assemble_return)

        with patch.object(
            reelsfy, "generate_subtitle", generate_subtitle_mock
        ), patch.object(
            reelsfy, "generate_hook_teaser", teaser_mock
        ), patch.object(
            reelsfy, "assemble_hook_teaser_final", assemble_mock
        ), patch.object(
            reelsfy, "generate_end_card", MagicMock(return_value="tmp/end-card.mp4")
        ), patch.dict(
            os.environ, {"END_CARD_ENABLED": end_card_enabled}
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
                subtitle_words=subtitle_words if subtitle_words is not None else SAMPLE_WORDS,
                hook=hook,
                clip_start_time=clip_start_time,
            )

        return teaser_mock, generate_subtitle_mock, assemble_mock

    def _is_direct_publish_call(self, call):
        args = call.args
        kwargs = call.kwargs
        return (
            args[:3] == ("final-test_cropped000.mp4", "vid", "/out")
            and not kwargs.get("main_output_path")
            and kwargs.get("video_title") == "Title"
        )

    def test_teaser_at_the_clip_opening_is_skipped(self):
        teaser_mock, generate_subtitle_mock, assemble_mock = self._run(
            hook=(0.0, 3.0, "A teaser"),
            clip_start_time=0.0,
        )

        teaser_mock.assert_not_called()
        assert generate_subtitle_mock.call_count == 1
        assert self._is_direct_publish_call(generate_subtitle_mock.call_args)
        assemble_mock.assert_not_called()

    def test_teaser_overlapping_only_the_first_five_seconds_is_skipped(self):
        # 3 s into a clip that starts at 100 s: still inside the guard.
        teaser_mock, generate_subtitle_mock, assemble_mock = self._run(
            hook=(103.0, 106.0, "A teaser"),
            clip_start_time=100.0,
        )

        teaser_mock.assert_not_called()
        assert generate_subtitle_mock.call_count == 1
        assert self._is_direct_publish_call(generate_subtitle_mock.call_args)
        assemble_mock.assert_not_called()

    def test_non_overlapping_teaser_still_assembles(self):
        teaser_mock, generate_subtitle_mock, assemble_mock = self._run(
            hook=(20.0, 23.0, "A teaser"),
            clip_start_time=0.0,
        )

        teaser_mock.assert_called_once_with(20.0, 23.0, "A teaser")
        assert generate_subtitle_mock.call_count == 1
        assert (
            generate_subtitle_mock.call_args.kwargs.get("main_output_path")
            == "tmp/hook-main.mp4"
        )
        assemble_mock.assert_called_once()

    def test_non_overlapping_teaser_with_matching_text_is_skipped(self):
        # Timings shifted (jump cuts) but the words did not: second net.
        teaser_mock, generate_subtitle_mock, assemble_mock = self._run(
            hook=(20.0, 23.0, "hello world"),
            clip_start_time=0.0,
            subtitle_words=SAMPLE_WORDS,
        )

        teaser_mock.assert_not_called()
        assert generate_subtitle_mock.call_count == 1
        assert self._is_direct_publish_call(generate_subtitle_mock.call_args)
        assemble_mock.assert_not_called()

    def test_matching_text_with_other_casing_and_punctuation_is_skipped(self):
        teaser_mock, _, assemble_mock = self._run(
            hook=(20.0, 23.0, "HELLO   World!"),
            clip_start_time=0.0,
            subtitle_words=SAMPLE_WORDS,
        )

        teaser_mock.assert_not_called()
        assemble_mock.assert_not_called()

    def test_teaser_sharing_only_a_formulaic_bigram_still_assembles(self):
        # The caption net must not silently drop a fine teaser just because
        # the clip opens with the same formulaic bigram.
        words = [
            {"word": "So", "start": 0.0, "end": 0.3},
            {"word": "today", "start": 0.3, "end": 0.6},
            {"word": "we", "start": 0.6, "end": 0.7},
            {"word": "ship", "start": 0.7, "end": 1.0},
            {"word": "the", "start": 1.0, "end": 1.1},
            {"word": "pipeline.", "start": 1.1, "end": 1.6},
        ]
        teaser_mock, generate_subtitle_mock, assemble_mock = self._run(
            hook=(20.0, 23.0, "So today the retry limit shipped"),
            clip_start_time=0.0,
            subtitle_words=words,
        )

        teaser_mock.assert_called_once_with(20.0, 23.0, "So today the retry limit shipped")
        assert (
            generate_subtitle_mock.call_args.kwargs.get("main_output_path")
            == "tmp/hook-main.mp4"
        )
        assemble_mock.assert_called_once()

    def test_clip_start_time_none_keeps_the_legacy_teaser(self):
        # Guard inert: a hook at the clip opening is still published.
        teaser_mock, generate_subtitle_mock, assemble_mock = self._run(
            hook=(0.0, 3.0, "A teaser"),
            clip_start_time=None,
        )

        teaser_mock.assert_called_once_with(0.0, 3.0, "A teaser")
        assert (
            generate_subtitle_mock.call_args.kwargs.get("main_output_path")
            == "tmp/hook-main.mp4"
        )
        assemble_mock.assert_called_once()

    def test_skipped_teaser_keeps_the_end_card_path(self):
        # With the end card ON, a skipped teaser must behave exactly like the
        # no-teaser case: burn to tmp/ and assemble main + end card.
        teaser_mock, generate_subtitle_mock, assemble_mock = self._run(
            hook=(0.0, 3.0, "A teaser"),
            clip_start_time=0.0,
            end_card_enabled="true",
        )

        teaser_mock.assert_not_called()
        assert generate_subtitle_mock.call_count == 1
        assert (
            generate_subtitle_mock.call_args.kwargs.get("main_output_path")
            == "tmp/hook-main.mp4"
        )
        assemble_mock.assert_called_once()
        assert assemble_mock.call_args.kwargs.get("end_card_path") == "tmp/end-card.mp4"


class TestNoTeaserPathByteIdentical:
    """Requirement 3.7: no hook fields -> exactly today's behavior."""

    def test_segment_without_hook_fields_parses_to_none(self):
        segment = {
            "start_time": "00:00:00,000",
            "end_time": "00:00:35,000",
            "title": "Test Segment",
        }

        assert reelsfy.parse_hook_fields(segment, 0.0, 35.0) is None

    def test_hook_fields_are_optional_in_json_template(self):
        # Old content.txt files (no hook keys) must keep working: the parse
        # helper tolerates segments lacking hook data.
        old_style_segment = {
            "start_time": "00:00:00,000",
            "end_time": "00:00:35,000",
            "title": "Test",
        }

        assert reelsfy.parse_hook_fields(old_style_segment, 0.0, 35.0) is None

    def test_no_teaser_constants_do_not_change_generate_subtitle_default(self):
        # The published filename derivation is shared, so the no-teaser path
        # produces the identical output path as before Story 7.
        source = inspect.getsource(reelsfy.generate_subtitle)

        assert "main_output_path" in source
        assert "compute_published_output_path(" in source

    def test_generate_subtitle_direct_publish_uses_mastering_once(
        self, tmp_path, monkeypatch
    ):
        # Direct-publish (no teaser) still masters exactly once.
        _, call_mock = _run_generate_subtitle(tmp_path, monkeypatch)

        commands = _commands_called(call_mock)
        final_cmds = [
            c
            for c in commands
            if f"{tmp_path / 'work' / 'output'}" in c
        ]
        assert len(final_cmds) == 1
        assert f"-af \"{reelsfy.AUDIO_MASTERING_FILTER}\"" in final_cmds[0]
