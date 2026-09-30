"""
Story 6 (P2.3): Overlay restyle + description fallback tests.

Covers:
1. build_overlay_filters emits drawtext filters with the binding style
   (fontsize=32, white@0.65, y=120 safe zone) and the historical chain
   labels ([v2] for the date, [v3] for the title/channel overlay).
2. Title precedence: video_title (arg) -> channel_name -> no title overlay.
3. get_channel_name: root .env CHANNEL_NAME with current-env fallback.
4. save_metadata appends "Date:" and "Overlay title:" lines only when the
   new params are provided; the legacy format stays byte-identical otherwise.
5. Wiring: generate_subtitle and the __main__-level metadata save receive
   the date and the same overlay-title precedence used for the burn.

All subprocess/ffmpeg usage is mocked - no real video is processed.
"""

import re
import sys
from pathlib import Path
from unittest.mock import Mock

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
        sys.modules[name] = __import__("unittest").mock.MagicMock()


for _name in ("pytube", "whisper", "audio_analyzer", "enhanced_audio_analyzer"):
    _ensure_stub(_name)

import reelsfy  # noqa: E402


FONTFILE = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
VIRAL_CONTENT = (
    '{"metadata": {"title": "T", "description": "D"}, "segments": []}'
)


class TestBuildOverlayFilters:
    """The burned overlays must use the binding restyled values."""

    def test_date_overlay_exact_style_top_left(self):
        filters, out_label = reelsfy.build_overlay_filters(
            "19/09/2026", None, input_label="0:v"
        )
        assert filters == [
            f"[0:v]drawtext=text='19/09/2026':fontfile={FONTFILE}"
            ":fontsize=32:fontcolor=white@0.65:x=30:y=120"
            ":shadowcolor=black:shadowx=2:shadowy=2[v2]"
        ]
        assert out_label == "v2"

    def test_title_overlay_exact_style_top_right(self):
        filters, out_label = reelsfy.build_overlay_filters(
            None, "My Channel", input_label="0:v"
        )
        assert filters == [
            f"[0:v]drawtext=text='My Channel':fontfile={FONTFILE}"
            ":fontsize=32:fontcolor=white@0.65:x=W-tw-30:y=120"
            ":shadowcolor=black:shadowx=2:shadowy=2[v3]"
        ]
        assert out_label == "v3"

    def test_both_overlays_chain_from_input_label(self):
        filters, out_label = reelsfy.build_overlay_filters(
            "19/09/2026", "[Show][Ep 1]", input_label="v1"
        )
        assert len(filters) == 2
        # Date chains from the subtitles output, title from the date output
        assert filters[0].startswith("[v1]drawtext=text='19/09/2026'")
        assert filters[0].endswith("[v2]")
        assert filters[1].startswith("[v2]drawtext=text='[Show][Ep 1]'")
        assert filters[1].endswith("[v3]")
        assert out_label == "v3"

    def test_no_overlays_keeps_input_label(self):
        filters, out_label = reelsfy.build_overlay_filters(
            None, None, input_label="v1"
        )
        assert filters == []
        assert out_label == "v1"


class TestOverlayEnvGates:
    """Both burned top overlays default OFF; only "true" enables one."""

    def test_both_disabled_when_unset(self, monkeypatch):
        monkeypatch.delenv("OVERLAY_DATE_ENABLED", raising=False)
        monkeypatch.delenv("OVERLAY_TITLE_ENABLED", raising=False)
        assert reelsfy.resolve_overlay_date_enabled() is False
        assert reelsfy.resolve_overlay_title_enabled() is False

    def test_enabled_only_on_literal_true(self, monkeypatch):
        monkeypatch.setenv("OVERLAY_DATE_ENABLED", "true")
        monkeypatch.setenv("OVERLAY_TITLE_ENABLED", "TRUE")
        assert reelsfy.resolve_overlay_date_enabled() is True
        assert reelsfy.resolve_overlay_title_enabled() is True

    @pytest.mark.parametrize("value", ["false", "", "1", "yes", "maybe"])
    def test_other_values_stay_off(self, monkeypatch, value):
        monkeypatch.setenv("OVERLAY_DATE_ENABLED", value)
        monkeypatch.setenv("OVERLAY_TITLE_ENABLED", value)
        assert reelsfy.resolve_overlay_date_enabled() is False
        assert reelsfy.resolve_overlay_title_enabled() is False


class TestResolveOverlayTitle:
    """Precedence: video_title (arg) beats channel_name beats none."""

    def test_video_title_beats_channel(self):
        assert (
            reelsfy.resolve_overlay_title("[Show][Ep 1]", "My Channel")
            == "[Show][Ep 1]"
        )

    def test_channel_fallback(self):
        assert reelsfy.resolve_overlay_title(None, "My Channel") == "My Channel"
        assert reelsfy.resolve_overlay_title("", "My Channel") == "My Channel"

    def test_none_when_both_missing(self):
        assert reelsfy.resolve_overlay_title(None, "") is None
        assert reelsfy.resolve_overlay_title(None, None) is None


class TestGetChannelName:
    """Root .env CHANNEL_NAME with current-env fallback."""

    def test_env_fallback_when_no_root_env_file(self, tmp_path, monkeypatch):
        # Nest the cwd so dirname(dirname(cwd)) == tmp_path has no .env
        inner = tmp_path / "a" / "b"
        inner.mkdir(parents=True)
        monkeypatch.chdir(inner)
        monkeypatch.delenv("CHANNEL_NAME", raising=False)

        assert reelsfy.get_channel_name() == ""

    def test_reads_channel_name_from_root_env(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text('CHANNEL_NAME="Env Channel"\n')
        inner = tmp_path / "x" / "y"
        inner.mkdir(parents=True)
        monkeypatch.chdir(inner)
        # .env wins over the current environment
        monkeypatch.setenv("CHANNEL_NAME", "EnvFallback")

        assert reelsfy.get_channel_name() == "Env Channel"


class TestSaveMetadataDescriptionFallback:
    """The burned date/overlay title must also land in description.txt."""

    @staticmethod
    def _metadata():
        return {
            "title": "My title",
            "description": "My description",
            "source_link": "https://example.com",
        }

    def test_legacy_format_byte_identical_without_new_params(self, tmp_path):
        reelsfy.save_metadata(self._metadata(), str(tmp_path))

        content = (tmp_path / "description.txt").read_text()
        assert content == (
            "Title:\nMy title\n\nDescription:\nMy description\n"
            "\nSource: https://example.com\n"
        )

    def test_appends_date_and_overlay_title_lines_in_order(self, tmp_path):
        reelsfy.save_metadata(
            self._metadata(),
            str(tmp_path),
            date_text="19/09/2026",
            overlay_title="My Channel",
        )

        content = (tmp_path / "description.txt").read_text()
        # Existing Title/Description/Source format untouched above the new lines
        assert content.startswith(
            "Title:\nMy title\n\nDescription:\nMy description\n"
            "\nSource: https://example.com\n"
        )
        assert content.endswith(
            "Source: https://example.com\nDate: 19/09/2026\nOverlay title: My Channel\n"
        )

    def test_date_only_appends_only_date_line(self, tmp_path):
        reelsfy.save_metadata(
            self._metadata(), str(tmp_path), date_text="19/09/2026"
        )

        content = (tmp_path / "description.txt").read_text()
        assert "Date: 19/09/2026" in content
        assert "Overlay title:" not in content

    def test_overlay_title_only_appends_only_title_line(self, tmp_path):
        reelsfy.save_metadata(
            self._metadata(), str(tmp_path), overlay_title="Chan"
        )

        content = (tmp_path / "description.txt").read_text()
        assert "Overlay title: Chan" in content
        assert "Date:" not in content


class TestGenerateSubtitleUsesOverlayHelper:
    """generate_subtitle must burn the restyled overlays with the precedence."""

    WORDS = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
    ]

    def _run(self, tmp_path, monkeypatch, video_title, channel, overlays=True):
        if overlays:
            monkeypatch.setenv("OVERLAY_DATE_ENABLED", "true")
            monkeypatch.setenv("OVERLAY_TITLE_ENABLED", "true")
        else:
            monkeypatch.delenv("OVERLAY_DATE_ENABLED", raising=False)
            monkeypatch.delenv("OVERLAY_TITLE_ENABLED", raising=False)
        (tmp_path / "tmp").mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(reelsfy, "get_channel_name", lambda: channel)

        commands = []

        def fake_call(cmd, *args, **kwargs):
            commands.append(cmd)
            return 0

        monkeypatch.setattr(reelsfy.subprocess, "call", fake_call)
        result = reelsfy.generate_subtitle(
            "seg000.mp4",
            "vid",
            "outputs/vid",
            video_title=video_title,
            subtitle_words=list(self.WORDS),
        )
        return result, commands

    def test_restyled_date_and_video_title_burned(self, tmp_path, monkeypatch):
        result, commands = self._run(
            tmp_path, monkeypatch, video_title="My Title", channel="TestChannel"
        )

        assert result is True
        assert len(commands) == 1
        command = commands[0]
        assert "fontsize=32:fontcolor=white@0.65:x=30:y=120" in command
        assert "fontsize=32:fontcolor=white@0.65:x=W-tw-30:y=120" in command
        assert "fontsize=38" not in command
        # Precedence: video_title wins over the channel name
        assert "drawtext=text='My Title':" in command
        assert "drawtext=text='TestChannel':" not in command

    def test_channel_name_fallback_burned(self, tmp_path, monkeypatch):
        result, commands = self._run(
            tmp_path, monkeypatch, video_title=None, channel="TestChannel"
        )

        assert result is True
        assert "drawtext=text='TestChannel':" in commands[0]

    def test_no_title_overlay_when_channel_empty(self, tmp_path, monkeypatch):
        result, commands = self._run(
            tmp_path, monkeypatch, video_title=None, channel=""
        )

        assert result is True
        # Date overlay is on, no title overlay
        assert commands[0].count("drawtext=") == 1
        assert "x=30:y=120" in commands[0]

    def test_no_overlays_burned_by_default(self, tmp_path, monkeypatch):
        result, commands = self._run(
            tmp_path,
            monkeypatch,
            video_title="My Title",
            channel="TestChannel",
            overlays=False,
        )

        assert result is True
        assert commands[0].count("drawtext=") == 0
        assert "x=30:y=120" not in commands[0]
        assert "x=W-tw-30:y=120" not in commands[0]

    def test_date_can_be_enabled_independently(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OVERLAY_DATE_ENABLED", "true")
        monkeypatch.delenv("OVERLAY_TITLE_ENABLED", raising=False)
        (tmp_path / "tmp").mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(reelsfy, "get_channel_name", lambda: "TestChannel")
        commands = []
        monkeypatch.setattr(
            reelsfy.subprocess, "call", lambda cmd, *a, **k: commands.append(cmd) or 0
        )

        result = reelsfy.generate_subtitle(
            "seg000.mp4",
            "vid",
            "outputs/vid",
            video_title="My Title",
            subtitle_words=list(self.WORDS),
        )

        assert result is True
        assert commands[0].count("drawtext=") == 1
        assert "x=30:y=120" in commands[0]
        assert "My Title" not in commands[0]


class TestMainMetadataWiring:
    """__main__ passes the date and burn-precedence overlay title to save_metadata."""

    def _run_main(self, monkeypatch, tmp_path, save_mock, extra_args, overlays=True):
        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"\x00\x01")

        if overlays:
            monkeypatch.setenv("OVERLAY_DATE_ENABLED", "true")
            monkeypatch.setenv("OVERLAY_TITLE_ENABLED", "true")
        else:
            monkeypatch.delenv("OVERLAY_DATE_ENABLED", raising=False)
            monkeypatch.delenv("OVERLAY_TITLE_ENABLED", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv", ["reelsfy.py", "--file", str(input_file), *extra_args]
        )
        monkeypatch.setattr(reelsfy.subprocess, "call", lambda *a, **k: 0)
        monkeypatch.setattr(
            reelsfy, "generate_transcript", lambda *a, **k: {"segments": []}
        )
        monkeypatch.setattr(
            reelsfy, "generate_viral", lambda *a, **k: {"content": VIRAL_CONTENT}
        )
        monkeypatch.setattr(reelsfy, "get_channel_name", lambda: "TestChannel")
        monkeypatch.setattr(reelsfy, "save_metadata", save_mock)
        monkeypatch.setattr(reelsfy, "cleanup_on_success", lambda *a, **k: None)
        monkeypatch.setattr(reelsfy, "cleanup_on_failure", lambda *a, **k: None)

        reelsfy.__main__()

    def test_save_metadata_receives_date_and_video_title(
        self, monkeypatch, tmp_path
    ):
        save_mock = Mock()
        self._run_main(
            monkeypatch, tmp_path, save_mock, ("--video-title", "My Title")
        )

        save_mock.assert_called_once()
        kwargs = save_mock.call_args.kwargs
        assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", kwargs["date_text"])
        # Precedence: args.video_title beats the channel name
        assert kwargs["overlay_title"] == "My Title"

    def test_save_metadata_falls_back_to_channel_name(
        self, monkeypatch, tmp_path
    ):
        save_mock = Mock()
        self._run_main(monkeypatch, tmp_path, save_mock, ())

        kwargs = save_mock.call_args.kwargs
        assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", kwargs["date_text"])
        assert kwargs["overlay_title"] == "TestChannel"

    def test_no_overlay_lines_when_overlays_disabled(
        self, monkeypatch, tmp_path
    ):
        save_mock = Mock()
        self._run_main(
            monkeypatch, tmp_path, save_mock, ("--video-title", "My Title"),
            overlays=False,
        )

        kwargs = save_mock.call_args.kwargs
        assert kwargs["date_text"] is None
        assert kwargs["overlay_title"] is None
