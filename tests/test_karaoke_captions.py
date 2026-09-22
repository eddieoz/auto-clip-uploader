"""
Story 4 (P1.2b): Karaoke ASS captions + transcript reuse.

Covers:
1. build_karaoke_ass - pure builder producing a complete ASS document:
   - PlayRes header (1080x1920) so Fontsize is predictable
   - exact style values (Fontsize=58, Alignment=2, MarginV=600, Outline=3, ...)
   - word grouping in lines of at most 3 words with punctuation breaks
   - current-word highlight override + base/word Dialogue pairs
   - ASS escaping ({/}) and ASS timestamp format (H:MM:SS.cc)
   - None on missing/empty input
2. generate_subtitle wiring:
   - subtitle_words provided -> burns tmp/<base>.ass, no force_style,
     no re-transcription
   - subtitle_words absent -> legacy re-transcription + SRT force_style path
     is kept intact

All subprocess/ffmpeg usage is mocked - no real video is processed.
"""

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


for _name in (
    "pytube",
    "whisper",
    "audio_analyzer",
    "enhanced_audio_analyzer",
    "transcription_service",
):
    _ensure_stub(_name)

import reelsfy  # noqa: E402


def _words(*texts):
    """Build a flat word list with sequential 0.5s timings."""
    return [
        {"word": t, "start": float(i) * 0.5, "end": float(i) * 0.5 + 0.4}
        for i, t in enumerate(texts)
    ]


def _dialogue_lines(ass_doc):
    return [
        line
        for line in ass_doc.splitlines()
        if line.startswith("Dialogue:")
    ]


def _style_fields(ass_doc):
    """Parse the Style line into a dict using the Format header names."""
    lines = ass_doc.splitlines()
    fmt = next(l for l in lines if l.startswith("Format:"))
    style = next(l for l in lines if l.startswith("Style:"))
    names = [n.strip() for n in fmt[len("Format:"):].split(",")]
    vals = [v.strip() for v in style[len("Style:"):].split(",")]
    return dict(zip(names, vals))


# ---------------------------------------------------------------------------
# build_karaoke_ass: header / style
# ---------------------------------------------------------------------------


class TestBuilderHeader:
    def test_returns_none_on_none_input(self):
        assert reelsfy.build_karaoke_ass(None) is None

    def test_returns_none_on_empty_input(self):
        assert reelsfy.build_karaoke_ass([]) is None

    def test_contains_playres_header(self):
        doc = reelsfy.build_karaoke_ass(_words("hello", "world"))
        assert "PlayResX: 1080" in doc
        assert "PlayResY: 1920" in doc

    def test_exact_style_values(self):
        doc = reelsfy.build_karaoke_ass(_words("hello", "world"))
        style = _style_fields(doc)
        # Fontsize=58, white primary, black outline, Outline=3, Shadow=1,
        # BorderStyle=1, Alignment=2 (bottom-center), MarginV=600 (upper
        # part of the bottom half, below the frame middle), L/R=60
        assert style["Fontsize"] == "58"
        assert style["PrimaryColour"] == "&H00FFFFFF"
        assert style["OutlineColour"] == "&H00000000"
        assert style["Outline"] == "3"
        assert style["Shadow"] == "1"
        assert style["BorderStyle"] == "1"
        assert style["Alignment"] == "2"
        assert style["MarginV"] == "600"
        assert style["MarginL"] == "60"
        assert style["MarginR"] == "60"
        # Bold always set
        assert style["Bold"] == "1"

    def test_default_font_family(self):
        doc = reelsfy.build_karaoke_ass(_words("hello", "world"))
        assert _style_fields(doc)["Fontname"] == "Noto Sans"

    def test_font_family_from_env(self, monkeypatch):
        monkeypatch.setenv("CAPTION_FONT_FAMILY", "Arial Black")
        doc = reelsfy.build_karaoke_ass(_words("hello", "world"))
        assert _style_fields(doc)["Fontname"] == "Arial Black"

    def test_default_highlight_color_override(self):
        doc = reelsfy.build_karaoke_ass(_words("hello", "world"))
        assert "{\\c&H0000FFFF&}" in doc

    def test_highlight_color_from_env(self, monkeypatch):
        monkeypatch.setenv("CAPTION_HIGHLIGHT_COLOR", "&H002020F0")
        doc = reelsfy.build_karaoke_ass(_words("hello", "world"))
        assert "{\\c&H002020F0&}" in doc


# ---------------------------------------------------------------------------
# build_karaoke_ass: grouping / text
# ---------------------------------------------------------------------------


class TestBuilderGrouping:
    def test_text_is_uppercased(self):
        doc = reelsfy.build_karaoke_ass(_words("hello", "world"))
        assert "HELLO" in doc
        assert "WORLD" in doc
        assert "hello" not in doc

    def test_groups_have_at_most_three_words(self):
        words = _words("a", "b", "c", "d", "e")
        doc = reelsfy.build_karaoke_ass(words)
        lines = _dialogue_lines(doc)
        # 5 words -> groups of 3 + 2, one Dialogue per word
        assert len(lines) == 5
        group1 = [line.split(",", 9)[9] for line in lines[:3]]
        group2 = [line.split(",", 9)[9] for line in lines[3:]]
        for text in group1:
            assert "A" in text and "B" in text and "C" in text
        for text in group2:
            assert "D" in text and "E" in text and "C" not in text

    def test_punctuation_starts_new_group(self):
        # "now." ends with punctuation -> the next word starts a new group
        # even though the current group has fewer than 3 words.
        words = _words("stop", "now.", "go")
        doc = reelsfy.build_karaoke_ass(words)
        lines = _dialogue_lines(doc)
        # groups [stop, now.] and [go] -> 2 + 1 dialogues
        assert len(lines) == 3
        first_group = [line.split(",", 9)[9] for line in lines[:2]]
        for text in first_group:
            plain = text.replace("{\\c&H0000FFFF&}", "").replace(
                "{\\c&H00FFFFFF&}", ""
            )
            assert plain.split()[0] == "STOP"
            assert "NOW." in plain
            assert "GO" not in plain
        assert lines[2].split(",", 9)[9] == "{\\c&H0000FFFF&}GO{\\c&H00FFFFFF&}"

    def test_one_dialogue_per_word_with_highlight(self):
        words = _words("one", "two", "three")
        doc = reelsfy.build_karaoke_ass(words)
        lines = _dialogue_lines(doc)
        # One Dialogue per word, each carrying the highlight override
        assert len(lines) == 3
        for line in lines:
            assert "{\\c&H0000FFFF&}" in line

    def test_current_word_is_highlighted(self):
        words = _words("one", "two", "three")
        doc = reelsfy.build_karaoke_ass(words)
        highlight_lines = [
            line for line in _dialogue_lines(doc) if "{\\c&H0000FFFF&}" in line
        ]
        highlighted_words = [
            line.split("{\\c&H0000FFFF&}", 1)[1].split("{\\c&H00FFFFFF&}", 1)[0]
            for line in highlight_lines
        ]
        assert highlighted_words == ["ONE", "TWO", "THREE"]

    def test_word_events_are_contiguous_without_overlap(self):
        words = [
            {"word": "one", "start": 0.0, "end": 0.5},
            {"word": "two", "start": 0.6, "end": 1.0},
            {"word": "three", "start": 1.1, "end": 2.0},
        ]
        doc = reelsfy.build_karaoke_ass(words)
        lines = _dialogue_lines(doc)
        starts = [line.split(",")[1] for line in lines]
        ends = [line.split(",")[2] for line in lines]
        # Each event spans until the next word starts; the last ends at the
        # group end -> continuous coverage, no overlap.
        assert ends[:-1] == starts[1:]
        assert starts[0] == "0:00:00.00"
        assert ends[-1] == "0:00:02.00"


# ---------------------------------------------------------------------------
# build_karaoke_ass: escaping / timestamps
# ---------------------------------------------------------------------------


class TestBuilderEscapingAndTime:
    def test_braces_are_escaped(self):
        doc = reelsfy.build_karaoke_ass(_words("{weird}"))
        assert "\\{" in doc
        assert "\\}" in doc
        # No unescaped override-start introduced by word text
        assert "{weird}" not in doc

    def test_comma_does_not_break_dialogue_fields(self):
        doc = reelsfy.build_karaoke_ass(_words("hello,", "world"))
        for line in _dialogue_lines(doc):
            # Dialogue has exactly 9 commas separating fields; Text is the
            # last field and may contain commas.
            fields = line.split(",")
            assert len(fields) >= 10
            assert "HELLO" in line

    def test_ass_timestamp_format(self):
        words = [{"word": "hi", "start": 1.0, "end": 3661.556}]
        doc = reelsfy.build_karaoke_ass(words)
        for line in _dialogue_lines(doc):
            parts = line.split(",")
            start_ts, end_ts = parts[1], parts[2]
            # H:MM:SS.cc
            for ts in (start_ts, end_ts):
                h, m, rest = ts.split(":")
                s, cs = rest.split(".")
                assert len(m) == 2 and len(s) == 2 and len(cs) == 2
        assert "0:00:01.00" in doc
        assert "1:01:01.55" in doc


# ---------------------------------------------------------------------------
# generate_subtitle wiring
# ---------------------------------------------------------------------------


class TestGenerateSubtitleWiring:
    def _run(self, subtitle_words, input_file="karaoke_wired.mp4", output_dir="/tmp/out"):
        captured = {}

        def fake_call(cmd, shell=False):
            captured["cmd"] = cmd
            return 0

        with patch.object(reelsfy.subprocess, "call", side_effect=fake_call):
            result = reelsfy.generate_subtitle(
                input_file,
                "video-id",
                output_dir,
                subtitle_words=subtitle_words,
            )
        return result, captured.get("cmd", "")

    def test_burns_ass_file_without_force_style(self, tmp_path):
        words = _words("hello", "karaoke", "world")
        result, cmd = self._run(
            words, input_file="karaoke_wired.mp4", output_dir=str(tmp_path)
        )

        assert result is True
        assert "subtitles='tmp/karaoke_wired.ass'" in cmd
        assert "force_style" not in cmd
        # ASS file must have been written with the builder output
        assert Path(PROJECT_ROOT / "tmp" / "karaoke_wired.ass").exists()
        ass_doc = (PROJECT_ROOT / "tmp" / "karaoke_wired.ass").read_text()
        assert "HELLO" in ass_doc
        # overlays chain unchanged: date drawtext follows the subtitles filter
        assert "drawtext" in cmd
        # cleanup
        (PROJECT_ROOT / "tmp" / "karaoke_wired.ass").unlink(missing_ok=True)

    def test_no_transcription_when_words_provided(self, tmp_path):
        words = _words("hello", "world")
        called_cmds = []
        try:
            with patch.object(
                reelsfy.subprocess,
                "call",
                side_effect=lambda c, shell=False: called_cmds.append(c) or 0,
            ):
                reelsfy.generate_subtitle(
                    "karaoke_notrans.mp4", "vid", str(tmp_path), subtitle_words=words
                )
        finally:
            (PROJECT_ROOT / "tmp" / "karaoke_notrans.ass").unlink(missing_ok=True)

        assert called_cmds, "ffmpeg burn command must still run"
        for c in called_cmds:
            assert "-vn -acodec pcm_s16le" not in c, (
                "must not extract audio for re-transcription when words given"
            )


class TestGenerateSubtitleFallback:
    def test_fallback_srt_path_unchanged_when_words_absent(self, monkeypatch):
        input_file = "karaoke_fallback.mp4"
        srt_path = PROJECT_ROOT / "tmp" / "karaoke_fallback.srt"
        audio_path = PROJECT_ROOT / "tmp" / "subtitle_audio_karaoke_fallback.mp4.wav"

        # Real transcriber stub: audio extraction "succeeds", transcribe
        # returns one segment -> SRT content built and burned with force_style.
        transcriber = MagicMock()
        transcriber.name = "stub"
        transcriber.transcribe.return_value = [
            {"start": 0.0, "end": 2.0, "text": "Hello world"}
        ]
        fake_service = MagicMock()
        fake_service.get_transcriber.return_value = transcriber
        monkeypatch.setitem(sys.modules, "transcription_service", fake_service)

        real_exists = os.path.exists
        patched_names = (
            "subtitle_audio_karaoke_fallback.mp4.wav",
            "karaoke_fallback.srt",
        )
        monkeypatch.setattr(
            reelsfy.os.path,
            "exists",
            lambda p: os.path.basename(str(p)) in patched_names or real_exists(p),
        )
        monkeypatch.setattr(reelsfy.os.path, "getsize", lambda p: 1000)

        captured = {}

        def fake_call(cmd, shell=False):
            captured.setdefault("cmds", []).append(cmd)
            return 0

        try:
            with patch.object(reelsfy.subprocess, "call", side_effect=fake_call):
                result = reelsfy.generate_subtitle(
                    input_file, "vid", str(PROJECT_ROOT / "tmp" / "out-karaoke")
                )
        finally:
            srt_path.unlink(missing_ok=True)

        assert result is True
        final_cmd = [c for c in captured["cmds"] if "filter_complex" in c][0]
        assert "subtitles='tmp/karaoke_fallback.srt'" in final_cmd
        assert "force_style=" in final_cmd
        assert "Alignment=2,MarginV=130" in final_cmd
        # the old path still re-transcribes: audio extraction happened first
        assert captured["cmds"][0].startswith("ffmpeg -y -i tmp/karaoke_fallback.mp4")

    def test_none_words_triggers_fallback(self, tmp_path):
        # Explicit None behaves exactly like the legacy call.
        with patch.object(reelsfy.subprocess, "call", return_value=1):
            result = reelsfy.generate_subtitle(
                "karaoke_none.mp4", "vid", str(tmp_path), subtitle_words=None
            )
        # No transcription available -> video written without subtitles,
        # function reports failure (ffmpeg mocked return code 1).
        assert result is False


# ---------------------------------------------------------------------------
# collect_segment_words: direct unit tests (final-fix F). The helper is
# load-bearing for karaoke captions AND jump-cut remapping but previously
# only had indirect coverage.
# ---------------------------------------------------------------------------


class TestCollectSegmentWords:
    SEGMENTS = [
        {
            "words": [
                {"word": "before", "start": 9.0, "end": 9.4},
                {"word": "overlap", "start": 9.8, "end": 10.2},
                {"word": "hello", "start": 10.5, "end": 11.0},
                {"word": "world", "start": 11.5, "end": 12.0},
                {"word": "edge", "start": 11.8, "end": 12.2},
                {"word": "after", "start": 12.5, "end": 13.0},
            ]
        }
    ]

    def _collect(self, segments="default", start="00:00:10,000", end="00:00:12,000"):
        return reelsfy.collect_segment_words(
            self.SEGMENTS if segments == "default" else segments, start, end
        )

    def test_selects_in_range_words_shifted_to_clip_relative(self):
        words = self._collect()

        by_word = {w["word"]: (w["start"], w["end"]) for w in words}
        assert by_word["hello"] == (0.5, 1.0)
        assert by_word["world"] == (1.5, 2.0)
        # Words outside the segment range are excluded
        assert "before" not in by_word
        assert "after" not in by_word

    def test_word_overlapping_seg_start_inclusive_midpoint(self):
        # "overlap" spans seg_start (9.8-10.2): its midpoint sits exactly
        # on seg_start, so it is selected and its negative shift clamps
        # to 0.0 (clip-relative times are never negative).
        words = self._collect()

        by_word = {w["word"]: (w["start"], w["end"]) for w in words}
        assert by_word["overlap"] == pytest.approx((0.0, 0.2))

    def test_word_ending_exactly_at_seg_end_included(self):
        # "world" ends exactly on seg_end (12.0) and is inside; "edge"
        # has its midpoint exactly on seg_end and is also inclusive.
        words = self._collect()

        by_word = {w["word"]: (w["start"], w["end"]) for w in words}
        assert by_word["world"] == pytest.approx((1.5, 2.0))
        assert by_word["edge"] == pytest.approx((1.8, 2.2))

    def test_none_or_empty_words_input_returns_none(self):
        assert self._collect(segments=None) is None
        assert self._collect(segments=[]) is None
        assert self._collect(segments=[{"words": []}]) is None
        assert self._collect(segments=[{}]) is None
