"""
Tests for S5 (Task 16): hook-first selection within the concluding topic.

Plan: docs/VIRAL_SELECTION_IMPROVEMENT_PLAN.md, story S5 (gap G5).

On short-form platforms the first 2 s decide retention, so the hook is
identified early by the Task-13 boundary LLM call (which already sees the
entire capture) and becomes a FIXED part of the answer. This suite covers:

1. ``resolve_topic_start_cue`` extended reply parsing — the cue number and
   the hook candidates are parsed independently: a missing/invalid
   candidates list never invalidates the cue number (and vice versa); the
   legacy integer-only replies still parse; garbage degrades to
   ``(None, None, None, [])`` without raising.
2. ``select_hook_candidate`` — pure selection of the FIRST hook candidate
   whose span (cue_start → cue_end mapped to seconds over the segments
   list) falls inside the snapped resolved window and spans 1.5-6 s;
   malformed candidates are skipped, never raising.
3. Mandatory-hook injection in the LIVE prompt: a surviving candidate
   extends the RESOLVED TOPIC WINDOW section with a MANDATORY HOOK block
   (instruction prose only — the JSON template is untouched); with no
   surviving candidate the section is byte-identical to Task 14's and the
   hook stays optional.
4. Downstream: ``parse_hook_fields`` (NOT weakened) accepts the echoed
   mandatory values.

The OpenAI client and heavy analyzers are replaced with mocks — no network
access and no real audio processing happens.
"""

import json
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

import reelsfy  # noqa: E402
import topic_analyzer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# 33 chars, '.' at index 10 → snap offset (10 + 1) / 33
START_TEXT = "First part. Second part continues"
HOOK_TEXT = "This result is completely insane"
NO_PUNCT = "no sentence punctuation anywhere in here"

# Resolved window for the S5 cue layout below: trim to cue 2 (15 s) then
# snap the start past the in-cue terminator → 25 s; the end stays 60 s.
WINDOW_START_SRT = "00:00:25,000"
WINDOW_END_SRT = "00:01:00,000"

# Primary-path cue layout:
#   cue 1: 0-15   (previous-topic tail)
#   cue 2: 15-45  (final topic starts; START_TEXT snaps the start to 25 s)
#   cue 3: 45-50  (short hook cue inside the resolved window)
#   cue 4: 50-60  (conclusion, no terminators)
HOOK_CUES = [
    {
        "start_time": 0.0,
        "end_time": 15.0,
        "start_time_srt": "00:00:00,000",
        "end_time_srt": "00:00:15,000",
        "text": NO_PUNCT,
    },
    {
        "start_time": 15.0,
        "end_time": 45.0,
        "start_time_srt": "00:00:15,000",
        "end_time_srt": "00:00:45,000",
        "text": START_TEXT,
    },
    {
        "start_time": 45.0,
        "end_time": 50.0,
        "start_time_srt": "00:00:45,000",
        "end_time_srt": "00:00:50,000",
        "text": HOOK_TEXT,
    },
    {
        "start_time": 50.0,
        "end_time": 60.0,
        "start_time_srt": "00:00:50,000",
        "end_time_srt": "00:01:00,000",
        "text": NO_PUNCT,
    },
]


def _cue(start, end, text):
    return {
        "start_time": start,
        "end_time": end,
        "start_time_srt": reelsfy.format_timestamp(start),
        "end_time_srt": reelsfy.format_timestamp(end),
        "text": text,
    }


def _candidate(cue_start, cue_end, text):
    return {"cue_start": cue_start, "cue_end": cue_end, "text": text}


def _topic_analysis_result():
    return {
        "segments": HOOK_CUES,
        "concluding_topic": {
            "text": "The final topic text",
            "topic_keywords": ["final"],
            "topic_confidence": 0.9,
        },
        "supporting_segments": [HOOK_CUES[0]],
        "optimized_segment": {
            "start_time": "00:00:00,000",
            "end_time": "00:01:00,000",
            "duration": 60.0,
            "title": "Final topic title",
        },
    }


def _llm_client_replying_with(text):
    """Build a patched OpenAI module whose reply is `text`."""
    mock_openai = MagicMock()
    mock_client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = text
    mock_client.chat.completions.create.return_value = response
    mock_openai.OpenAI.return_value = mock_client
    return mock_openai


def _resolve_with_reply(monkeypatch, reply, segments=None):
    """Run resolve_topic_start_cue with the LLM pinned to `reply`."""
    if segments is None:
        segments = HOOK_CUES
    monkeypatch.delenv("TOPIC_BOUNDARY_LLM", raising=False)
    mock_openai = _llm_client_replying_with(reply)
    with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
        result = reelsfy.resolve_topic_start_cue("srt content", segments)
    return result


def _run_generate_viral(
    topic_analysis, resolve_return, audio_path=None, output_dir=None
):
    """Run generate_viral with the LLM and heavy dependencies mocked.

    The boundary pass is stubbed with `resolve_return` (the new
    (cue_index, earlier_start_cue, hook_candidates) shape). Returns
    (result, captured messages, resolve_topic_start_cue mock).
    """
    analyzer_instance = MagicMock()
    analyzer_instance.analyze_transcript.return_value = topic_analysis
    fake_topic_analyzer = MagicMock(return_value=analyzer_instance)

    metadata = {"content": json.dumps({"title": "T", "description": "D"})}
    final_reply = json.dumps(
        {
            "segments": [
                {
                    "start_time": "00:00:25,000",
                    "end_time": "00:01:00,000",
                    "title": "T",
                    "description": "D",
                }
            ]
        }
    )

    captured = {}

    def _capture(model, messages, n, stop, **kwargs):
        captured["messages"] = messages
        response = MagicMock()
        response.choices[0].message.content = final_reply
        return response

    openai_module = MagicMock()
    openai_module.OpenAI.return_value.chat.completions.create.side_effect = (
        _capture
    )

    with patch.object(
        topic_analyzer, "TopicAnalyzer", fake_topic_analyzer
    ), patch.object(
        reelsfy, "generate_metadata", return_value=metadata
    ), patch.object(
        reelsfy, "resolve_topic_start_cue", return_value=resolve_return
    ) as resolve_mock, patch.object(
        reelsfy.openai, "OpenAI", openai_module.OpenAI
    ), patch.object(
        reelsfy,
        "check_audio_file",
        return_value=(audio_path is not None, "stubbed"),
    ), patch.object(
        sys.modules["enhanced_audio_analyzer"],
        "EnhancedAudioAnalyzer",
        MagicMock(),
    ):
        result = reelsfy.generate_viral(
            "srt content", audio_path=audio_path, output_dir=output_dir
        )

    return result, captured, resolve_mock


# ---------------------------------------------------------------------------
# Extended boundary reply parsing
# ---------------------------------------------------------------------------


class TestParseBoundaryReply:
    """Cue number and hook candidates parsed independently; never raises."""

    def test_cue_and_candidates_both_parsed(self, monkeypatch):
        reply = json.dumps(
            {
                "cue_number": 2,
                "hook_candidates": [
                    _candidate(3, 3, "Hook one"),
                    _candidate(3, 4, "Hook two"),
                ],
            }
        )
        result = _resolve_with_reply(monkeypatch, reply)
        assert result == (
            2,
            None,
            None,
            [_candidate(3, 3, "Hook one"), _candidate(3, 4, "Hook two")],
        )

    def test_cue_only_reply_yields_empty_candidates(self, monkeypatch):
        result = _resolve_with_reply(
            monkeypatch, json.dumps({"cue_number": 3})
        )
        assert result == (3, None, None, [])

    def test_candidates_only_reply_yields_none_cue(self, monkeypatch):
        reply = json.dumps(
            {"hook_candidates": [_candidate(3, 3, "The hook")]}
        )
        result = _resolve_with_reply(monkeypatch, reply)
        assert result == (None, None, None, [_candidate(3, 3, "The hook")])

    def test_invalid_cue_does_not_invalidate_candidates(self, monkeypatch):
        reply = json.dumps(
            {
                "cue_number": 99,
                "hook_candidates": [_candidate(3, 3, "The hook")],
            }
        )
        result = _resolve_with_reply(monkeypatch, reply)
        assert result == (None, None, None, [_candidate(3, 3, "The hook")])

    def test_invalid_candidates_do_not_invalidate_cue(self, monkeypatch):
        reply = json.dumps(
            {"cue_number": 2, "hook_candidates": "not a list"}
        )
        result = _resolve_with_reply(monkeypatch, reply)
        assert result == (2, None, None, [])

    def test_malformed_candidate_entries_are_skipped(self, monkeypatch):
        reply = json.dumps(
            {
                "cue_number": 2,
                "hook_candidates": [
                    None,
                    42,
                    _candidate("45", 50, "string times"),
                    {"cue_start": 3, "cue_end": 3},
                    _candidate(3, 3, "   "),
                    _candidate(3, 3, "Good hook"),
                ],
            }
        )
        result = _resolve_with_reply(monkeypatch, reply)
        cue_index, topic_complete, earlier_start_cue, candidates = result
        assert cue_index == 2
        assert topic_complete is None
        assert earlier_start_cue is None
        assert candidates == [_candidate(3, 3, "Good hook")]

    def test_legacy_integer_reply_still_parses(self, monkeypatch):
        for reply in ("3", "cue 3", " 3 ", "Cue 3"):
            assert _resolve_with_reply(monkeypatch, reply) == (3, None, None, [])

    def test_garbage_reply_degrades_to_none_pair(self, monkeypatch):
        for reply in (
            "sorry, I cannot help with that",
            "",
            "none",
            json.dumps(["unexpected", "shape"]),
        ):
            assert _resolve_with_reply(monkeypatch, reply) == (None, None, None, [])

    def test_out_of_range_legacy_integer_degrades_to_none_pair(
        self, monkeypatch
    ):
        for reply in ("0", "99", "9999"):
            assert _resolve_with_reply(monkeypatch, reply) == (None, None, None, [])

    def test_exception_degrades_to_none_pair(self, monkeypatch):
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = (
            RuntimeError("api down")
        )
        monkeypatch.setenv("TOPIC_BOUNDARY_LLM", "true")
        with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
            result = reelsfy.resolve_topic_start_cue("srt content", HOOK_CUES)
        assert result == (None, None, None, [])

    def test_gate_off_degrades_to_none_pair(self, monkeypatch):
        mock_openai = MagicMock()
        monkeypatch.setenv("TOPIC_BOUNDARY_LLM", "false")
        with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
            result = reelsfy.resolve_topic_start_cue("srt content", HOOK_CUES)
        assert result == (None, None, None, [])
        mock_openai.OpenAI.assert_not_called()

    def test_empty_segments_degrade_to_none_pair(self, monkeypatch):
        result = _resolve_with_reply(monkeypatch, "3", segments=[])
        assert result == (None, None, None, [])

    def test_prompt_ask_for_hook_candidates(self, monkeypatch):
        captured = {}

        def _capture(model, messages, n, stop):
            captured["messages"] = messages
            response = MagicMock()
            response.choices[0].message.content = "2"
            return response

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = (
            _capture
        )
        monkeypatch.setenv("TOPIC_BOUNDARY_LLM", "true")
        with patch.object(reelsfy.openai, "OpenAI", mock_openai.OpenAI):
            reelsfy.resolve_topic_start_cue("srt content", HOOK_CUES)

        user_content = captured["messages"][-1]["content"]
        # The hook request rides along on the boundary prompt.
        assert "hook_candidates" in user_content
        assert "cue_start" in user_content
        assert "cue_end" in user_content
        # The original cue-number contract is still spelled out.
        for index, segment in enumerate(HOOK_CUES, start=1):
            assert f"cue {index}:" in user_content
            assert segment["text"] in user_content
        assert "concludes" in user_content.lower()
        assert "begins" in user_content.lower()


# ---------------------------------------------------------------------------
# select_hook_candidate (pure helper)
# ---------------------------------------------------------------------------


# Story B1 cue layout: the resolved window is 25-60 s, so the opening
# duplicate guard covers 25-30 s.
#   cue 1: 25-28  (inside the opening guard — the topic's first sentence)
#   cue 2: 28-33  (starts inside the guard, runs past it)
#   cue 3: 35-40  (entirely past the guard — selectable)
#   cue 4: 45-60  (tail)
OPENING_CUES = [
    _cue(25.0, 28.0, "Opening sentence"),
    _cue(28.0, 33.0, "Straddling sentence"),
    _cue(35.0, 40.0, "Mid-window hook"),
    _cue(45.0, 60.0, NO_PUNCT),
]


class TestSelectHookCandidate:
    """First candidate inside the snapped window with a 1.5-6 s span."""

    def test_first_passing_candidate_returned(self):
        candidates = [
            _candidate(1, 1, "Outside hook"),
            _candidate(3, 3, "Inside hook"),
            _candidate(4, 4, "Too long hook"),
        ]

        selected = reelsfy.select_hook_candidate(
            candidates, WINDOW_START_SRT, WINDOW_END_SRT, HOOK_CUES
        )

        assert selected == {
            "hook_start_time": "00:00:45,000",
            "hook_end_time": "00:00:50,000",
            "hook_text": "Inside hook",
        }

    def test_candidate_outside_window_skipped(self):
        # Cue 1 (0-15 s) sits entirely before the 25-60 s window.
        candidates = [_candidate(1, 1, "Outside hook")]
        assert (
            reelsfy.select_hook_candidate(
                candidates, WINDOW_START_SRT, WINDOW_END_SRT, HOOK_CUES
            )
            is None
        )

    def test_candidate_partially_overlapping_window_skipped(self):
        # Cue 2 (15-45 s) starts before the 25 s window start.
        candidates = [_candidate(2, 2, "Overlapping hook")]
        assert (
            reelsfy.select_hook_candidate(
                candidates, WINDOW_START_SRT, WINDOW_END_SRT, HOOK_CUES
            )
            is None
        )

    def test_span_longer_than_max_duration_skipped(self):
        # Cue 2 spans 30 s — far above the 6 s ceiling.
        candidates = [_candidate(2, 2, "Too long hook")]
        assert (
            reelsfy.select_hook_candidate(
                candidates, WINDOW_START_SRT, WINDOW_END_SRT, HOOK_CUES
            )
            is None
        )

    def test_span_shorter_than_min_duration_skipped(self):
        cues = [_cue(40.0, 41.0, "tiny cue"), _cue(41.0, 60.0, NO_PUNCT)]
        candidates = [_candidate(1, 1, "Tiny hook")]

        selected = reelsfy.select_hook_candidate(
            candidates, "00:00:40,000", "00:01:00,000", cues
        )

        assert selected is None

    def test_malformed_candidates_skipped_without_raising(self):
        candidates = [
            None,
            42,
            "not a dict",
            _candidate("45", 50, "string cues"),
            {"cue_start": 3, "cue_end": 3},
            _candidate(3, 3, "   "),
            _candidate(0, 3, "zero cue"),
            _candidate(3, 99, "out of range"),
            _candidate(4, 3, "reversed cues"),
            _candidate(3, 3, "Good hook"),
        ]

        selected = reelsfy.select_hook_candidate(
            candidates, WINDOW_START_SRT, WINDOW_END_SRT, HOOK_CUES
        )

        assert selected == {
            "hook_start_time": "00:00:45,000",
            "hook_end_time": "00:00:50,000",
            "hook_text": "Good hook",
        }

    def test_no_valid_candidate_returns_none(self):
        candidates = [_candidate(1, 1, "Outside"), _candidate(2, 2, "Too long")]
        assert (
            reelsfy.select_hook_candidate(
                candidates, WINDOW_START_SRT, WINDOW_END_SRT, HOOK_CUES
            )
            is None
        )

    def test_empty_candidates_return_none(self):
        assert (
            reelsfy.select_hook_candidate(
                [], WINDOW_START_SRT, WINDOW_END_SRT, HOOK_CUES
            )
            is None
        )

    def test_non_list_candidates_return_none(self):
        assert (
            reelsfy.select_hook_candidate(
                None, WINDOW_START_SRT, WINDOW_END_SRT, HOOK_CUES
            )
            is None
        )
        assert (
            reelsfy.select_hook_candidate(
                "nope", WINDOW_START_SRT, WINDOW_END_SRT, HOOK_CUES
            )
            is None
        )

    def test_missing_segments_return_none(self):
        candidates = [_candidate(3, 3, "Hook")]
        assert (
            reelsfy.select_hook_candidate(
                candidates, WINDOW_START_SRT, WINDOW_END_SRT, []
            )
            is None
        )

    def test_unparseable_window_returns_none(self):
        candidates = [_candidate(3, 3, "Hook")]
        assert (
            reelsfy.select_hook_candidate(
                candidates, "not-a-timestamp", WINDOW_END_SRT, HOOK_CUES
            )
            is None
        )


# ---------------------------------------------------------------------------
# Story B1: the opening guard — a teaser that repeats the clip's first
# seconds is a bug, not a hook. Only the opening is guarded; every other
# check keeps its current behaviour.
# ---------------------------------------------------------------------------


class TestOpeningDuplicateGuard:
    """Candidates overlapping the first ~5 s of the window are rejected."""

    def test_candidate_inside_opening_guard_rejected(self):
        # Cue 1 spans window_start+0.5 s .. window_start+3 s.
        candidates = [_candidate(1, 1, "Opening sentence")]

        assert (
            reelsfy.select_hook_candidate(
                candidates, WINDOW_START_SRT, WINDOW_END_SRT, OPENING_CUES
            )
            is None
        )

    def test_candidate_starting_past_the_guard_selected(self):
        # Cue 3 (35-40 s) starts ~10 s after the window start.
        candidates = [_candidate(3, 3, "Mid-window hook")]

        selected = reelsfy.select_hook_candidate(
            candidates, WINDOW_START_SRT, WINDOW_END_SRT, OPENING_CUES
        )

        assert selected == {
            "hook_start_time": "00:00:35,000",
            "hook_end_time": "00:00:40,000",
            "hook_text": "Mid-window hook",
        }

    def test_only_guarded_candidate_returns_none(self):
        # Degradation invariant: no survivor → no teaser, never an error.
        candidates = [
            _candidate(1, 1, "Opening sentence"),
            _candidate(2, 2, "Straddling sentence"),
        ]

        assert (
            reelsfy.select_hook_candidate(
                candidates, WINDOW_START_SRT, WINDOW_END_SRT, OPENING_CUES
            )
            is None
        )

    def test_span_overlapping_the_guard_rejected(self):
        # Cue 2 (28-33 s) starts inside the 25-30 s guard and runs past it:
        # overlap is what disqualifies it, not merely starting inside.
        candidates = [_candidate(2, 2, "Straddling sentence")]

        assert (
            reelsfy.select_hook_candidate(
                candidates, WINDOW_START_SRT, WINDOW_END_SRT, OPENING_CUES
            )
            is None
        )

    def test_guarded_candidate_skipped_for_a_later_survivor(self):
        # Ranking order is preserved: the guarded candidate is skipped, the
        # next survivor is returned unchanged.
        candidates = [
            _candidate(1, 1, "Opening sentence"),
            _candidate(3, 3, "Mid-window hook"),
        ]

        selected = reelsfy.select_hook_candidate(
            candidates, WINDOW_START_SRT, WINDOW_END_SRT, OPENING_CUES
        )

        assert selected == {
            "hook_start_time": "00:00:35,000",
            "hook_end_time": "00:00:40,000",
            "hook_text": "Mid-window hook",
        }

    def test_guard_constant_is_five_seconds(self):
        assert reelsfy.HOOK_DUPLICATE_GUARD_SECONDS == 5.0


# ---------------------------------------------------------------------------
# Mandatory-hook injection into the LIVE prompt
# ---------------------------------------------------------------------------

EXPECTED_TASK14_SECTION = (
    "\nRESOLVED TOPIC WINDOW:\n"
    "- Resolved start: 00:00:25,000\n"
    "- Resolved end: 00:01:00,000\n"
    "- The final topic begins at cue 2 of the numbered transcript cues above\n"
    "The returned segment MUST start at or after the resolved start time and "
    "end at or before the resolved end time. Prefer exactly the resolved "
    "boundaries; only a clear sentence boundary justifies a slight "
    "adjustment within them.\n"
)


class TestMandatoryHookInjection:
    """A surviving candidate extends the resolved-window section; otherwise
    the prompt is byte-identical to Task 14's."""

    def test_mandatory_hook_block_reaches_the_live_prompt(self):
        result, captured, _ = _run_generate_viral(
            _topic_analysis_result(),
            (2, None, None, [_candidate(3, 3, HOOK_TEXT)]),
        )

        assert result, "generate_viral must return the merged JSON"
        user_content = captured["messages"][1]["content"]
        assert "MANDATORY HOOK" in user_content
        assert (
            "You MUST return hook_start_time / hook_end_time / hook_text "
            "with EXACTLY these values" in user_content
        )
        assert "hook_start_time: 00:00:45,000" in user_content
        assert "hook_end_time: 00:00:50,000" in user_content
        assert f'hook_text: "{HOOK_TEXT}"' in user_content
        # The resolved-window section itself is still present, unchanged.
        assert EXPECTED_TASK14_SECTION in user_content

    def test_hook_block_is_instruction_prose_after_the_resolved_section(self):
        result, captured, _ = _run_generate_viral(
            _topic_analysis_result(),
            (2, None, None, [_candidate(3, 3, HOOK_TEXT)]),
        )

        user_content = captured["messages"][1]["content"]
        assert user_content.index("RESOLVED TOPIC WINDOW") < user_content.index(
            "MANDATORY HOOK"
        )
        # Instruction prose only: the JSON template stays untouched (the
        # hook fields are already declared optional inside it).
        json_template = reelsfy.build_segment_json_template()
        assert "MANDATORY" not in json_template

    def test_no_surviving_candidate_keeps_prompt_without_hook_block(self):
        result, captured, _ = _run_generate_viral(
            _topic_analysis_result(), (2, None, None, [])
        )

        user_content = captured["messages"][1]["content"]
        assert "MANDATORY HOOK" not in user_content
        # Byte-identical to Task 14's resolved-window section.
        assert EXPECTED_TASK14_SECTION in user_content

    def test_gate_off_keeps_prompt_without_hook_block(self):
        # Gate off → the boundary pass returns (None, None, None, []): the resolved
        # window (untrimmed + snapped) still reaches the prompt, without a
        # hook block and without the cue line.
        result, captured, resolve_mock = _run_generate_viral(
            _topic_analysis_result(), (None, None, None, [])
        )

        user_content = captured["messages"][1]["content"]
        assert "MANDATORY HOOK" not in user_content
        assert "The final topic begins at cue" not in user_content
        assert "RESOLVED TOPIC WINDOW" in user_content
        resolve_mock.assert_called_once()

    def test_fallback_path_never_injects_a_hook_block(self, tmp_path):
        result, captured, resolve_mock = _run_generate_viral(
            {"error": "topic analysis exploded"}, (None, None, None, [])
        )

        assert result, "fallback path must still produce a result"
        resolve_mock.assert_not_called()
        user_content = captured["messages"][1]["content"]
        assert "MANDATORY HOOK" not in user_content
        assert "RESOLVED TOPIC WINDOW" not in user_content


# ---------------------------------------------------------------------------
# Downstream: parse_hook_fields accepts the echoed mandatory values
# ---------------------------------------------------------------------------


class TestDownstreamParseHookFields:
    """The echoed mandatory hook values pass today's validation unchanged."""

    def test_parse_hook_fields_accepts_mandatory_values(self):
        segment = {
            "start_time": "00:00:25,000",
            "end_time": "00:01:00,000",
            "hook_start_time": "00:00:45,000",
            "hook_end_time": "00:00:50,000",
            "hook_text": HOOK_TEXT,
        }
        seg_start = reelsfy.parse_srt_time(segment["start_time"])
        seg_end = reelsfy.parse_srt_time(segment["end_time"])

        hook = reelsfy.parse_hook_fields(segment, seg_start, seg_end)

        assert hook == (
            reelsfy.parse_srt_time("00:00:45,000"),
            reelsfy.parse_srt_time("00:00:50,000"),
            HOOK_TEXT,
        )
