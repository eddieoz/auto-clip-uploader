"""
Tests for S1: word-boundary phrase matching in TopicAnalyzer.

Covers:
1. The ``phrase_matches`` helper directly (false positives eliminated, true
   positives preserved, edge cases: empty text/phrase, regex-special chars).
2. Behavior-level confidence: a cue that previously gained a bogus transition
   boost via substring matching ("person" contains "so") no longer gets it,
   while cues with real transition/conclusion phrases still get boosted.

Matching-semantics-only constraint: no scoring weights, thresholds, or phrase
list contents are changed by the implementation under test.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
REELSFY_DIR = PROJECT_ROOT / "reels-clips-automator"

sys.path.insert(0, str(REELSFY_DIR))

from topic_analyzer import TopicAnalyzer, phrase_matches  # noqa: E402


# ---------------------------------------------------------------------------
# Direct unit tests on the phrase_matches helper
# ---------------------------------------------------------------------------


class TestPhraseMatchesFalsePositives:
    """Substrings inside larger words must no longer match."""

    @pytest.mark.parametrize(
        "text,phrase",
        [
            ("the person arrived", "so"),
            ("the season finale", "so"),
            ("no reason given", "so"),
            ("i know", "now"),
            ("nowhere to be found", "now"),
            ("a butterfly flew by", "but"),
            ("without a doubt", "but"),
            ("buttercup", "but"),
        ],
    )
    def test_words_containing_phrase_substring_do_not_match(self, text, phrase):
        assert phrase_matches(text, phrase) is False


class TestPhraseMatchesTruePositives:
    """Real standalone phrases must still match."""

    @pytest.mark.parametrize(
        "text,phrase",
        [
            ("However, there is more", "however"),
            ("so we begin", "so"),
            ("So we begin", "so"),
            ("But wait", "but"),
            ("now the next part", "now"),
            # Multi-word phrase across punctuation and varied spacing
            ("on the other hand, consider this", "on the other hand"),
            ("on  the   other hand", "on the other hand"),
            ("on the\nother hand", "on the other hand"),
            # Conclusion phrases still match mid-sentence
            ("and the key takeaway is simple", "the key takeaway"),
            ("the bottom line is that testing matters", "bottom line"),
            ("what this means for you", "what this means"),
            # Coherence phrases still match
            ("results were better, as a result of fixes", "as a result"),
        ],
    )
    def test_standalone_phrases_match(self, text, phrase):
        assert phrase_matches(text, phrase) is True


class TestPhraseMatchesEdgeCases:
    """Edge cases: empty inputs and regex-special characters."""

    def test_empty_text(self):
        assert phrase_matches("", "so") is False

    def test_empty_phrase(self):
        assert phrase_matches("some text", "") is False

    def test_whitespace_only_phrase(self):
        assert phrase_matches("some text", "   ") is False

    def test_regex_special_characters_are_escaped(self):
        # A dot must not act as a wildcard: "e.g." must not match "exg this"
        assert phrase_matches("e.g. this works", "e.g. this") is True
        assert phrase_matches("aexg this works", "e.g. this") is False

    def test_plus_signs_are_escaped(self):
        assert phrase_matches("we ship c++ style code", "c++ style") is True
        assert phrase_matches("we ship cxx style code", "c++ style") is False


# ---------------------------------------------------------------------------
# Behavior-level tests through TopicAnalyzer
# ---------------------------------------------------------------------------

BOGUS_CUE = "The person arrived and shared the season plan with everyone"
CLEAN_CUE = "The visitor arrived and shared the monthly plan with everyone"
TRANSITION_CUE = "So we begin the nightly builds discussion right away"
CONCLUSION_CUE = "Finally the key takeaway is to automate the deployments"


def _make_srt(texts):
    """Build a small SRT transcript from cue texts with fixed timings."""
    cues = []
    for i, text in enumerate(texts):
        cues.append(f"{i + 1}")
        cues.append(f"00:00:{i * 5:02d},000 --> 00:00:{i * 5 + 5:02d},000")
        cues.append(text)
        cues.append("")
    return "\n".join(cues)


def _analyzed_segments(texts):
    analyzer = TopicAnalyzer()
    segments = analyzer.parse_srt_segments(_make_srt(texts))
    analyzer.analyze_topic_transitions(segments)
    return analyzer, segments


class TestBehaviorLevelConfidence:
    """Confidence boosts must follow word-boundary matching, not substrings."""

    def test_substring_cue_precondition_holds(self):
        # Non-vacuity guard: the old substring matching would fire on this cue.
        assert "so" in BOGUS_CUE.lower()
        assert "so" not in CLEAN_CUE.lower()

    def test_bogus_substring_cue_gets_no_transition_boost(self):
        analyzer, segments = _analyzed_segments([BOGUS_CUE])
        segment = segments[0]
        # Confidence must equal the unboosted base calculation (no +0.1).
        assert segment.topic_confidence == pytest.approx(
            analyzer._calculate_topic_confidence(segment)
        )

    def test_bogus_confidence_matches_clean_equivalent(self):
        _, bogus = _analyzed_segments([BOGUS_CUE])
        _, clean = _analyzed_segments([CLEAN_CUE])
        assert bogus[0].topic_confidence == pytest.approx(clean[0].topic_confidence)

    def test_real_transition_cue_still_boosted_by_0_1(self):
        analyzer, segments = _analyzed_segments([TRANSITION_CUE])
        segment = segments[0]
        assert segment.topic_confidence == pytest.approx(
            analyzer._calculate_topic_confidence(segment) + 0.1
        )

    def test_conclusion_cue_still_boosted_by_0_2(self):
        analyzer, segments = _analyzed_segments([CONCLUSION_CUE])
        segment = segments[0]
        assert segment.topic_confidence == pytest.approx(
            analyzer._calculate_topic_confidence(segment) + 0.2
        )

    def test_coherence_false_positive_not_boosted(self):
        # "leads total" contains the substring "leads to" but is not the phrase.
        _, bogus = _analyzed_segments(["this leads total confusion about results"])
        _, clean = _analyzed_segments(["this leads utter confusion about results"])
        assert bogus[0].topic_confidence == pytest.approx(clean[0].topic_confidence)

    def test_real_coherence_keyword_still_boosted(self):
        # Identical shape except "means that" -> "implies that" (not a listed
        # phrase); the only confidence difference must be the 0.05 coherence
        # bonus for the real phrase.
        _, with_cue = _analyzed_segments(["this means that the tests stay green"])
        _, without_cue = _analyzed_segments(["this implies that the tests stay green"])
        diff = with_cue[0].topic_confidence - without_cue[0].topic_confidence
        assert diff == pytest.approx(0.05)
