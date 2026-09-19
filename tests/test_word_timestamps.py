"""
Unit tests for word-level timestamps (Story 3 / P1.2a).

Verifies that both transcribers request word timestamps from their backends
(word_timestamps=True) in BOTH language paths (configured and auto-detect),
map the backend words into the segment contract:

    {
        "start": float,
        "end": float,
        "text": str,
        "words": [{"word": str, "start": float, "end": float}, ...]
    }

and degrade gracefully (omit the "words" key) when the backend provides none.
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'reels-clips-automator'))

from transcribers.whisper_transcriber import WhisperTranscriber
from transcribers.faster_whisper_transcriber import FasterWhisperTranscriber


def _env_without_language():
    """Environment dict with TRANSCRIPTION_LANGUAGE removed (auto-detect path)."""
    return {k: v for k, v in os.environ.items() if k != 'TRANSCRIPTION_LANGUAGE'}


class TestWhisperWordTimestamps:
    """Test word-level timestamps from WhisperTranscriber."""

    def _mock_whisper_backend(self):
        """Build a mock whisper backend returning the real API shape with words."""
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": " Hello world. ",
                    "words": [
                        {"word": " Hello", "start": 0.0, "end": 0.6, "probability": 0.98},
                        {"word": " world. ", "start": 0.6, "end": 1.4, "probability": 0.95},
                    ],
                }
            ]
        }
        mock_model.transcribe.return_value = mock_result
        mock_whisper.load_model.return_value = mock_model
        return mock_whisper, mock_model

    def test_auto_detect_path_requests_word_timestamps(self):
        """
        Given no configured language (auto-detect path)
        When transcription is performed
        Then word_timestamps=True is passed and words are mapped
        """
        mock_whisper, mock_model = self._mock_whisper_backend()

        with patch.dict(os.environ, _env_without_language(), clear=True):
            with patch.dict('sys.modules', {'whisper': mock_whisper}):
                transcriber = WhisperTranscriber()
                result = transcriber.transcribe("/path/to/audio.wav")

                assert mock_model.transcribe.call_args.kwargs.get("word_timestamps") is True

        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 1.5
        assert result[0]["text"] == "Hello world."
        assert result[0]["words"] == [
            {"word": "Hello", "start": 0.0, "end": 0.6},
            {"word": "world.", "start": 0.6, "end": 1.4},
        ]

    def test_configured_language_path_requests_word_timestamps(self):
        """
        Given a configured language (e.g. 'pt')
        When transcription is performed
        Then word_timestamps=True is passed along with the language and words are mapped
        """
        mock_whisper, mock_model = self._mock_whisper_backend()

        with patch.dict(os.environ, {'TRANSCRIPTION_LANGUAGE': 'pt'}):
            with patch.dict('sys.modules', {'whisper': mock_whisper}):
                transcriber = WhisperTranscriber()
                result = transcriber.transcribe("/path/to/audio.wav")

                assert mock_model.transcribe.call_args.kwargs.get("word_timestamps") is True
                assert mock_model.transcribe.call_args.kwargs.get("language") == 'pt'

        assert result[0]["words"] == [
            {"word": "Hello", "start": 0.0, "end": 0.6},
            {"word": "world.", "start": 0.6, "end": 1.4},
        ]

    def test_word_text_is_stripped(self):
        """
        Given backend words carry leading/trailing whitespace
        When mapping into the contract
        Then each word's text is stripped of whitespace
        """
        mock_whisper, _ = self._mock_whisper_backend()

        with patch.dict(os.environ, _env_without_language(), clear=True):
            with patch.dict('sys.modules', {'whisper': mock_whisper}):
                transcriber = WhisperTranscriber()
                result = transcriber.transcribe("/path/to/audio.wav")

        for word in result[0]["words"]:
            assert word["word"] == word["word"].strip()

    def test_no_words_available_omits_key(self):
        """
        Given a backend segment without a words list (e.g. mock or backend limitation)
        When transcription is performed
        Then the segment simply omits the "words" key (no empty list)
        """
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": " test "}
            ]
        }
        mock_model.transcribe.return_value = mock_result
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(os.environ, _env_without_language(), clear=True):
            with patch.dict('sys.modules', {'whisper': mock_whisper}):
                transcriber = WhisperTranscriber()
                result = transcriber.transcribe("/path/to/audio.wav")

        assert result == [{"start": 0.0, "end": 1.0, "text": "test"}]
        assert "words" not in result[0]


class TestFasterWhisperWordTimestamps:
    """Test word-level timestamps from FasterWhisperTranscriber."""

    def _mock_faster_whisper_backend(self):
        """Build a mock faster_whisper backend returning the real API shape with words."""
        mock_fw = MagicMock()
        mock_model = MagicMock()

        def word(w, s, e):
            obj = MagicMock()
            obj.word = w
            obj.start = s
            obj.end = e
            return obj

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.5
        mock_segment.text = " Hello world. "
        mock_segment.words = [
            word(" Hello", 0.0, 0.6),
            word(" world. ", 0.6, 1.4),
        ]

        mock_info = MagicMock()
        mock_info.language = 'en'
        mock_info.duration = 1.5

        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_fw.WhisperModel.return_value = mock_model
        return mock_fw, mock_model

    def test_auto_detect_path_requests_word_timestamps(self):
        """
        Given no configured language (auto-detect path)
        When transcription is performed
        Then word_timestamps=True is passed and words are mapped
        """
        mock_fw, mock_model = self._mock_faster_whisper_backend()

        with patch.dict(os.environ, _env_without_language(), clear=True):
            with patch.dict('sys.modules', {'faster_whisper': mock_fw}):
                transcriber = FasterWhisperTranscriber()
                result = transcriber.transcribe("/path/to/audio.wav")

                assert mock_model.transcribe.call_args.kwargs.get("word_timestamps") is True

        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 1.5
        assert result[0]["text"] == "Hello world."
        assert result[0]["words"] == [
            {"word": "Hello", "start": 0.0, "end": 0.6},
            {"word": "world.", "start": 0.6, "end": 1.4},
        ]

    def test_configured_language_path_requests_word_timestamps(self):
        """
        Given a configured language (e.g. 'pt')
        When transcription is performed
        Then word_timestamps=True is passed along with the language and words are mapped
        """
        mock_fw, mock_model = self._mock_faster_whisper_backend()

        with patch.dict(os.environ, {'TRANSCRIPTION_LANGUAGE': 'pt'}):
            with patch.dict('sys.modules', {'faster_whisper': mock_fw}):
                transcriber = FasterWhisperTranscriber()
                result = transcriber.transcribe("/path/to/audio.wav")

                assert mock_model.transcribe.call_args.kwargs.get("word_timestamps") is True
                assert mock_model.transcribe.call_args.kwargs.get("language") == 'pt'

        assert result[0]["words"] == [
            {"word": "Hello", "start": 0.0, "end": 0.6},
            {"word": "world.", "start": 0.6, "end": 1.4},
        ]

    def test_word_text_is_stripped(self):
        """
        Given backend words carry leading/trailing whitespace
        When mapping into the contract
        Then each word's text is stripped of whitespace
        """
        mock_fw, _ = self._mock_faster_whisper_backend()

        with patch.dict(os.environ, _env_without_language(), clear=True):
            with patch.dict('sys.modules', {'faster_whisper': mock_fw}):
                transcriber = FasterWhisperTranscriber()
                result = transcriber.transcribe("/path/to/audio.wav")

        for word in result[0]["words"]:
            assert word["word"] == word["word"].strip()

    def test_no_words_available_omits_key(self):
        """
        Given a backend segment whose words list is None (backend limitation)
        When transcription is performed
        Then the segment simply omits the "words" key (no empty list)
        """
        mock_fw = MagicMock()
        mock_model = MagicMock()

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = " test "
        mock_segment.words = None

        mock_info = MagicMock()
        mock_info.language = 'en'
        mock_info.duration = 1.0

        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_fw.WhisperModel.return_value = mock_model

        with patch.dict(os.environ, _env_without_language(), clear=True):
            with patch.dict('sys.modules', {'faster_whisper': mock_fw}):
                transcriber = FasterWhisperTranscriber()
                result = transcriber.transcribe("/path/to/audio.wav")

        assert result == [{"start": 0.0, "end": 1.0, "text": "test"}]
        assert "words" not in result[0]
