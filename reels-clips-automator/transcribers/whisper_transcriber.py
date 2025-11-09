import os
import logging
from typing import List, Dict, Any

from transcribers.base import BaseTranscriber

logger = logging.getLogger(__name__)


class WhisperTranscriber(BaseTranscriber):
    """Standard Whisper transcriber implementation."""

    def __init__(self):
        self._model = None
        self._model_size = os.getenv('WHISPER_MODEL', 'small')
        # Language configuration - None means auto-detect, or specify language code (e.g., 'pt', 'en', 'es')
        self._language = os.getenv('TRANSCRIPTION_LANGUAGE', None)

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to the audio file to transcribe

        Returns:
            List of transcription segments
        """
        if not self.is_available():
            raise RuntimeError("Whisper is not available")

        try:
            import whisper

            if self._model is None:
                logger.info(f"Loading Whisper model: {self._model_size}")
                self._model = whisper.load_model(self._model_size)

            logger.info(f"Transcribing audio: {audio_path}")

            # Use configured language or auto-detect
            if self._language:
                logger.info(f"Using configured language: {self._language}")
                result = self._model.transcribe(audio_path, language=self._language)
            else:
                logger.info("Using automatic language detection")
                result = self._model.transcribe(audio_path)

            # Convert Whisper segments to standard format
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })

            return segments

        except (ImportError, SystemError) as e:
            logger.error(f"Whisper is not available: {e}")
            raise RuntimeError("Whisper is not available")
        except Exception as e:
            logger.error(f"Error during Whisper transcription: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Whisper is available."""
        try:
            import whisper
            return True
        except (ImportError, SystemError) as e:
            logger.warning(f"Whisper package is not available: {e}")
            return False

    @property
    def name(self) -> str:
        """Return the name of this transcriber."""
        return f"Whisper ({self._model_size})"