import os
import logging
from typing import List, Dict, Any

from transcribers.base import BaseTranscriber

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber(BaseTranscriber):
    """Faster Whisper transcriber implementation for improved performance."""

    def __init__(self):
        self._model = None
        self._model_size = os.getenv('FASTER_WHISPER_MODEL_SIZE', 'small')
        self._device = os.getenv('FASTER_WHISPER_DEVICE', 'auto')
        self._compute_type = os.getenv('FASTER_WHISPER_COMPUTE_TYPE', 'default')

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe audio file using Faster Whisper.

        Args:
            audio_path: Path to the audio file to transcribe

        Returns:
            List of transcription segments in standard format
        """
        if not self.is_available():
            raise RuntimeError("Faster Whisper is not available")

        try:
            from faster_whisper import WhisperModel

            if self._model is None:
                logger.info(f"Loading Faster Whisper model: {self._model_size} on {self._device}")
                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type
                )

            logger.info(f"Transcribing audio with Faster Whisper: {audio_path}")
            segments, info = self._model.transcribe(audio_path, language='en')

            # Convert Faster Whisper segments to standard format
            result_segments = []
            for segment in segments:
                result_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })

            logger.info(f"Transcription completed. Language: {info.language}, "
                       f"Duration: {info.duration:.2f}s, Segments: {len(result_segments)}")

            return result_segments

        except ImportError as e:
            logger.error(f"Faster Whisper is not available: {e}")
            raise RuntimeError("Faster Whisper is not available")
        except Exception as e:
            logger.error(f"Error during Faster Whisper transcription: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Faster Whisper is available."""
        try:
            from faster_whisper import WhisperModel
            return True
        except ImportError as e:
            logger.warning(f"Faster Whisper package is not available: {e}")
            return False

    @property
    def name(self) -> str:
        """Return the name of this transcriber."""
        return f"Faster Whisper ({self._model_size})"