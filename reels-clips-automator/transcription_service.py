import os
import logging
from typing import Optional

from transcribers.base import BaseTranscriber
from transcribers.whisper_transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)


class TranscriptionServiceFactory:
    """Factory for creating transcription service instances."""

    @staticmethod
    def create_transcriber(model_type: Optional[str] = None) -> BaseTranscriber:
        """
        Create a transcriber instance based on the specified model type.

        Args:
            model_type: The type of transcriber to create. If None, uses STT_MODEL
                       environment variable or defaults to "whisper".

        Returns:
            BaseTranscriber instance

        Raises:
            ValueError: If an unsupported model type is specified
            RuntimeError: If the requested transcriber is not available
        """
        if model_type is None:
            model_type = os.getenv('STT_MODEL', 'whisper').lower()

        logger.info(f"Creating transcriber for model type: {model_type}")

        if model_type == 'whisper':
            transcriber = WhisperTranscriber()
        elif model_type == 'faster-whisper':
            try:
                from transcribers.faster_whisper_transcriber import FasterWhisperTranscriber
                transcriber = FasterWhisperTranscriber()
            except ImportError:
                logger.warning("Faster Whisper not available, falling back to standard Whisper")
                transcriber = WhisperTranscriber()
        else:
            logger.error(f"Unsupported STT model type: {model_type}")
            logger.info("Falling back to standard Whisper")
            transcriber = WhisperTranscriber()

        if not transcriber.is_available():
            if model_type != 'whisper':
                logger.warning(f"{transcriber.name} is not available, falling back to Whisper")
                transcriber = WhisperTranscriber()
                if not transcriber.is_available():
                    raise RuntimeError("No transcription service is available")
            else:
                raise RuntimeError(f"{transcriber.name} is not available")

        logger.info(f"Using transcriber: {transcriber.name}")
        return transcriber


def get_transcriber() -> BaseTranscriber:
    """
    Convenience function to get a transcriber instance.

    Returns:
        BaseTranscriber instance ready for use
    """
    return TranscriptionServiceFactory.create_transcriber()