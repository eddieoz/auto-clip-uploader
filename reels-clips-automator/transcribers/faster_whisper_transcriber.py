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
        # Language configuration - None means auto-detect, or specify language code (e.g., 'pt', 'en', 'es')
        self._language = os.getenv('TRANSCRIPTION_LANGUAGE', None)

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
                try:
                    self._model = WhisperModel(
                        self._model_size,
                        device=self._device,
                        compute_type=self._compute_type
                    )
                except Exception as cuda_error:
                    error_msg = str(cuda_error).lower()
                    if self._device in ['cuda', 'auto'] and any(term in error_msg for term in ['cuda', 'cudnn', 'gpu', 'nvml']):
                        logger.warning(f"CUDA error detected: {cuda_error}")
                        logger.info("Falling back to CPU processing...")
                        self._device = 'cpu'
                        self._compute_type = 'int8'  # More efficient for CPU
                        self._model = WhisperModel(
                            self._model_size,
                            device=self._device,
                            compute_type=self._compute_type
                        )
                        logger.info(f"Successfully switched to CPU mode: {self._device} with {self._compute_type}")
                    else:
                        raise

            logger.info(f"Transcribing audio with Faster Whisper: {audio_path}")
            logger.info(f"Model configuration: {self._model_size} on {self._device} with {self._compute_type}")

            # Use configured language or auto-detect
            if self._language:
                logger.info(f"Using configured language: {self._language}")
                segments, info = self._model.transcribe(audio_path, language=self._language)
            else:
                logger.info("Using automatic language detection")
                segments, info = self._model.transcribe(audio_path)

            logger.info(f"Transcription info - Language: {info.language}, Duration: {info.duration:.2f}s")

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
            # Test if we can create a model instance (handles CUDA issues)
            if self._device in ['cuda', 'auto']:
                try:
                    # Quick test to see if CUDA is working
                    test_model = WhisperModel('tiny', device='cuda', compute_type='float16')
                    del test_model  # Clean up
                    logger.info("CUDA acceleration available for Faster Whisper")
                except Exception as cuda_error:
                    logger.warning(f"CUDA not available for Faster Whisper: {cuda_error}")
                    logger.info("Faster Whisper will fall back to CPU if needed")
            return True
        except ImportError as e:
            logger.warning(f"Faster Whisper package is not available: {e}")
            return False

    @property
    def name(self) -> str:
        """Return the name of this transcriber."""
        return f"Faster Whisper ({self._model_size})"