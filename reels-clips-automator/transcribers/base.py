from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseTranscriber(ABC):
    """Abstract base class for all transcription services."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe audio file and return segments.

        Args:
            audio_path: Path to the audio file to transcribe

        Returns:
            List of transcription segments with format:
            [
                {
                    "start": float,  # Start time in seconds
                    "end": float,    # End time in seconds
                    "text": str      # Transcribed text
                },
                ...
            ]
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the transcriber is available and properly configured.

        Returns:
            True if transcriber can be used, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this transcriber."""
        pass