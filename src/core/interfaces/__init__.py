"""
Core Interfaces (Ports) - Define contracts for external dependencies
"""

from abc import ABC, abstractmethod
from typing import List, Optional, BinaryIO
from pathlib import Path

from ..entities import Video, AudioSegment, Speaker


class VideoProcessorInterface(ABC):
    """Interface for video processing operations"""

    @abstractmethod
    def load_video(self, path: Path) -> Video:
        """Load video file and extract metadata"""
        pass

    @abstractmethod
    def cut_video(self, video: Video, start: float, end: float) -> Video:
        """Cut video from start to end time"""
        pass

    @abstractmethod
    def trim_video(self, video: Video, segments: List[tuple[float, float]]) -> Video:
        """Trim video keeping only specified segments"""
        pass

    @abstractmethod
    def export_video(self, video: Video, output_path: Path, format: str = "mp4") -> Path:
        """Export processed video to file"""
        pass


class AudioAnalyzerInterface(ABC):
    """Interface for audio analysis operations"""

    @abstractmethod
    def extract_audio(self, video: Video) -> BinaryIO:
        """Extract audio from video"""
        pass

    @abstractmethod
    def detect_speakers(self, audio: BinaryIO) -> List[Speaker]:
        """Detect and separate speakers in audio"""
        pass

    @abstractmethod
    def transcribe_audio(self, audio: BinaryIO, language: str = "auto") -> List[AudioSegment]:
        """Transcribe audio to text with timestamps"""
        pass


class AIServiceInterface(ABC):
    """Interface for AI services"""

    @abstractmethod
    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """Generate summary of text"""
        pass

    @abstractmethod
    def extract_keywords(self, text: str, count: int = 10) -> List[str]:
        """Extract keywords from text"""
        pass

    @abstractmethod
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of text"""
        pass


class StorageInterface(ABC):
    """Interface for file storage operations"""

    @abstractmethod
    def save_file(self, file: BinaryIO, path: Path) -> Path:
        """Save file to storage"""
        pass

    @abstractmethod
    def load_file(self, path: Path) -> BinaryIO:
        """Load file from storage"""
        pass

    @abstractmethod
    def delete_file(self, path: Path) -> bool:
        """Delete file from storage"""
        pass

    @abstractmethod
    def list_files(self, directory: Path) -> List[Path]:
        """List files in directory"""
        pass