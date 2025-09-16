"""
Domain Entities - Core business objects
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from pathlib import Path


@dataclass
class Video:
    """Video entity representing a video file"""
    id: str
    path: Path
    duration: float  # seconds
    fps: float
    resolution: tuple[int, int]  # (width, height)
    codec: str
    size: int  # bytes
    created_at: datetime
    metadata: dict = None


@dataclass
class AudioSegment:
    """Audio segment with speaker information"""
    start_time: float
    end_time: float
    speaker_id: Optional[str] = None
    text: Optional[str] = None
    confidence: float = 0.0


@dataclass
class Speaker:
    """Speaker entity"""
    id: str
    name: Optional[str] = None
    segments: List[AudioSegment] = None
    total_duration: float = 0.0

    def __post_init__(self):
        if self.segments is None:
            self.segments = []


@dataclass
class EditOperation:
    """Video edit operation"""
    type: str  # cut, trim, merge, effect
    start_time: float
    end_time: float
    parameters: dict = None


@dataclass
class Project:
    """Project entity containing all video editing data"""
    id: str
    name: str
    video: Optional[Video] = None
    speakers: List[Speaker] = None
    operations: List[EditOperation] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.speakers is None:
            self.speakers = []
        if self.operations is None:
            self.operations = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()