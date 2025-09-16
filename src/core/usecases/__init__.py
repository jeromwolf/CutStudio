"""
Use Cases - Application business logic
"""

from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass

from ..entities import Project, Video, Speaker, AudioSegment
from ..interfaces import (
    VideoProcessorInterface,
    AudioAnalyzerInterface,
    AIServiceInterface,
    StorageInterface
)


@dataclass
class ProcessVideoUseCase:
    """Use case for processing a video file"""

    video_processor: VideoProcessorInterface
    audio_analyzer: AudioAnalyzerInterface
    storage: StorageInterface

    def execute(self, video_path: Path, project_name: str) -> Project:
        """
        Process a video file:
        1. Load video
        2. Extract audio
        3. Detect speakers
        4. Create project
        """
        # Load video
        video = self.video_processor.load_video(video_path)

        # Extract and analyze audio
        audio = self.audio_analyzer.extract_audio(video)
        speakers = self.audio_analyzer.detect_speakers(audio)

        # Create project
        import uuid
        from datetime import datetime

        project = Project(
            id=str(uuid.uuid4()),
            name=project_name,
            video=video,
            speakers=speakers,
            created_at=datetime.now()
        )

        return project


@dataclass
class TranscribeVideoUseCase:
    """Use case for transcribing video audio"""

    audio_analyzer: AudioAnalyzerInterface
    ai_service: AIServiceInterface

    def execute(self, project: Project, language: str = "auto") -> Project:
        """
        Transcribe video audio:
        1. Extract audio from video
        2. Transcribe audio
        3. Assign transcriptions to speakers
        4. Generate summaries
        """
        # Extract audio
        audio = self.audio_analyzer.extract_audio(project.video)

        # Transcribe
        segments = self.audio_analyzer.transcribe_audio(audio, language)

        # Assign to speakers
        for segment in segments:
            # Find matching speaker by time overlap
            for speaker in project.speakers:
                for speaker_segment in speaker.segments:
                    if (segment.start_time >= speaker_segment.start_time and
                        segment.end_time <= speaker_segment.end_time):
                        speaker_segment.text = segment.text
                        speaker_segment.confidence = segment.confidence
                        break

        return project


@dataclass
class ExportVideoUseCase:
    """Use case for exporting edited video"""

    video_processor: VideoProcessorInterface
    storage: StorageInterface

    def execute(self, project: Project, output_path: Path, format: str = "mp4") -> Path:
        """
        Export edited video:
        1. Apply edit operations
        2. Export to file
        3. Save to storage
        """
        video = project.video

        # Apply operations
        for operation in project.operations:
            if operation.type == "cut":
                video = self.video_processor.cut_video(
                    video,
                    operation.start_time,
                    operation.end_time
                )
            elif operation.type == "trim":
                # Handle trim operation
                pass

        # Export video
        exported_path = self.video_processor.export_video(video, output_path, format)

        return exported_path


@dataclass
class SummarizeContentUseCase:
    """Use case for generating AI summaries"""

    ai_service: AIServiceInterface

    def execute(self, project: Project) -> dict:
        """
        Generate summaries for project:
        1. Summarize each speaker's content
        2. Generate overall summary
        3. Extract keywords
        """
        summaries = {}

        # Summarize each speaker
        for speaker in project.speakers:
            # Combine all text from speaker
            full_text = " ".join([
                seg.text for seg in speaker.segments
                if seg.text
            ])

            if full_text:
                summaries[speaker.id] = {
                    "summary": self.ai_service.summarize_text(full_text),
                    "keywords": self.ai_service.extract_keywords(full_text),
                    "sentiment": self.ai_service.analyze_sentiment(full_text)
                }

        # Generate overall summary
        all_text = " ".join([
            seg.text for speaker in project.speakers
            for seg in speaker.segments if seg.text
        ])

        if all_text:
            summaries["overall"] = {
                "summary": self.ai_service.summarize_text(all_text),
                "keywords": self.ai_service.extract_keywords(all_text),
                "sentiment": self.ai_service.analyze_sentiment(all_text)
            }

        return summaries