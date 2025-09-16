"""
CutStudio Pro Engine - 확장 가능한 코어 시스템
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
import logging
from abc import ABC, abstractmethod

# 이벤트 시스템
class EventType(Enum):
    """시스템 이벤트 타입"""
    MEDIA_LOADED = "media_loaded"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_PROGRESS = "analysis_progress"
    ANALYSIS_COMPLETED = "analysis_completed"
    CUT_STARTED = "cut_started"
    CUT_COMPLETED = "cut_completed"
    EXPORT_STARTED = "export_started"
    EXPORT_COMPLETED = "export_completed"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class Event:
    """시스템 이벤트"""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class EventBus:
    """이벤트 버스 - 모듈 간 통신"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """이벤트 구독"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """이벤트 구독 해제"""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)
    
    def emit(self, event: Event):
        """이벤트 발생"""
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    logging.error(f"Event callback error: {e}")


# 플러그인 시스템
class PluginInterface(ABC):
    """플러그인 인터페이스"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """플러그인 이름"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """플러그인 버전"""
        pass
    
    @abstractmethod
    def initialize(self, engine: 'CutStudioEngine'):
        """플러그인 초기화"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """플러그인 정리"""
        pass


class AnalysisPlugin(PluginInterface):
    """분석 플러그인 기본 클래스"""
    
    @abstractmethod
    def analyze(self, media_path: Path, options: Dict[str, Any]) -> Dict[str, Any]:
        """미디어 분석 수행"""
        pass


class ExportPlugin(PluginInterface):
    """내보내기 플러그인 기본 클래스"""
    
    @abstractmethod
    def export(self, data: Dict[str, Any], output_path: Path) -> bool:
        """데이터 내보내기"""
        pass


class PluginManager:
    """플러그인 관리자"""
    
    def __init__(self):
        self._plugins: Dict[str, PluginInterface] = {}
        self._analysis_plugins: Dict[str, AnalysisPlugin] = {}
        self._export_plugins: Dict[str, ExportPlugin] = {}
    
    def register_plugin(self, plugin: PluginInterface):
        """플러그인 등록"""
        self._plugins[plugin.name] = plugin
        
        # 타입별 분류
        if isinstance(plugin, AnalysisPlugin):
            self._analysis_plugins[plugin.name] = plugin
        elif isinstance(plugin, ExportPlugin):
            self._export_plugins[plugin.name] = plugin
    
    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """플러그인 조회"""
        return self._plugins.get(name)
    
    def get_analysis_plugins(self) -> Dict[str, AnalysisPlugin]:
        """분석 플러그인 목록"""
        return self._analysis_plugins.copy()
    
    def get_export_plugins(self) -> Dict[str, ExportPlugin]:
        """내보내기 플러그인 목록"""
        return self._export_plugins.copy()


# 프로젝트 데이터 구조
@dataclass
class MediaInfo:
    """미디어 파일 정보"""
    path: Path
    duration: float
    format: str
    size: tuple[int, int] = (0, 0)  # (width, height) for video
    fps: Optional[float] = None
    sample_rate: Optional[int] = None


@dataclass
class SpeakerSegment:
    """화자 구간 정보"""
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float = 0.0
    transcript: str = ""
    summary: str = ""
    thumbnail_path: Optional[Path] = None


@dataclass
class ProjectData:
    """프로젝트 데이터"""
    media_info: Optional[MediaInfo] = None
    speakers: Dict[str, List[SpeakerSegment]] = field(default_factory=dict)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    export_history: List[Dict[str, Any]] = field(default_factory=list)


# 메인 엔진
class CutStudioEngine:
    """
    CutStudio Pro 확장 가능한 엔진
    
    Features:
    - 이벤트 기반 아키텍처
    - 플러그인 시스템
    - 비동기 처리
    - 상태 관리
    """
    
    def __init__(self):
        self.event_bus = EventBus()
        self.plugin_manager = PluginManager()
        self.project_data = ProjectData()
        self._processing = False
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 기본 플러그인 로드
        self._load_core_plugins()
    
    def _load_core_plugins(self):
        """코어 플러그인 로드"""
        # TODO: 기본 화자 감지, 음성 인식, 요약 플러그인 등록
        pass
    
    def load_media(self, media_path: Path) -> bool:
        """미디어 파일 로드"""
        try:
            # 미디어 정보 추출
            from utils import get_media_info
            info = get_media_info(str(media_path))
            
            self.project_data.media_info = MediaInfo(
                path=media_path,
                duration=info.get('duration', 0),
                format=media_path.suffix,
                size=(info.get('width', 0), info.get('height', 0)),
                fps=info.get('fps'),
                sample_rate=info.get('sample_rate')
            )
            
            # 이벤트 발생
            self.event_bus.emit(Event(
                EventType.MEDIA_LOADED,
                {"media_info": self.project_data.media_info}
            ))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Media loading failed: {e}")
            self.event_bus.emit(Event(
                EventType.ERROR_OCCURRED,
                {"error": str(e), "context": "media_loading"}
            ))
            return False
    
    async def analyze_speakers(self, options: Dict[str, Any] = None) -> bool:
        """화자 구분 분석"""
        if not self.project_data.media_info:
            raise ValueError("No media loaded")
        
        self._processing = True
        options = options or {}
        
        try:
            # 분석 시작 이벤트
            self.event_bus.emit(Event(EventType.ANALYSIS_STARTED))
            
            # 분석 플러그인 실행 (기본: UnifiedSpeakerDetector)
            from services.speaker_detection import UnifiedSpeakerDetector
            detector = UnifiedSpeakerDetector()
            
            # 진행률 콜백
            def progress_callback(progress: float):
                self.event_bus.emit(Event(
                    EventType.ANALYSIS_PROGRESS,
                    {"progress": progress}
                ))
            
            # 화자 감지 실행
            segments = await asyncio.to_thread(
                detector.detect_speakers,
                str(self.project_data.media_info.path),
                **options
            )
            
            # 결과 저장
            speakers = {}
            for segment in segments:
                speaker_id = segment.get('speaker', 'unknown')
                if speaker_id not in speakers:
                    speakers[speaker_id] = []
                
                speakers[speaker_id].append(SpeakerSegment(
                    speaker_id=speaker_id,
                    start_time=segment.get('start', 0),
                    end_time=segment.get('end', 0),
                    confidence=segment.get('confidence', 0)
                ))
            
            self.project_data.speakers = speakers
            
            # 완료 이벤트
            self.event_bus.emit(Event(
                EventType.ANALYSIS_COMPLETED,
                {"speakers": speakers}
            ))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Speaker analysis failed: {e}")
            self.event_bus.emit(Event(
                EventType.ERROR_OCCURRED,
                {"error": str(e), "context": "speaker_analysis"}
            ))
            return False
        
        finally:
            self._processing = False
    
    async def generate_summaries(self) -> bool:
        """화자별 요약 생성"""
        if not self.project_data.speakers:
            raise ValueError("No speaker analysis available")
        
        try:
            from services.summarization import SummarizationService
            summarizer = SummarizationService()
            
            # 각 화자별 요약 생성
            for speaker_id, segments in self.project_data.speakers.items():
                for segment in segments:
                    if segment.transcript:
                        summary = await asyncio.to_thread(
                            summarizer.summarize_text,
                            segment.transcript
                        )
                        segment.summary = summary
            
            return True
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return False
    
    def cut_speaker_segments(self, speaker_id: str, output_dir: Path) -> List[Path]:
        """화자별 구간 자르기"""
        if speaker_id not in self.project_data.speakers:
            raise ValueError(f"Speaker {speaker_id} not found")
        
        try:
            from video_editor import VideoEditor
            editor = VideoEditor()
            editor.load_video(str(self.project_data.media_info.path))
            
            self.event_bus.emit(Event(EventType.CUT_STARTED, {"speaker_id": speaker_id}))
            
            output_files = []
            segments = self.project_data.speakers[speaker_id]
            
            for i, segment in enumerate(segments):
                output_path = editor.cut_video(segment.start_time, segment.end_time)
                if output_path:
                    # 의미있는 이름으로 변경
                    final_path = output_dir / f"{speaker_id}_segment_{i+1}.mp4"
                    Path(output_path).rename(final_path)
                    output_files.append(final_path)
            
            self.event_bus.emit(Event(
                EventType.CUT_COMPLETED,
                {"speaker_id": speaker_id, "files": output_files}
            ))
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"Cut operation failed: {e}")
            self.event_bus.emit(Event(
                EventType.ERROR_OCCURRED,
                {"error": str(e), "context": "cutting"}
            ))
            return []
    
    def get_project_summary(self) -> Dict[str, Any]:
        """프로젝트 요약 정보"""
        if not self.project_data.media_info:
            return {}
        
        total_segments = sum(len(segments) for segments in self.project_data.speakers.values())
        
        return {
            "media_info": {
                "duration": self.project_data.media_info.duration,
                "format": self.project_data.media_info.format,
                "size": self.project_data.media_info.size
            },
            "analysis": {
                "speaker_count": len(self.project_data.speakers),
                "total_segments": total_segments,
                "speakers": list(self.project_data.speakers.keys())
            },
            "processing": {
                "is_processing": self._processing
            }
        }
    
    def register_plugin(self, plugin: PluginInterface):
        """플러그인 등록"""
        plugin.initialize(self)
        self.plugin_manager.register_plugin(plugin)
        self.logger.info(f"Plugin registered: {plugin.name} v{plugin.version}")
    
    def cleanup(self):
        """리소스 정리"""
        for plugin in self.plugin_manager._plugins.values():
            plugin.cleanup()