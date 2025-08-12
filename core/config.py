"""
앱 설정 및 상수 관리
"""
from typing import Dict, List, Tuple


class AppConfig:
    """CutStudio 설정 관리 클래스"""
    
    # 지원 파일 형식
    SUPPORTED_VIDEO_FORMATS = ['mp4', 'avi', 'mov', 'mkv']
    SUPPORTED_AUDIO_FORMATS = ['mp3', 'wav', 'm4a']
    
    # Whisper 모델 설정
    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
    WHISPER_MODEL_SIZES = {
        "tiny": "39M",
        "base": "74M", 
        "small": "244M",
        "medium": "769M",
        "large": "1550M"
    }
    
    # 화자 감지 방법
    DETECTION_METHODS = {
        "허깅페이스 AI (가장 정확)": "huggingface",
        "실용적 감지기 (균형)": "practical",
        "고급 감지기 (정밀)": "advanced",
        "자동 선택": "auto",
        "간단 감지기 (빠름)": "simple"
    }
    
    # 화자 감지 예상 시간 (분/10분 영상 기준)
    DETECTION_TIME_ESTIMATES = {
        "huggingface": (6, 10),
        "practical": (3, 5),
        "advanced": (4, 6),
        "auto": (3, 6),
        "simple": (1, 2)
    }
    
    # UI 설정
    THUMBNAIL_WIDTH = 150
    TIMELINE_CARD_HEIGHT = 200
    MAX_DISPLAY_SEGMENTS = 100
    
    # 요약 설정
    MIN_SEGMENT_LENGTH_FOR_SUMMARY = 50  # 최소 텍스트 길이
    SUMMARY_MAX_LENGTH = 200
    KEYWORDS_MAX_COUNT = 5
    
    # 파일 경로
    TEMP_DIR = "temp"
    PROCESSED_DIR = "processed"
    DOWNLOADS_DIR = "downloads"
    MODELS_DIR = "pretrained_models"
    
    # 성능 설정
    VIDEO_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_CONCURRENT_THREADS = 4
    
    @staticmethod
    def get_estimated_time(method: str, video_duration_minutes: float) -> Tuple[float, float]:
        """
        화자 감지 예상 시간 계산
        
        Args:
            method: 감지 방법
            video_duration_minutes: 비디오 길이 (분)
            
        Returns:
            (최소 시간, 최대 시간) 튜플 (분)
        """
        if method not in AppConfig.DETECTION_TIME_ESTIMATES:
            return (1, 5)
        
        min_rate, max_rate = AppConfig.DETECTION_TIME_ESTIMATES[method]
        scale_factor = video_duration_minutes / 10.0
        
        return (
            min_rate * scale_factor,
            max_rate * scale_factor
        )
    
    @staticmethod
    def get_whisper_model_info(model_name: str) -> Dict[str, str]:
        """Whisper 모델 정보 반환"""
        return {
            "name": model_name,
            "size": AppConfig.WHISPER_MODEL_SIZES.get(model_name, "Unknown"),
            "description": f"{model_name.capitalize()} 모델 ({AppConfig.WHISPER_MODEL_SIZES.get(model_name, 'Unknown')})"
        }
    
    @staticmethod
    def validate_file_format(filename: str, file_type: str = "video") -> bool:
        """파일 형식 유효성 검사"""
        ext = filename.lower().split('.')[-1]
        if file_type == "video":
            return ext in AppConfig.SUPPORTED_VIDEO_FORMATS
        elif file_type == "audio":
            return ext in AppConfig.SUPPORTED_AUDIO_FORMATS
        return False