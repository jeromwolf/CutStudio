"""
Analyze Module - AI 기반 미디어 분석
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# 기존 서비스 임포트
from services.speaker_detection import UnifiedSpeakerDetector
from services.speech_processing import SpeechProcessor
from services.summarization import SummarizationService


class AnalyzeModule:
    """
    미디어 분석 모듈
    
    Features:
    - Speaker diarization
    - Speech recognition
    - Content summarization
    - Keyword extraction
    """
    
    def __init__(self):
        """모듈 초기화"""
        self.speaker_detector = UnifiedSpeakerDetector()
        self.speech_processor = None
        self.summarizer = SummarizationService()
        
        # 음성 인식기 초기화
        try:
            from services.speech.speech_transcriber import SpeechRecognizer
            recognizer = SpeechRecognizer()
            self.speech_processor = SpeechProcessor(recognizer)
        except ImportError:
            print("Warning: Speech recognition not available")
    
    def analyze(
        self,
        media_path: Path,
        mode: str = "balanced",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        종합 미디어 분석
        
        Args:
            media_path: 미디어 파일 경로
            mode: 분석 모드 (fast, balanced, accurate)
            progress_callback: 진행 상황 콜백
            
        Returns:
            분석 결과
        """
        results = {
            "media_path": str(media_path),
            "mode": mode,
            "timestamp": time.time(),
            "speakers": {},
            "summary": {},
            "keywords": []
        }
        
        # 1단계: 화자 감지 (40%)
        if progress_callback:
            progress_callback(0, 100, "화자 감지 중...")
        
        speaker_segments = self._detect_speakers(media_path, mode)
        
        # 화자 감지 결과 검증
        if not speaker_segments:
            print(f"Warning: No speakers detected for {media_path}")
            # 기본 세그먼트를 반환하도록 했으므로 이 경우는 드물 것
        
        num_speakers = len(set(s['speaker'] for s in speaker_segments)) if speaker_segments else 0
        
        if progress_callback:
            progress_callback(40, 100, f"{num_speakers}명의 화자 감지됨")
        
        # 2단계: 음성 인식 (40%)
        if self.speech_processor and speaker_segments:
            if progress_callback:
                progress_callback(40, 100, "음성 인식 중...")
            
            recognized_segments = self._recognize_speech(
                media_path,
                speaker_segments,
                progress_callback
            )
        else:
            recognized_segments = speaker_segments
        
        if progress_callback:
            progress_callback(80, 100, "음성 인식 완료")
        
        # 3단계: 내용 분석 (20%)
        if progress_callback:
            progress_callback(80, 100, "내용 분석 중...")
        
        # 화자별 결과 정리
        speakers_data = self._organize_by_speaker(recognized_segments)
        
        # AI 요약 생성
        for speaker_id, data in speakers_data.items():
            if data["text"]:
                # 요약 생성
                summary = self.summarizer.summarize_text(data["text"], max_length=200)
                data["summary"] = summary
                
                # 키워드 추출
                keywords = self._extract_keywords(data["text"])
                data["keywords"] = keywords
        
        results["speakers"] = speakers_data
        
        # 전체 요약
        all_text = " ".join([d["text"] for d in speakers_data.values() if d["text"]])
        if all_text:
            results["summary"] = {
                "full": self.summarizer.summarize_text(all_text, max_length=300),
                "keywords": self._extract_keywords(all_text, top_n=10)
            }
        
        if progress_callback:
            progress_callback(100, 100, "분석 완료!")
        
        return results
    
    def _detect_speakers(self, media_path: Path, mode: str) -> List[Dict[str, Any]]:
        """화자 감지"""
        # 모드별 설정 (UnifiedSpeakerDetector의 실제 모드와 일치하도록 수정)
        settings = {
            "fast": {"method": "fast", "min_duration": 3.0},
            "balanced": {"method": "balanced", "min_duration": 2.0},
            "accurate": {"method": "accurate", "min_duration": 1.0}
        }
        
        config = settings.get(mode, settings["balanced"])
        
        try:
            # 화자 감지 실행
            segments = self.speaker_detector.detect_speakers(
                str(media_path),
                mode=config["method"],
                min_duration=config["min_duration"]
            )
            
            # None이나 빈 결과 처리
            if segments is None:
                print(f"Warning: Speaker detection returned None for {media_path}")
                segments = []
            
            # 결과 검증
            if not segments:
                print(f"Warning: No speaker segments detected for {media_path}")
                # 폴백: 전체 오디오를 하나의 세그먼트로 처리
                try:
                    from moviepy.editor import VideoFileClip, AudioFileClip
                    from utils import is_audio_file
                    
                    if is_audio_file(str(media_path)):
                        with AudioFileClip(str(media_path)) as audio:
                            duration = audio.duration
                    else:
                        with VideoFileClip(str(media_path)) as video:
                            duration = video.duration
                    
                    # 최소한 하나의 세그먼트 생성
                    segments = [{
                        'speaker': 'SPEAKER_0',
                        'start': 0.0,
                        'end': duration,
                        'duration': duration,
                        'confidence': 0.5
                    }]
                    print(f"Created fallback segment with duration {duration}s")
                except Exception as e:
                    print(f"Failed to create fallback segment: {e}")
            
            return segments
            
        except Exception as e:
            print(f"Error in _detect_speakers: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _recognize_speech(
        self,
        media_path: Path,
        segments: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """음성 인식"""
        if not self.speech_processor:
            return segments
        
        # 진행 상황 래퍼
        def speech_progress(current, total, message):
            if progress_callback:
                # 40-80% 구간에서 진행
                progress = 40 + (40 * current / total)
                progress_callback(int(progress), 100, message)
        
        # 음성 인식 실행
        recognized = self.speech_processor.execute_recognition(
            str(media_path),
            segments,
            model_size="tiny",  # 빠른 처리를 위해 작은 모델 사용
            progress_callback=speech_progress
        )
        
        return recognized
    
    def _organize_by_speaker(self, segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """화자별로 세그먼트 정리"""
        speakers = {}
        
        for segment in segments:
            speaker_id = segment["speaker"]
            
            if speaker_id not in speakers:
                speakers[speaker_id] = {
                    "speaker_id": speaker_id,
                    "segments": [],
                    "total_duration": 0,
                    "text": "",
                    "summary": "",
                    "keywords": []
                }
            
            speakers[speaker_id]["segments"].append(segment)
            speakers[speaker_id]["total_duration"] += segment["duration"]
            
            # 텍스트 결합
            if "text" in segment and segment["text"]:
                speakers[speaker_id]["text"] += " " + segment["text"]
        
        # 텍스트 정리
        for speaker_data in speakers.values():
            speaker_data["text"] = speaker_data["text"].strip()
            speaker_data["duration"] = round(speaker_data["total_duration"], 1)
        
        return speakers
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """키워드 추출 (간단한 버전)"""
        if not text:
            return []
        
        # 간단한 TF 기반 키워드 추출
        import re
        from collections import Counter
        
        # 단어 추출 (한글, 영문)
        words = re.findall(r'\b[가-힣]+\b|\b[a-zA-Z]+\b', text.lower())
        
        # 불용어 제거 (간단한 버전)
        stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '와', '과', 
                     'the', 'is', 'at', 'on', 'and', 'a', 'to', 'in', 'of'}
        words = [w for w in words if w not in stopwords and len(w) > 1]
        
        # 빈도 계산
        word_freq = Counter(words)
        
        # 상위 키워드 반환
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def analyze_education_context(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """교육 컨텍스트 분석"""
        edu_context = {
            "lecture_type": "unknown",
            "main_speaker": None,
            "interaction_level": "low",
            "topics": []
        }
        
        speakers = analysis_results.get("speakers", {})
        
        if speakers:
            # 주요 화자 찾기 (가장 많이 말한 사람)
            main_speaker = max(
                speakers.items(),
                key=lambda x: x[1]["total_duration"]
            )
            
            edu_context["main_speaker"] = main_speaker[0]
            
            # 상호작용 수준 평가
            speaker_count = len(speakers)
            if speaker_count == 1:
                edu_context["lecture_type"] = "monologue"
                edu_context["interaction_level"] = "none"
            elif speaker_count == 2:
                edu_context["lecture_type"] = "dialogue"
                edu_context["interaction_level"] = "medium"
            else:
                edu_context["lecture_type"] = "discussion"
                edu_context["interaction_level"] = "high"
            
            # 주제 추출 (전체 키워드에서)
            if "summary" in analysis_results:
                edu_context["topics"] = analysis_results["summary"].get("keywords", [])
        
        return edu_context