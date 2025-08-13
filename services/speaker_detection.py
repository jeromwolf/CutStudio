"""
통합 화자 감지 서비스
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime


class UnifiedSpeakerDetector:
    """
    여러 화자 감지 모듈을 통합 관리하는 서비스 클래스
    
    지원 모드:
    - 'fast': 빠른 처리 (practical_speaker_detector)
    - 'balanced': 균형잡힌 성능 (speaker_detector)
    - 'accurate': 높은 정확도 (enhanced_speaker_detector)
    - 'best': 최고 품질 (huggingface_speaker_detector)
    - 'auto': 자동 선택
    """
    
    def __init__(self):
        self.detectors = {}
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """사용 가능한 화자 감지기들을 초기화합니다."""
        # 최적화된 감지기 (긴 오디오 파일용)
        try:
            from services.speaker_detectors.optimized_speaker_detector import OptimizedSpeakerDetector
            self.detectors['optimized'] = {
                'detector': OptimizedSpeakerDetector(base_detector='practical'),
                'name': '최적화 감지기 (긴 파일용)',
                'description': '긴 오디오 파일에 최적화',
                'estimated_time_factor': 0.2
            }
        except Exception as e:
            print(f"최적화 감지기 로드 실패: {e}")
        
        # 기본 감지기는 항상 로드
        try:
            from services.speaker_detectors.speaker_detector import SpeakerDetector
            self.detectors['balanced'] = {
                'detector': SpeakerDetector(),
                'name': '균형잡힌 감지기',
                'description': '안정성과 성능의 균형',
                'estimated_time_factor': 0.3  # 비디오 길이 대비 처리 시간 비율
            }
        except Exception as e:
            st.warning(f"기본 화자 감지기 로드 실패: {str(e)}")
        
        # 실용적 감지기 (빠른 처리)
        try:
            from services.speaker_detectors.practical_speaker_detector import PracticalSpeakerDetector
            self.detectors['fast'] = {
                'detector': PracticalSpeakerDetector(),
                'name': '빠른 감지기',
                'description': '빠른 처리 속도',
                'estimated_time_factor': 0.1
            }
        except:
            pass
        
        # 고급 감지기 (높은 정확도)
        try:
            from services.speaker_detectors.enhanced_speaker_detector import EnhancedSpeakerDetector
            self.detectors['accurate'] = {
                'detector': EnhancedSpeakerDetector(),
                'name': '정밀 감지기',
                'description': '높은 정확도',
                'estimated_time_factor': 0.5
            }
        except:
            pass
        
        # HuggingFace 감지기 (최고 품질)
        if os.getenv('HUGGINGFACE_TOKEN'):
            try:
                from services.speaker_detectors.huggingface_speaker_detector import HuggingFaceSpeakerDetector
                self.detectors['best'] = {
                    'detector': HuggingFaceSpeakerDetector(),
                    'name': 'AI 감지기 (HuggingFace)',
                    'description': '최고 정확도 (95%)',
                    'estimated_time_factor': 0.8
                }
            except:
                pass
    
    def get_available_modes(self) -> Dict[str, Dict[str, str]]:
        """
        사용 가능한 감지 모드들을 반환합니다.
        
        Returns:
            모드별 정보 딕셔너리
        """
        modes = {}
        for mode, info in self.detectors.items():
            modes[mode] = {
                'name': info['name'],
                'description': info['description']
            }
        
        # 자동 모드 추가
        modes['auto'] = {
            'name': '자동 선택',
            'description': '비디오 길이에 따라 최적 모드 선택'
        }
        
        return modes
    
    def detect_speakers(
        self,
        video_path: str,
        mode: str = 'auto',
        num_speakers: Optional[int] = None,
        min_duration: float = 2.0,
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        화자를 감지합니다.
        
        Args:
            video_path: 비디오 파일 경로
            mode: 감지 모드 ('fast', 'balanced', 'accurate', 'best', 'auto')
            num_speakers: 예상 화자 수 (None이면 자동 감지)
            min_duration: 최소 세그먼트 길이
            progress_callback: 진행 상황 콜백
            **kwargs: 추가 매개변수
            
        Returns:
            화자 세그먼트 리스트
        """
        # 자동 모드 선택
        if mode == 'auto':
            mode = self._select_best_mode(video_path)
            if progress_callback:
                progress_callback(0, 100, f"자동 선택 모드: {self.detectors[mode]['name']}")
        
        # 선택된 감지기가 없으면 기본값 사용
        if mode not in self.detectors:
            available_modes = list(self.detectors.keys())
            if not available_modes:
                raise ValueError("사용 가능한 화자 감지기가 없습니다.")
            mode = available_modes[0]
            st.warning(f"요청한 모드를 사용할 수 없어 '{self.detectors[mode]['name']}'를 사용합니다.")
        
        # 화자 감지 실행
        detector_info = self.detectors[mode]
        detector = detector_info['detector']
        
        try:
            # 감지기별 매개변수 조정
            detect_params = {
                'video_path': video_path,
                'min_duration': min_duration
            }
            
            if num_speakers is not None:
                detect_params['num_speakers'] = num_speakers
            
            # 진행 상황 콜백 지원 여부 확인
            if progress_callback and hasattr(detector.detect_speakers, '__code__'):
                if 'progress_callback' in detector.detect_speakers.__code__.co_varnames:
                    detect_params['progress_callback'] = progress_callback
            
            # 추가 매개변수 전달
            detect_params.update(kwargs)
            
            # 화자 감지 실행
            segments = detector.detect_speakers(**detect_params)
            
            # None 반환 시 빈 리스트로 처리
            if segments is None:
                st.warning(f"{detector_info['name']}에서 화자를 감지하지 못했습니다.")
                segments = []
            
            # 결과 검증 및 정규화
            segments = self._normalize_segments(segments)
            
            return segments
            
        except Exception as e:
            st.error(f"화자 감지 실패 ({detector_info['name']}): {str(e)}")
            
            # 폴백 시도
            if mode != 'balanced' and 'balanced' in self.detectors:
                st.info("기본 감지기로 재시도합니다...")
                return self.detect_speakers(
                    video_path, 
                    mode='balanced',
                    num_speakers=num_speakers,
                    min_duration=min_duration,
                    progress_callback=progress_callback,
                    **kwargs
                )
            
            raise
    
    def _select_best_mode(self, video_path: str) -> str:
        """
        비디오 특성에 따라 최적의 감지 모드를 선택합니다.
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            선택된 모드
        """
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            from utils import is_audio_file
            
            # 미디어 길이 확인
            if is_audio_file(video_path):
                with AudioFileClip(video_path) as audio:
                    duration_minutes = audio.duration / 60
            else:
                with VideoFileClip(video_path) as video:
                    duration_minutes = video.duration / 60
            
            # 길이에 따른 모드 선택
            if duration_minutes > 30:
                # 매우 긴 파일: 최적화 감지기
                if 'optimized' in self.detectors:
                    return 'optimized'
            elif duration_minutes < 5:
                # 짧은 영상: 정확도 우선
                if 'best' in self.detectors:
                    return 'best'
                elif 'accurate' in self.detectors:
                    return 'accurate'
            elif duration_minutes < 20:
                # 중간 길이: 균형
                if 'balanced' in self.detectors:
                    return 'balanced'
            else:
                # 긴 영상: 속도 우선
                if 'fast' in self.detectors:
                    return 'fast'
            
            # 기본값
            return list(self.detectors.keys())[0]
            
        except Exception:
            # 오류 시 기본 모드
            return 'balanced' if 'balanced' in self.detectors else list(self.detectors.keys())[0]
    
    def _normalize_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        세그먼트 형식을 정규화합니다.
        
        Args:
            segments: 원본 세그먼트 리스트 (None일 수 있음)
            
        Returns:
            정규화된 세그먼트 리스트
        """
        # None이나 빈 리스트 처리
        if not segments:
            return []
        
        normalized = []
        
        for seg in segments:
            # 필수 필드 확인
            if not all(key in seg for key in ['speaker', 'start', 'end']):
                continue
            
            # duration 계산
            if 'duration' not in seg:
                seg['duration'] = seg['end'] - seg['start']
            
            # 화자 이름 정규화
            if not seg['speaker'].startswith('SPEAKER_'):
                seg['speaker'] = f"SPEAKER_{seg['speaker']}"
            
            normalized.append(seg)
        
        # 시간순 정렬
        normalized.sort(key=lambda x: x['start'])
        
        return normalized
    
    def estimate_processing_time(
        self,
        video_duration: float,
        mode: str = 'auto'
    ) -> Tuple[float, float]:
        """
        예상 처리 시간을 계산합니다.
        
        Args:
            video_duration: 비디오 길이 (초)
            mode: 감지 모드
            
        Returns:
            (최소 시간, 최대 시간) 튜플 (초)
        """
        if mode == 'auto':
            # 자동 모드는 평균값 사용
            factors = [info['estimated_time_factor'] for info in self.detectors.values()]
            avg_factor = sum(factors) / len(factors) if factors else 0.3
            min_time = video_duration * avg_factor * 0.8
            max_time = video_duration * avg_factor * 1.2
        else:
            if mode in self.detectors:
                factor = self.detectors[mode]['estimated_time_factor']
                min_time = video_duration * factor * 0.8
                max_time = video_duration * factor * 1.2
            else:
                # 기본값
                min_time = video_duration * 0.3
                max_time = video_duration * 0.5
        
        return (min_time, max_time)
    
    def get_mode_description(self, mode: str) -> str:
        """
        모드에 대한 설명을 반환합니다.
        
        Args:
            mode: 감지 모드
            
        Returns:
            모드 설명
        """
        if mode == 'auto':
            return "비디오 길이에 따라 자동으로 최적 모드를 선택합니다."
        elif mode in self.detectors:
            info = self.detectors[mode]
            return f"{info['name']}: {info['description']}"
        else:
            return "알 수 없는 모드"
    
    def export_segments(
        self,
        segments: List[Dict[str, Any]],
        format: str = 'json',
        output_path: Optional[str] = None
    ) -> str:
        """
        화자 세그먼트를 다양한 형식으로 내보냅니다.
        
        Args:
            segments: 화자 세그먼트 리스트
            format: 출력 형식 ('json', 'csv', 'rttm')
            output_path: 저장 경로 (None이면 문자열 반환)
            
        Returns:
            포맷된 문자열
        """
        if format == 'json':
            import json
            content = json.dumps(segments, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(
                output, 
                fieldnames=['speaker', 'start', 'end', 'duration']
            )
            writer.writeheader()
            writer.writerows(segments)
            content = output.getvalue()
        
        elif format == 'rttm':
            # RTTM 형식 (화자 분리 표준 형식)
            lines = []
            for seg in segments:
                lines.append(
                    f"SPEAKER video 1 {seg['start']:.3f} {seg['duration']:.3f} "
                    f"<NA> <NA> {seg['speaker']} <NA> <NA>"
                )
            content = '\n'.join(lines)
        
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
        
        # 파일로 저장
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return content