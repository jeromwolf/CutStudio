"""
최적화된 비디오 편집기
"""
from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np
from pathlib import Path
import tempfile
import os
from typing import Optional, List, Dict, Any, Tuple
from utils.performance import PerformanceOptimizer, VideoChunkProcessor
import streamlit as st


class OptimizedVideoEditor:
    """메모리 효율적인 비디오 편집기"""
    
    def __init__(self):
        self.video_clip = None
        self.video_path = None
        self.performance_optimizer = PerformanceOptimizer()
        self._video_info_cache = {}
    
    def load_video(self, video_path: str, lazy_load: bool = True):
        """
        비디오 파일 로드 (지연 로딩 지원)
        
        Args:
            video_path: 비디오 파일 경로
            lazy_load: True면 실제 사용 시점까지 로딩 지연
        """
        self.video_path = video_path
        
        if not lazy_load:
            self.video_clip = VideoFileClip(video_path)
        else:
            # 기본 정보만 캐싱
            file_hash = self.performance_optimizer.get_file_hash(video_path)
            self._video_info_cache = self.performance_optimizer.cached_video_info(
                file_hash, video_path
            )
    
    def _ensure_video_loaded(self):
        """비디오가 로드되었는지 확인"""
        if self.video_clip is None and self.video_path:
            self.video_clip = VideoFileClip(self.video_path)
    
    @property
    def duration(self) -> float:
        """비디오 길이 반환"""
        if self._video_info_cache:
            return self._video_info_cache.get('duration', 0)
        self._ensure_video_loaded()
        return self.video_clip.duration if self.video_clip else 0
    
    def trim_video_optimized(
        self, 
        start_time: float, 
        end_time: float, 
        output_path: str,
        use_chunks: bool = True
    ):
        """
        메모리 효율적인 비디오 자르기
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            output_path: 출력 경로
            use_chunks: 청크 단위 처리 여부
        """
        self._ensure_video_loaded()
        
        # 짧은 클립은 일반 처리
        if (end_time - start_time) < 60 or not use_chunks:
            trimmed = self.video_clip.subclip(start_time, end_time)
            trimmed.write_videofile(output_path, codec='libx264', audio_codec='aac')
            trimmed.close()
        else:
            # 긴 클립은 청크 단위 처리
            self._process_long_clip(start_time, end_time, output_path)
    
    def _process_long_clip(self, start_time: float, end_time: float, output_path: str):
        """긴 클립을 청크 단위로 처리"""
        from moviepy.editor import concatenate_videoclips
        
        chunk_duration = 30.0  # 30초 단위
        chunks = []
        
        try:
            current = start_time
            while current < end_time:
                chunk_end = min(current + chunk_duration, end_time)
                
                # 청크 추출
                chunk = self.video_clip.subclip(current, chunk_end)
                chunks.append(chunk)
                
                current = chunk_end
            
            # 청크 결합
            final_video = concatenate_videoclips(chunks)
            final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
            
        finally:
            # 메모리 정리
            for chunk in chunks:
                chunk.close()
            if 'final_video' in locals():
                final_video.close()
    
    def generate_speaker_profile_optimized(
        self, 
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        메모리 효율적인 화자 프로필 생성
        
        Args:
            segments: 화자 세그먼트 리스트
            
        Returns:
            화자별 프로필 정보
        """
        self._ensure_video_loaded()
        
        speaker_profiles = {}
        
        # 화자별로 그룹화
        from collections import defaultdict
        speaker_segments = defaultdict(list)
        
        for seg in segments:
            speaker_segments[seg['speaker']].append(seg)
        
        # 각 화자별 프로필 생성
        for speaker_id, segs in speaker_segments.items():
            profile = {
                'speaker_id': speaker_id,
                'has_thumbnail': False,
                'has_summary': False
            }
            
            # 대표 썸네일 생성 (첫 번째 세그먼트의 중간 지점)
            if segs:
                first_seg = segs[0]
                mid_time = (first_seg['start'] + first_seg['end']) / 2
                
                try:
                    # 메모리 효율적인 썸네일 생성
                    thumbnail = self._generate_thumbnail_optimized(mid_time)
                    if thumbnail:
                        profile['thumbnail'] = thumbnail
                        profile['has_thumbnail'] = True
                except:
                    pass
            
            # 통계 정보
            total_duration = sum(seg['duration'] for seg in segs)
            profile['summary'] = {
                'speaker_id': speaker_id,
                'total_duration': total_duration,
                'segment_count': len(segs),
                'participation_rate': (total_duration / self.duration * 100) if self.duration > 0 else 0,
                'first_appearance': min(seg['start'] for seg in segs),
                'last_appearance': max(seg['end'] for seg in segs)
            }
            profile['has_summary'] = True
            
            speaker_profiles[speaker_id] = profile
        
        return speaker_profiles
    
    def _generate_thumbnail_optimized(
        self, 
        timestamp: float, 
        size: Tuple[int, int] = (150, 100)
    ) -> Optional[Dict[str, Any]]:
        """
        메모리 효율적인 썸네일 생성
        
        Args:
            timestamp: 썸네일 시간
            size: 썸네일 크기
            
        Returns:
            썸네일 정보
        """
        import base64
        import io
        from PIL import Image
        
        try:
            # 단일 프레임만 추출
            frame = self.video_clip.get_frame(timestamp)
            
            # PIL 이미지로 변환 및 리사이즈
            pil_image = Image.fromarray(frame)
            pil_image.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Base64 인코딩
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85, optimize=True)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            # 메모리 정리
            del frame
            buffer.close()
            
            return {
                'image_base64': img_str,
                'timestamp': timestamp,
                'size': size
            }
            
        except Exception:
            return None
    
    def extract_audio_optimized(
        self, 
        output_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """
        메모리 효율적인 오디오 추출
        
        Args:
            output_path: 출력 경로
            start_time: 시작 시간 (선택)
            end_time: 종료 시간 (선택)
        """
        self._ensure_video_loaded()
        
        if self.video_clip.audio is None:
            raise ValueError("비디오에 오디오가 없습니다.")
        
        # 구간 지정
        if start_time is not None or end_time is not None:
            audio = self.video_clip.audio.subclip(
                start_time or 0,
                end_time or self.video_clip.duration
            )
        else:
            audio = self.video_clip.audio
        
        # 오디오 저장
        audio.write_audiofile(output_path, codec='mp3')
        
        # 메모리 정리
        if audio != self.video_clip.audio:
            audio.close()
    
    def apply_effect_optimized(
        self,
        effect_type: str,
        output_path: str,
        **kwargs
    ):
        """
        메모리 효율적인 효과 적용
        
        Args:
            effect_type: 효과 타입
            output_path: 출력 경로
            **kwargs: 효과별 매개변수
        """
        self._ensure_video_loaded()
        
        # 짧은 비디오는 일반 처리
        if self.duration < 60:
            self._apply_effect_normal(effect_type, output_path, **kwargs)
        else:
            # 긴 비디오는 청크 처리
            self._apply_effect_chunked(effect_type, output_path, **kwargs)
    
    def _apply_effect_normal(self, effect_type: str, output_path: str, **kwargs):
        """일반 효과 적용"""
        if effect_type == "grayscale":
            processed = self.video_clip.fx(lambda clip: clip.fx(
                lambda get_frame, t: np.dot(get_frame(t), [0.299, 0.587, 0.114])[:, :, np.newaxis].repeat(3, axis=2)
            ))
        elif effect_type == "speed":
            speed_factor = kwargs.get('speed_factor', 1.0)
            processed = self.video_clip.fx(lambda clip: clip.speedx(speed_factor))
        else:
            processed = self.video_clip
        
        processed.write_videofile(output_path, codec='libx264', audio_codec='aac')
        processed.close()
    
    def _apply_effect_chunked(self, effect_type: str, output_path: str, **kwargs):
        """청크 단위 효과 적용"""
        from moviepy.editor import concatenate_videoclips
        
        with VideoChunkProcessor(self.video_path, chunk_duration=30) as processor:
            processed_chunks = []
            
            while True:
                chunk_data = processor.get_next_chunk()
                if chunk_data is None:
                    break
                
                chunk, start, end = chunk_data
                
                # 효과 적용
                if effect_type == "grayscale":
                    processed_chunk = chunk.fx(lambda clip: clip.fx(
                        lambda get_frame, t: np.dot(get_frame(t), [0.299, 0.587, 0.114])[:, :, np.newaxis].repeat(3, axis=2)
                    ))
                else:
                    processed_chunk = chunk
                
                processed_chunks.append(processed_chunk)
            
            # 청크 결합 및 저장
            final_video = concatenate_videoclips(processed_chunks)
            final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
            
            # 메모리 정리
            for chunk in processed_chunks:
                chunk.close()
            final_video.close()
    
    def estimate_processing_requirements(self) -> Dict[str, Any]:
        """비디오 처리에 필요한 리소스 예측"""
        if not self.video_path:
            return {}
        
        return self.performance_optimizer.estimate_memory_usage(self.video_path)
    
    def close(self):
        """리소스 정리"""
        if self.video_clip:
            self.video_clip.close()
            self.video_clip = None
        
        # 캐시 정리
        self._video_info_cache.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()