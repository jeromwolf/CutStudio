"""
성능 최적화 유틸리티
"""
import os
import shutil
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Generator
import streamlit as st
from functools import lru_cache
import hashlib


class PerformanceOptimizer:
    """성능 최적화를 위한 유틸리티 클래스"""
    
    @staticmethod
    @contextmanager
    def temp_file_manager(suffix: str = '') -> Generator[str, None, None]:
        """
        임시 파일을 안전하게 관리하는 컨텍스트 매니저
        
        사용 예:
        with temp_file_manager('.mp4') as temp_path:
            # temp_path 사용
            pass
        # 자동으로 파일 삭제됨
        """
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            yield temp_path
        finally:
            # 파일이 존재하면 삭제
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @staticmethod
    def cleanup_old_files(directory: str, max_age_hours: int = 24):
        """
        오래된 파일 자동 정리
        
        Args:
            directory: 정리할 디렉토리
            max_age_hours: 최대 보관 시간 (시간)
        """
        import time
        
        if not os.path.exists(directory):
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            # 파일의 수정 시간 확인
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                
                # 오래된 파일 삭제
                if file_age > max_age_seconds:
                    try:
                        os.unlink(filepath)
                    except:
                        pass
    
    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """
        파일의 해시값 계산 (캐싱용)
        
        Args:
            filepath: 파일 경로
            
        Returns:
            파일의 MD5 해시값
        """
        hash_md5 = hashlib.md5()
        
        with open(filepath, "rb") as f:
            # 청크 단위로 읽어서 메모리 효율성 향상
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    @staticmethod
    @lru_cache(maxsize=32)
    def cached_video_info(file_hash: str, video_path: str) -> dict:
        """
        비디오 정보를 캐싱하여 반복 로딩 방지
        
        Args:
            file_hash: 파일 해시값
            video_path: 비디오 경로
            
        Returns:
            비디오 정보 딕셔너리
        """
        from moviepy.editor import VideoFileClip
        
        with VideoFileClip(video_path) as video:
            return {
                'duration': video.duration,
                'fps': video.fps,
                'size': (video.w, video.h),
                'has_audio': video.audio is not None
            }
    
    @staticmethod
    def process_video_in_chunks(
        video_path: str,
        chunk_duration: float = 60.0,
        process_func: callable = None,
        progress_callback: callable = None
    ):
        """
        비디오를 청크 단위로 처리하여 메모리 사용량 최적화
        
        Args:
            video_path: 비디오 경로
            chunk_duration: 청크 길이 (초)
            process_func: 각 청크를 처리할 함수
            progress_callback: 진행 상황 콜백
        """
        from moviepy.editor import VideoFileClip
        
        with VideoFileClip(video_path) as video:
            total_duration = video.duration
            num_chunks = int(total_duration / chunk_duration) + 1
            
            results = []
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                end_time = min((i + 1) * chunk_duration, total_duration)
                
                # 청크 추출
                chunk = video.subclip(start_time, end_time)
                
                # 처리 함수 실행
                if process_func:
                    result = process_func(chunk, start_time, end_time)
                    results.append(result)
                
                # 진행 상황 업데이트
                if progress_callback:
                    progress = (i + 1) / num_chunks
                    progress_callback(progress, f"처리 중... {i+1}/{num_chunks}")
                
                # 메모리 정리
                del chunk
            
            return results
    
    @staticmethod
    def estimate_memory_usage(video_path: str) -> dict:
        """
        비디오 처리에 필요한 예상 메모리 사용량 계산
        
        Args:
            video_path: 비디오 경로
            
        Returns:
            메모리 사용량 정보
        """
        import os
        from moviepy.editor import VideoFileClip
        
        file_size = os.path.getsize(video_path)
        
        with VideoFileClip(video_path) as video:
            # 프레임당 메모리 = width * height * 3 (RGB) * 4 (float32)
            frame_memory = video.w * video.h * 3 * 4
            
            # 초당 프레임 메모리
            memory_per_second = frame_memory * video.fps
            
            return {
                'file_size_mb': file_size / (1024 * 1024),
                'frame_memory_mb': frame_memory / (1024 * 1024),
                'memory_per_second_mb': memory_per_second / (1024 * 1024),
                'estimated_peak_memory_mb': (memory_per_second * 10) / (1024 * 1024),  # 10초 버퍼 기준
                'recommended_chunk_duration': 60 if memory_per_second > 100 * 1024 * 1024 else 120
            }
    
    @staticmethod
    def optimize_streamlit_cache():
        """
        Streamlit 캐시 최적화 설정
        """
        # 세션 상태 정리
        if 'temp_data' in st.session_state:
            # 임시 데이터 정리
            for key in list(st.session_state.keys()):
                if key.startswith('temp_') or key.startswith('_'):
                    del st.session_state[key]
        
        # 가비지 컬렉션 강제 실행
        import gc
        gc.collect()


class VideoChunkProcessor:
    """비디오 청크 단위 처리를 위한 클래스"""
    
    def __init__(self, video_path: str, chunk_duration: float = 60.0):
        self.video_path = video_path
        self.chunk_duration = chunk_duration
        self.current_chunk = None
        self.current_index = 0
    
    def __enter__(self):
        from moviepy.editor import VideoFileClip
        self.video = VideoFileClip(self.video_path)
        self.total_chunks = int(self.video.duration / self.chunk_duration) + 1
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_chunk:
            self.current_chunk.close()
        if hasattr(self, 'video'):
            self.video.close()
    
    def get_next_chunk(self):
        """다음 청크 반환"""
        if self.current_index >= self.total_chunks:
            return None
        
        start_time = self.current_index * self.chunk_duration
        end_time = min((self.current_index + 1) * self.chunk_duration, self.video.duration)
        
        # 이전 청크 정리
        if self.current_chunk:
            self.current_chunk.close()
        
        self.current_chunk = self.video.subclip(start_time, end_time)
        self.current_index += 1
        
        return self.current_chunk, start_time, end_time
    
    def reset(self):
        """청크 인덱스 초기화"""
        self.current_index = 0
        if self.current_chunk:
            self.current_chunk.close()
            self.current_chunk = None


# 전역 최적화 함수
def enable_performance_mode():
    """
    성능 모드 활성화
    - 자동 임시 파일 정리
    - 메모리 최적화
    - 캐싱 활성화
    """
    # 임시 디렉토리 정리 (24시간 이상된 파일)
    for directory in ['temp', 'processed', 'downloads']:
        PerformanceOptimizer.cleanup_old_files(directory, max_age_hours=24)
    
    # Streamlit 캐시 최적화
    PerformanceOptimizer.optimize_streamlit_cache()
    
    # 성능 모드 플래그 설정
    st.session_state['performance_mode'] = True