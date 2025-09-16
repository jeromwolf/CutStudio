"""
Import Module - 파일 가져오기 및 검증
"""

import streamlit as st
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any, Optional
import os
import yt_dlp
from moviepy.editor import VideoFileClip, AudioFileClip
import json


class ImportModule:
    """
    파일 임포트 모듈
    
    Features:
    - File upload handling
    - YouTube download
    - Media validation
    - Format conversion
    """
    
    def __init__(self):
        """모듈 초기화"""
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # 지원 포맷
        self.video_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        self.audio_formats = ['mp3', 'wav', 'm4a', 'aac', 'flac']
        self.supported_formats = self.video_formats + self.audio_formats
    
    def save_uploaded_file(self, uploaded_file) -> Path:
        """
        업로드된 파일 저장
        
        Args:
            uploaded_file: Streamlit UploadedFile 객체
            
        Returns:
            저장된 파일 경로
        """
        # 고유 파일명 생성
        timestamp = Path(uploaded_file.name).stem
        extension = Path(uploaded_file.name).suffix
        filename = f"{timestamp}_{id(uploaded_file)}{extension}"
        filepath = self.temp_dir / filename
        
        # 파일 저장
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return filepath
    
    def download_youtube(self, url: str) -> Path:
        """
        YouTube 동영상 다운로드
        
        Args:
            url: YouTube URL
            
        Returns:
            다운로드된 파일 경로
        """
        # 다운로드 옵션 설정
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'cookiefile': 'youtube_cookies.txt' if os.path.exists('youtube_cookies.txt') else None
        }
        
        # 다운로드 실행
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            
            # 확장자 보정
            if not Path(filename).exists():
                filename = filename.rsplit('.', 1)[0] + '.mp4'
            
            return Path(filename)
    
    def validate_media(self, filepath: Path) -> Dict[str, Any]:
        """
        미디어 파일 검증 및 정보 추출
        
        Args:
            filepath: 미디어 파일 경로
            
        Returns:
            미디어 정보 딕셔너리
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # 파일 크기 확인
        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        if file_size > 2048:  # 2GB 제한
            raise ValueError(f"File too large: {file_size:.1f}MB (max 2GB)")
        
        # 확장자 확인
        extension = filepath.suffix.lower()[1:]
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported format: {extension}")
        
        # 미디어 타입 결정
        media_type = "video" if extension in self.video_formats else "audio"
        
        # 미디어 정보 추출
        try:
            if media_type == "video":
                clip = VideoFileClip(str(filepath))
                info = {
                    "type": "video",
                    "duration": clip.duration,
                    "fps": clip.fps,
                    "size": (clip.w, clip.h),
                    "has_audio": clip.audio is not None,
                    "filename": filepath.name,
                    "file_size": file_size,
                    "format": extension
                }
                clip.close()
            else:
                clip = AudioFileClip(str(filepath))
                info = {
                    "type": "audio",
                    "duration": clip.duration,
                    "filename": filepath.name,
                    "file_size": file_size,
                    "format": extension
                }
                clip.close()
            
            return info
            
        except Exception as e:
            raise ValueError(f"Invalid media file: {str(e)}")
    
    def convert_format(self, filepath: Path, target_format: str) -> Path:
        """
        파일 포맷 변환
        
        Args:
            filepath: 원본 파일 경로
            target_format: 대상 포맷
            
        Returns:
            변환된 파일 경로
        """
        if target_format not in self.supported_formats:
            raise ValueError(f"Unsupported target format: {target_format}")
        
        # 출력 파일 경로
        output_path = filepath.with_suffix(f".{target_format}")
        
        # 변환 실행
        if filepath.suffix[1:] in self.video_formats:
            clip = VideoFileClip(str(filepath))
            if target_format in self.video_formats:
                clip.write_videofile(str(output_path), codec='libx264')
            else:  # 비디오 → 오디오
                clip.audio.write_audiofile(str(output_path))
            clip.close()
        else:  # 오디오
            clip = AudioFileClip(str(filepath))
            clip.write_audiofile(str(output_path))
            clip.close()
        
        return output_path
    
    def extract_metadata(self, filepath: Path) -> Dict[str, Any]:
        """
        미디어 메타데이터 추출
        
        Args:
            filepath: 미디어 파일 경로
            
        Returns:
            메타데이터 딕셔너리
        """
        metadata = {
            "filename": filepath.name,
            "file_size": filepath.stat().st_size,
            "created": filepath.stat().st_ctime,
            "modified": filepath.stat().st_mtime
        }
        
        # yt-dlp로 추가 메타데이터 추출 시도
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(str(filepath), download=False)
                metadata.update({
                    "title": info.get("title", filepath.stem),
                    "duration": info.get("duration", 0),
                    "format": info.get("format", "unknown")
                })
        except:
            pass
        
        return metadata
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        오래된 임시 파일 정리
        
        Args:
            max_age_hours: 최대 보관 시간
        """
        import time
        current_time = time.time()
        
        for file in self.temp_dir.iterdir():
            if file.is_file():
                file_age = (current_time - file.stat().st_mtime) / 3600
                if file_age > max_age_hours:
                    try:
                        file.unlink()
                    except:
                        pass
    
    def get_youtube_info(self, url: str) -> Dict[str, Any]:
        """
        YouTube 정보 미리보기
        
        Args:
            url: YouTube URL
            
        Returns:
            동영상 정보
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            return {
                "title": info.get("title", "Unknown"),
                "duration": info.get("duration", 0),
                "uploader": info.get("uploader", "Unknown"),
                "view_count": info.get("view_count", 0),
                "thumbnail": info.get("thumbnail", ""),
                "description": info.get("description", "")[:200]
            }