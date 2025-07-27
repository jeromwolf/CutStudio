import yt_dlp
import os
from pathlib import Path
import re

class YouTubeDownloader:
    def __init__(self):
        self.download_dir = Path("downloads")
        self.download_dir.mkdir(exist_ok=True)
        
    def sanitize_filename(self, filename):
        """파일명에서 특수문자 제거"""
        # Windows와 Unix 시스템 모두에서 안전한 파일명으로 변환
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = filename.strip()
        return filename[:200]  # 파일명 길이 제한
    
    def get_video_info(self, url):
        """YouTube 동영상 정보 가져오기"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # 사용 가능한 형식 가져오기
                formats = []
                if 'formats' in info:
                    for f in info['formats']:
                        if f.get('vcodec') != 'none' and f.get('acodec') != 'none':
                            format_info = {
                                'format_id': f['format_id'],
                                'ext': f['ext'],
                                'resolution': f.get('resolution', 'N/A'),
                                'filesize': f.get('filesize', 0),
                                'quality': f.get('quality', 0)
                            }
                            
                            # 해상도로 정렬하기 위한 높이 값 추가
                            if 'height' in f:
                                format_info['height'] = f['height']
                            else:
                                format_info['height'] = 0
                                
                            formats.append(format_info)
                
                # 해상도별로 정렬 (높은 것부터)
                formats.sort(key=lambda x: x['height'], reverse=True)
                
                # 중복 제거 (같은 해상도는 하나만)
                seen_resolutions = set()
                unique_formats = []
                for f in formats:
                    if f['resolution'] not in seen_resolutions:
                        seen_resolutions.add(f['resolution'])
                        unique_formats.append(f)
                
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'description': info.get('description', ''),
                    'thumbnail': info.get('thumbnail', ''),
                    'formats': unique_formats[:5]  # 상위 5개 해상도만 표시
                }
                
        except Exception as e:
            print(f"동영상 정보 가져오기 실패: {e}")
            return None
    
    def download_video(self, url, format_id=None, progress_callback=None):
        """YouTube 동영상 다운로드"""
        try:
            # 진행률 표시를 위한 훅
            def progress_hook(d):
                if d['status'] == 'downloading' and progress_callback:
                    total = d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0)
                    downloaded = d.get('downloaded_bytes', 0)
                    if total > 0:
                        percent = (downloaded / total) * 100
                        progress_callback(percent, downloaded, total)
                elif d['status'] == 'finished':
                    if progress_callback:
                        progress_callback(100, 0, 0)
            
            # 다운로드 옵션 설정
            ydl_opts = {
                'outtmpl': str(self.download_dir / '%(title)s.%(ext)s'),
                'progress_hooks': [progress_hook],
                'quiet': True,
                'no_warnings': True,
            }
            
            # 특정 형식이 지정된 경우
            if format_id:
                ydl_opts['format'] = format_id
            else:
                # 최고 품질 비디오 + 오디오
                ydl_opts['format'] = 'best[ext=mp4]/best'
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # 다운로드된 파일 경로 찾기
                filename = ydl.prepare_filename(info)
                # 실제 확장자로 변경
                actual_filename = filename.rsplit('.', 1)[0] + '.' + info['ext']
                
                if os.path.exists(actual_filename):
                    return actual_filename
                elif os.path.exists(filename):
                    return filename
                else:
                    # 다운로드 디렉토리에서 파일 찾기
                    title = self.sanitize_filename(info.get('title', 'video'))
                    for file in self.download_dir.glob(f"{title}.*"):
                        if file.is_file():
                            return str(file)
                    
                    return None
                    
        except Exception as e:
            print(f"동영상 다운로드 실패: {e}")
            return None
    
    def download_audio_only(self, url, progress_callback=None):
        """오디오만 다운로드 (MP3)"""
        try:
            def progress_hook(d):
                if d['status'] == 'downloading' and progress_callback:
                    total = d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0)
                    downloaded = d.get('downloaded_bytes', 0)
                    if total > 0:
                        percent = (downloaded / total) * 100
                        progress_callback(percent, downloaded, total)
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(self.download_dir / '%(title)s.%(ext)s'),
                'progress_hooks': [progress_hook],
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # MP3 파일 경로 찾기
                title = self.sanitize_filename(info.get('title', 'audio'))
                mp3_file = self.download_dir / f"{title}.mp3"
                
                if mp3_file.exists():
                    return str(mp3_file)
                    
                # 다른 가능한 파일명 찾기
                for file in self.download_dir.glob(f"{title}.*"):
                    if file.suffix.lower() in ['.mp3', '.m4a', '.webm']:
                        return str(file)
                        
                return None
                
        except Exception as e:
            print(f"오디오 다운로드 실패: {e}")
            return None
    
    def get_playlist_info(self, url):
        """재생목록 정보 가져오기"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if 'entries' in info:
                    return {
                        'title': info.get('title', 'Unknown Playlist'),
                        'uploader': info.get('uploader', 'Unknown'),
                        'video_count': len(info['entries']),
                        'videos': [
                            {
                                'title': entry.get('title', 'Unknown'),
                                'url': f"https://youtube.com/watch?v={entry['id']}",
                                'duration': entry.get('duration', 0)
                            }
                            for entry in info['entries'][:10]  # 처음 10개만
                        ]
                    }
                else:
                    return None
                    
        except Exception as e:
            print(f"재생목록 정보 가져오기 실패: {e}")
            return None