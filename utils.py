from moviepy.editor import VideoFileClip, AudioFileClip
import cv2

def get_video_info(video_path):
    """동영상 정보 추출"""
    try:
        clip = VideoFileClip(video_path)
        info = {
            'duration': clip.duration,
            'fps': clip.fps,
            'width': clip.w,
            'height': clip.h,
            'size': (clip.w, clip.h)
        }
        clip.close()
        return info
    except Exception as e:
        print(f"동영상 정보 추출 실패: {e}")
        return {
            'duration': 0,
            'fps': 0,
            'width': 0,
            'height': 0,
            'size': (0, 0)
        }

def format_time(seconds):
    """초를 시:분:초 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def get_video_thumbnail(video_path, time_position=0):
    """동영상 썸네일 생성"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(fps * time_position)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        return None
    except Exception as e:
        print(f"썸네일 생성 실패: {e}")
        return None


def is_audio_file(file_path):
    """파일이 오디오 파일인지 확인"""
    if isinstance(file_path, str):
        ext = file_path.lower().split('.')[-1]
    else:
        # UploadedFile 객체의 경우
        ext = file_path.name.lower().split('.')[-1]
    return ext in ['m4a', 'mp3', 'wav', 'aac', 'flac', 'ogg', 'wma']


def is_video_file(file_path):
    """파일이 비디오 파일인지 확인"""
    if isinstance(file_path, str):
        ext = file_path.lower().split('.')[-1]
    else:
        # UploadedFile 객체의 경우
        ext = file_path.name.lower().split('.')[-1]
    return ext in ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm']


def get_media_info(media_path):
    """미디어 파일 정보 추출 (비디오/오디오 모두 지원)"""
    try:
        # 오디오 파일인 경우
        if is_audio_file(media_path):
            clip = AudioFileClip(media_path)
            info = {
                'duration': clip.duration,
                'fps': clip.fps if hasattr(clip, 'fps') else None,
                'width': None,
                'height': None,
                'size': (None, None),
                'type': 'audio',
                'audio_channels': clip.nchannels if hasattr(clip, 'nchannels') else None,
                'audio_fps': clip.fps if hasattr(clip, 'fps') else None
            }
            clip.close()
            return info
        # 비디오 파일인 경우
        else:
            clip = VideoFileClip(media_path)
            info = {
                'duration': clip.duration,
                'fps': clip.fps,
                'width': clip.w,
                'height': clip.h,
                'size': (clip.w, clip.h),
                'type': 'video',
                'audio_channels': clip.audio.nchannels if clip.audio and hasattr(clip.audio, 'nchannels') else None,
                'audio_fps': clip.audio.fps if clip.audio and hasattr(clip.audio, 'fps') else None
            }
            clip.close()
            return info
    except Exception as e:
        print(f"미디어 정보 추출 실패: {e}")
        return {
            'duration': 0,
            'fps': 0,
            'width': 0,
            'height': 0,
            'size': (0, 0),
            'type': 'unknown'
        }