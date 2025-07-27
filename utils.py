from moviepy.editor import VideoFileClip
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