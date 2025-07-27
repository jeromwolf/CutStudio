from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
from pathlib import Path
import tempfile
import os
from speaker_detector import SpeakerDetector

class VideoEditor:
    def __init__(self):
        self.video_clip = None
        self.output_dir = Path("processed")
        self.output_dir.mkdir(exist_ok=True)
        self.temp_counter = 0
        self.speaker_detector = SpeakerDetector()
        self.video_path = None
    
    def load_video(self, video_path):
        """동영상 파일 로드"""
        try:
            self.video_clip = VideoFileClip(video_path)
            self.video_path = video_path
            return True
        except Exception as e:
            print(f"동영상 로드 실패: {e}")
            return False
    
    def get_output_path(self, prefix="output"):
        """출력 파일 경로 생성"""
        self.temp_counter += 1
        return str(self.output_dir / f"{prefix}_{self.temp_counter}.mp4")
    
    def cut_video(self, start_time, end_time):
        """동영상 자르기"""
        if self.video_clip is None:
            return None
        
        try:
            cut_clip = self.video_clip.subclip(start_time, end_time)
            output_path = self.get_output_path("cut")
            cut_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            cut_clip.close()
            return output_path
        except Exception as e:
            print(f"동영상 자르기 실패: {e}")
            return None
    
    def trim_video(self, trim_start, trim_end):
        """동영상 트림 (앞뒤 제거)"""
        if self.video_clip is None:
            return None
        
        try:
            duration = self.video_clip.duration
            trimmed_clip = self.video_clip.subclip(trim_start, duration - trim_end)
            output_path = self.get_output_path("trim")
            trimmed_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            trimmed_clip.close()
            return output_path
        except Exception as e:
            print(f"동영상 트림 실패: {e}")
            return None
    
    def apply_grayscale(self):
        """흑백 효과 적용"""
        if self.video_clip is None:
            return None
        
        try:
            grayscale_clip = self.video_clip.fx(vfx.blackwhite)
            output_path = self.get_output_path("grayscale")
            grayscale_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            grayscale_clip.close()
            return output_path
        except Exception as e:
            print(f"흑백 효과 적용 실패: {e}")
            return None
    
    def apply_fade_in(self, duration=1.0):
        """페이드 인 효과 적용"""
        if self.video_clip is None:
            return None
        
        try:
            fade_in_clip = self.video_clip.fx(vfx.fadein, duration)
            output_path = self.get_output_path("fadein")
            fade_in_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            fade_in_clip.close()
            return output_path
        except Exception as e:
            print(f"페이드 인 효과 적용 실패: {e}")
            return None
    
    def apply_fade_out(self, duration=1.0):
        """페이드 아웃 효과 적용"""
        if self.video_clip is None:
            return None
        
        try:
            fade_out_clip = self.video_clip.fx(vfx.fadeout, duration)
            output_path = self.get_output_path("fadeout")
            fade_out_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            fade_out_clip.close()
            return output_path
        except Exception as e:
            print(f"페이드 아웃 효과 적용 실패: {e}")
            return None
    
    def change_speed(self, speed_factor):
        """재생 속도 변경"""
        if self.video_clip is None:
            return None
        
        try:
            speed_clip = self.video_clip.fx(vfx.speedx, speed_factor)
            output_path = self.get_output_path("speed")
            speed_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            speed_clip.close()
            return output_path
        except Exception as e:
            print(f"속도 변경 실패: {e}")
            return None
    
    def detect_speakers(self, min_duration=2.0):
        """화자 구간 감지"""
        if self.video_path is None:
            return None
        
        return self.speaker_detector.detect_speakers(self.video_path, min_duration)
    
    def cut_by_speaker(self, speaker_segments):
        """화자별로 동영상 자르기"""
        if self.video_clip is None or not speaker_segments:
            return []
        
        output_files = []
        try:
            for i, segment in enumerate(speaker_segments):
                clip = self.video_clip.subclip(segment['start'], segment['end'])
                output_path = self.get_output_path(f"{segment['speaker']}_segment{i+1}")
                clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                clip.close()
                
                output_files.append({
                    'path': output_path,
                    'speaker': segment['speaker'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment['duration'],
                    'text': segment.get('text', '')
                })
            
            return output_files
        except Exception as e:
            print(f"화자별 자르기 실패: {e}")
            return output_files
    
    def cut_single_speaker(self, speaker_name):
        """특정 화자의 구간만 추출하여 하나의 영상으로 만들기"""
        if self.video_clip is None or self.video_path is None:
            return None
        
        try:
            # 화자 구간 감지
            segments = self.speaker_detector.detect_speakers(self.video_path)
            if not segments:
                return None
            
            # 특정 화자의 구간만 필터링
            speaker_segments = [s for s in segments if s['speaker'] == speaker_name]
            if not speaker_segments:
                return None
            
            # 각 구간의 클립 생성
            clips = []
            for segment in speaker_segments:
                clip = self.video_clip.subclip(segment['start'], segment['end'])
                clips.append(clip)
            
            # 클립들을 연결
            final_clip = concatenate_videoclips(clips)
            output_path = self.get_output_path(f"{speaker_name}_combined")
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            
            # 리소스 정리
            for clip in clips:
                clip.close()
            final_clip.close()
            
            return output_path
            
        except Exception as e:
            print(f"특정 화자 추출 실패: {e}")
            return None
    
    def __del__(self):
        """리소스 정리"""
        if self.video_clip:
            self.video_clip.close()