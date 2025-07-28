from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
from moviepy.config import change_settings
from pathlib import Path
import tempfile
import os
from speaker_detector import SpeakerDetector
import logging
import subprocess

# MoviePy 로깅 비활성화
logging.getLogger('moviepy').setLevel(logging.ERROR)

# FFmpeg 경로 명시적 설정
FFMPEG_BINARY = '/opt/homebrew/bin/ffmpeg'
if os.path.exists(FFMPEG_BINARY):
    change_settings({"FFMPEG_BINARY": FFMPEG_BINARY})

# 고급 화자 감지 옵션
try:
    from improved_speaker_detector import ImprovedSpeakerDetector
    ADVANCED_DETECTOR_AVAILABLE = True
except ImportError:
    ADVANCED_DETECTOR_AVAILABLE = False
    ImprovedSpeakerDetector = None

# 향상된 화자 감지 옵션 (고도화된 특징 추출)
try:
    from enhanced_speaker_detector import EnhancedSpeakerDetector
    ENHANCED_DETECTOR_AVAILABLE = True
except ImportError:
    ENHANCED_DETECTOR_AVAILABLE = False
    EnhancedSpeakerDetector = None

# 실용적인 화자 감지 옵션
try:
    from practical_speaker_detector import PracticalSpeakerDetector
    PRACTICAL_DETECTOR_AVAILABLE = True
except ImportError:
    PRACTICAL_DETECTOR_AVAILABLE = False
    PracticalSpeakerDetector = None

class VideoEditor:
    def __init__(self):
        self.video_clip = None
        self.output_dir = Path("processed")
        self.output_dir.mkdir(exist_ok=True)
        self.temp_counter = 0
        self.speaker_detector = SpeakerDetector()
        self.advanced_detector = ImprovedSpeakerDetector() if ADVANCED_DETECTOR_AVAILABLE else None
        self.enhanced_detector = EnhancedSpeakerDetector() if ENHANCED_DETECTOR_AVAILABLE else None
        self.practical_detector = PracticalSpeakerDetector() if PRACTICAL_DETECTOR_AVAILABLE else None
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
            
            # 비디오 쓰기 설정
            if cut_clip.audio is not None:
                cut_clip.write_videofile(
                    output_path, 
                    codec='libx264', 
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    fps=self.video_clip.fps,
                    preset='medium',
                    threads=4,
                    logger=None
                )
            else:
                cut_clip.write_videofile(
                    output_path, 
                    codec='libx264',
                    fps=self.video_clip.fps,
                    preset='medium',
                    threads=4,
                    logger=None
                )
            
            cut_clip.close()
            return output_path
        except Exception as e:
            print(f"동영상 자르기 실패: {e}")
            import traceback
            traceback.print_exc()
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
    
    def detect_speakers(self, min_duration=2.0, num_speakers=2, use_simple=False, use_advanced=False, use_enhanced=False, use_practical=False):
        """화자 구간 감지"""
        if self.video_path is None:
            return None
        
        # 실용적인 감지기 사용 (균형잡힌 성능)
        if use_practical and self.practical_detector:
            return self.practical_detector.detect_speakers(
                self.video_path,
                min_duration,
                num_speakers=num_speakers
            )
        
        # 향상된 감지기 사용 (최고 성능)
        if use_enhanced and self.enhanced_detector:
            return self.enhanced_detector.detect_speakers(
                self.video_path,
                min_duration,
                num_speakers=num_speakers
            )
        
        # 고급 감지기 사용
        if use_advanced and self.advanced_detector:
            return self.advanced_detector.detect_speakers(
                self.video_path,
                min_duration,
                num_speakers=num_speakers
            )
        
        # 기본 감지기 사용
        return self.speaker_detector.detect_speakers(
            self.video_path, 
            min_duration, 
            num_speakers=num_speakers,
            use_simple=use_simple
        )
    
    def cut_video_ffmpeg(self, input_path, output_path, start_time, end_time):
        """FFmpeg를 직접 사용하여 비디오 자르기"""
        try:
            duration = end_time - start_time
            cmd = [
                FFMPEG_BINARY,
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'fast',
                '-movflags', 'faststart',
                '-y',  # 덮어쓰기
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return True
            else:
                print(f"FFmpeg 오류: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"FFmpeg 실행 실패: {e}")
            return False
    
    def cut_by_speaker(self, speaker_segments):
        """화자별로 동영상 자르기"""
        if self.video_clip is None or not speaker_segments:
            return []
        
        output_files = []
        try:
            for i, segment in enumerate(speaker_segments):
                try:
                    # 시작과 끝 시간 검증
                    start_time = max(0, segment['start'])
                    end_time = min(self.video_clip.duration, segment['end'])
                    
                    if end_time <= start_time:
                        print(f"세그먼트 {i+1} 스킵: 유효하지 않은 시간 범위")
                        continue
                    
                    clip = self.video_clip.subclip(start_time, end_time)
                    output_path = self.get_output_path(f"{segment['speaker']}_segment{i+1}")
                    
                    # 짧은 클립 체크
                    if clip.duration < 0.1:
                        print(f"세그먼트 {i+1} 스킵: 너무 짧음 ({clip.duration}초)")
                        clip.close()
                        continue
                    
                    # 임시 파일 경로 생성
                    temp_video = tempfile.mktemp(suffix='.mp4')
                    
                    try:
                        # 오디오가 있는지 확인
                        if clip.audio is not None:
                            # 오디오 파라미터 설정
                            audio_params = ['-c:a', 'aac', '-b:a', '128k']
                            
                            clip.write_videofile(
                                temp_video,
                                codec='libx264',
                                audio_codec='aac',
                                preset='ultrafast',  # 빠른 인코딩
                                threads=4,
                                ffmpeg_params=['-movflags', 'faststart'],
                                audio_bitrate='128k',
                                logger=None,
                                temp_audiofile=None  # 임시 오디오 파일 사용하지 않음
                            )
                        else:
                            # 오디오가 없는 경우
                            clip.write_videofile(
                                temp_video,
                                codec='libx264',
                                preset='ultrafast',
                                threads=4,
                                ffmpeg_params=['-movflags', 'faststart'],
                                logger=None
                            )
                        
                        # 임시 파일을 최종 위치로 이동
                        import shutil
                        shutil.move(temp_video, output_path)
                        
                        output_files.append({
                            'path': output_path,
                            'speaker': segment['speaker'],
                            'start': segment['start'],
                            'end': segment['end'],
                            'duration': segment['duration'],
                            'text': segment.get('text', '')
                        })
                        
                    finally:
                        # 임시 파일 정리
                        if os.path.exists(temp_video):
                            os.remove(temp_video)
                        clip.close()
                        
                except Exception as e:
                    print(f"세그먼트 {i+1} MoviePy 실패: {e}")
                    
                    # FFmpeg 직접 사용으로 재시도
                    if self.video_path:
                        print(f"FFmpeg로 재시도 중...")
                        output_path = self.get_output_path(f"{segment['speaker']}_segment{i+1}")
                        
                        if self.cut_video_ffmpeg(self.video_path, output_path, start_time, end_time):
                            output_files.append({
                                'path': output_path,
                                'speaker': segment['speaker'],
                                'start': segment['start'],
                                'end': segment['end'],
                                'duration': segment['duration'],
                                'text': segment.get('text', '')
                            })
                            print(f"세그먼트 {i+1} FFmpeg 성공")
                        else:
                            print(f"세그먼트 {i+1} FFmpeg도 실패")
                    
                    # 실패한 세그먼트는 건너뛰고 계속 진행
                    continue
            
            return output_files
        except Exception as e:
            print(f"화자별 자르기 실패: {e}")
            import traceback
            traceback.print_exc()
            return output_files
    
    def cut_single_speaker(self, speaker_name):
        """특정 화자의 구간만 추출하여 하나의 영상으로 만들기"""
        if self.video_clip is None or self.video_path is None:
            return None
        
        try:
            # 출력 디렉토리 확인
            self.output_dir.mkdir(exist_ok=True)
            
            # 세션 상태에서 이미 감지된 segments 사용
            import streamlit as st
            if 'speaker_segments' in st.session_state:
                segments = st.session_state.speaker_segments
            else:
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
                try:
                    clip = self.video_clip.subclip(segment['start'], segment['end'])
                    clips.append(clip)
                except Exception as e:
                    print(f"구간 추출 실패 ({segment['start']}-{segment['end']}): {e}")
                    continue
            
            if not clips:
                return None
            
            # 클립들을 연결
            final_clip = concatenate_videoclips(clips, method="compose")
            output_path = self.get_output_path(f"{speaker_name}_combined")
            
            # 오디오가 있는지 확인
            if final_clip.audio is not None:
                final_clip.write_videofile(
                    output_path, 
                    codec='libx264', 
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    logger=None
                )
            else:
                final_clip.write_videofile(
                    output_path, 
                    codec='libx264',
                    logger=None
                )
            
            # 리소스 정리
            for clip in clips:
                clip.close()
            final_clip.close()
            
            return output_path
            
        except Exception as e:
            print(f"특정 화자 추출 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def __del__(self):
        """리소스 정리"""
        if self.video_clip:
            self.video_clip.close()