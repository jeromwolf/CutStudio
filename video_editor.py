from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx
import cv2
import numpy as np
from PIL import Image
import base64
import io
from moviepy.config import change_settings
from pathlib import Path
import tempfile
import os
from services.speaker_detectors.speaker_detector import SpeakerDetector
import logging
import subprocess
from utils import is_audio_file

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

# 허깅페이스 기반 화자 감지 옵션
try:
    from huggingface_speaker_detector import HuggingFaceSpeakerDetector
    HUGGINGFACE_DETECTOR_AVAILABLE = True
except ImportError:
    HUGGINGFACE_DETECTOR_AVAILABLE = False
    HuggingFaceSpeakerDetector = None

class VideoEditor:
    def __init__(self):
        self.video_clip = None
        self.audio_clip = None  # 오디오 전용 클립
        self.media_type = None  # 'video' 또는 'audio'
        self.output_dir = Path("processed")
        self.output_dir.mkdir(exist_ok=True)
        self.temp_counter = 0
        self.speaker_detector = SpeakerDetector()
        self.advanced_detector = ImprovedSpeakerDetector() if ADVANCED_DETECTOR_AVAILABLE else None
        self.enhanced_detector = EnhancedSpeakerDetector() if ENHANCED_DETECTOR_AVAILABLE else None
        self.practical_detector = PracticalSpeakerDetector() if PRACTICAL_DETECTOR_AVAILABLE else None
        self.huggingface_detector = HuggingFaceSpeakerDetector() if HUGGINGFACE_DETECTOR_AVAILABLE else None
        self.video_path = None
    
    def load_video(self, video_path):
        """미디어 파일 로드 (비디오/오디오 모두 지원)"""
        try:
            if is_audio_file(video_path):
                # 오디오 파일 로드
                self.audio_clip = AudioFileClip(video_path)
                self.video_clip = None
                self.media_type = 'audio'
            else:
                # 비디오 파일 로드
                self.video_clip = VideoFileClip(video_path)
                self.audio_clip = None
                self.media_type = 'video'
            
            self.video_path = video_path
            return True
        except Exception as e:
            print(f"미디어 로드 실패: {e}")
            return False
    
    def get_output_path(self, prefix="output"):
        """출력 파일 경로 생성"""
        self.temp_counter += 1
        ext = 'm4a' if self.media_type == 'audio' else 'mp4'
        return str(self.output_dir / f"{prefix}_{self.temp_counter}.{ext}")
    
    def cut_video(self, start_time, end_time):
        """미디어 자르기 (비디오/오디오 모두 지원)"""
        if self.video_clip is None and self.audio_clip is None:
            return None
        
        try:
            if self.media_type == 'audio':
                # 오디오 파일 자르기
                cut_clip = self.audio_clip.subclip(start_time, end_time)
                output_path = self.get_output_path("cut")
                
                cut_clip.write_audiofile(
                    output_path,
                    codec='aac',
                    logger=None
                )
            else:
                # 비디오 파일 자르기
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
            print(f"미디어 자르기 실패: {e}")
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
    
    def detect_speakers(self, min_duration=2.0, num_speakers=2, use_simple=False, use_advanced=False, use_enhanced=False, use_practical=False, use_huggingface=False, progress_callback=None):
        """화자 구간 감지"""
        if self.video_path is None:
            return None
        
        # 진행 상황 콜백 호출
        if progress_callback:
            progress_callback("초기화 중...", 5)
        
        # 허깅페이스 감지기 사용 (최신 AI 모델)
        if use_huggingface and self.huggingface_detector:
            if progress_callback:
                progress_callback("허깅페이스 AI 모델 실행 중...", 20)
            return self.huggingface_detector.detect_speakers(
                self.video_path,
                min_duration,
                num_speakers=num_speakers
            )
        
        # 실용적인 감지기 사용 (균형잡힌 성능)
        if use_practical and self.practical_detector:
            if progress_callback:
                progress_callback("실용적 감지기 실행 중...", 20)
            return self.practical_detector.detect_speakers(
                self.video_path,
                min_duration,
                num_speakers=num_speakers
            )
        
        # 향상된 감지기 사용 (최고 성능)
        if use_enhanced and self.enhanced_detector:
            if progress_callback:
                progress_callback("향상된 감지기 실행 중...", 20)
            return self.enhanced_detector.detect_speakers(
                self.video_path,
                min_duration,
                num_speakers=num_speakers
            )
        
        # 고급 감지기 사용
        if use_advanced and self.advanced_detector:
            if progress_callback:
                progress_callback("고급 감지기 실행 중...", 20)
            return self.advanced_detector.detect_speakers(
                self.video_path,
                min_duration,
                num_speakers=num_speakers
            )
        
        # 기본 감지기 사용
        if progress_callback:
            progress_callback("기본 감지기 실행 중...", 20)
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
            video_duration = self.video_clip.duration
            for segment in speaker_segments:
                try:
                    # 시간이 동영상 길이를 초과하지 않도록 보정
                    start_time = min(segment['start'], video_duration)
                    end_time = min(segment['end'], video_duration)
                    
                    # 시작 시간이 끝 시간보다 크거나 같으면 스킵
                    if start_time >= end_time:
                        print(f"잘못된 시간 구간 스킵: {segment['start']}-{segment['end']}")
                        continue
                    
                    clip = self.video_clip.subclip(start_time, end_time)
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
    
    def generate_speaker_thumbnails(self, speaker_segments, thumbnail_size=(150, 100)):
        """화자별 썸네일 생성"""
        if not speaker_segments:
            return {}
        
        # 오디오 파일인 경우 빈 딕셔너리 반환
        if self.video_clip is None:
            return {}
        
        thumbnails = {}
        
        try:
            for segment in speaker_segments:
                speaker_id = segment['speaker']
                
                # 이미 해당 화자의 썸네일이 있으면 건너뛰기
                if speaker_id in thumbnails:
                    continue
                
                # 세그먼트 중간 지점에서 프레임 추출
                mid_time = (segment['start'] + segment['end']) / 2
                
                try:
                    # MoviePy로 프레임 추출
                    frame = self.video_clip.get_frame(mid_time)
                    
                    # numpy 배열을 PIL Image로 변환
                    pil_image = Image.fromarray(frame)
                    
                    # 썸네일 크기로 리사이즈
                    pil_image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                    
                    # Base64로 인코딩하여 웹에서 사용할 수 있도록 변환
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='JPEG', quality=85)
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    
                    thumbnails[speaker_id] = {
                        'image_base64': img_str,
                        'timestamp': mid_time,
                        'speaker': speaker_id,
                        'width': pil_image.width,
                        'height': pil_image.height
                    }
                    
                except Exception as e:
                    print(f"화자 {speaker_id} 썸네일 생성 실패: {e}")
                    # 기본 썸네일 생성 (검은 이미지)
                    default_img = Image.new('RGB', thumbnail_size, color='black')
                    buffer = io.BytesIO()
                    default_img.save(buffer, format='JPEG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    
                    thumbnails[speaker_id] = {
                        'image_base64': img_str,
                        'timestamp': mid_time,
                        'speaker': speaker_id,
                        'width': thumbnail_size[0],
                        'height': thumbnail_size[1],
                        'is_default': True
                    }
            
            return thumbnails
            
        except Exception as e:
            print(f"썸네일 생성 전체 실패: {e}")
            return {}
    
    def generate_speaker_summary(self, speaker_segments):
        """화자별 내용 요약 생성"""
        if not speaker_segments:
            return {}
        
        summary = {}
        
        try:
            # 화자별로 그룹화
            speaker_groups = {}
            for segment in speaker_segments:
                speaker_id = segment['speaker']
                if speaker_id not in speaker_groups:
                    speaker_groups[speaker_id] = []
                speaker_groups[speaker_id].append(segment)
            
            # 각 화자별 요약 정보 생성
            for speaker_id, segments in speaker_groups.items():
                total_duration = sum(seg['duration'] for seg in segments)
                segment_count = len(segments)
                
                # 발화 시간대 정보
                time_ranges = [(seg['start'], seg['end']) for seg in segments]
                first_appearance = min(seg['start'] for seg in segments)
                last_appearance = max(seg['end'] for seg in segments)
                
                # 평균 발화 길이
                avg_duration = total_duration / segment_count if segment_count > 0 else 0
                
                # 텍스트 내용 합치기 (음성 인식 결과가 있는 경우)
                texts = [seg.get('text', '') for seg in segments if seg.get('text')]
                combined_text = ' '.join(texts) if texts else "음성 인식 결과 없음"
                
                summary[speaker_id] = {
                    'speaker': speaker_id,
                    'total_duration': round(total_duration, 2),
                    'segment_count': segment_count,
                    'avg_duration': round(avg_duration, 2),
                    'first_appearance': round(first_appearance, 2),
                    'last_appearance': round(last_appearance, 2),
                    'time_ranges': time_ranges,
                    'text_content': combined_text,
                    'participation_rate': 0  # 전체 비디오 대비 비율, 나중에 계산
                }
            
            # 전체 미디어 길이 대비 참여율 계산
            total_duration = None
            if self.video_clip:
                total_duration = self.video_clip.duration
            elif self.audio_clip:
                total_duration = self.audio_clip.duration
            
            if total_duration:
                for speaker_id in summary:
                    participation_rate = (summary[speaker_id]['total_duration'] / total_duration) * 100
                    summary[speaker_id]['participation_rate'] = round(participation_rate, 1)
            
            return summary
            
        except Exception as e:
            print(f"화자 요약 생성 실패: {e}")
            return {}
    
    def generate_speaker_profile(self, speaker_segments):
        """화자별 프로필 정보 생성 (썸네일 + 요약)"""
        thumbnails = self.generate_speaker_thumbnails(speaker_segments)
        summaries = self.generate_speaker_summary(speaker_segments)
        
        profiles = {}
        
        # 썸네일과 요약 정보 결합
        all_speakers = set(list(thumbnails.keys()) + list(summaries.keys()))
        
        for speaker_id in all_speakers:
            profiles[speaker_id] = {
                'speaker': speaker_id,
                'thumbnail': thumbnails.get(speaker_id, {}),
                'summary': summaries.get(speaker_id, {}),
                'has_thumbnail': speaker_id in thumbnails,
                'has_summary': speaker_id in summaries
            }
        
        return profiles

    def __del__(self):
        """리소스 정리"""
        if self.video_clip:
            self.video_clip.close()