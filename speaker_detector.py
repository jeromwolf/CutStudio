import os
import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import speech_recognition as sr
import json

# pyannote는 선택적으로 import
try:
    from pyannote.audio import Pipeline
    import torch
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    Pipeline = None
    torch = None

class SpeakerDetector:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.pipeline = None
        if PYANNOTE_AVAILABLE and torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        
    def initialize_pipeline(self, hf_token=None):
        """화자 분리 파이프라인 초기화"""
        if not PYANNOTE_AVAILABLE:
            print("pyannote가 설치되지 않았습니다. 간단한 VAD를 사용합니다.")
            return False
            
        try:
            if hf_token and Pipeline:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
            else:
                # 간단한 대안으로 음성 구간 검출만 수행
                self.pipeline = None
            
            if self.pipeline and self.device:
                self.pipeline.to(self.device)
            return True
        except Exception as e:
            print(f"파이프라인 초기화 실패: {e}")
            return False
    
    def extract_audio(self, video_path):
        """비디오에서 오디오 추출"""
        try:
            video = VideoFileClip(video_path)
            audio_path = tempfile.mktemp(suffix=".wav")
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            return audio_path
        except Exception as e:
            print(f"오디오 추출 실패: {e}")
            return None
    
    def detect_speakers(self, video_path, min_duration=2.0):
        """화자 구간 감지"""
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return None
        
        try:
            if self.pipeline:
                # Pyannote를 사용한 화자 분리
                diarization = self.pipeline(audio_path)
                
                segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.duration >= min_duration:
                        segments.append({
                            'start': turn.start,
                            'end': turn.end,
                            'speaker': speaker,
                            'duration': turn.duration
                        })
            else:
                # 간단한 음성 활동 감지 (VAD) 사용
                segments = self.simple_vad(audio_path, min_duration)
            
            # 임시 파일 삭제
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return segments
            
        except Exception as e:
            print(f"화자 감지 실패: {e}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return None
    
    def simple_vad(self, audio_path, min_duration=2.0):
        """간단한 음성 활동 감지"""
        try:
            audio = AudioSegment.from_wav(audio_path)
            
            # 음성 구간 감지 (간단한 에너지 기반)
            segments = []
            chunk_size = 1000  # 1초
            energy_threshold = audio.dBFS + 10  # 평균보다 10dB 높은 구간
            
            current_segment = None
            speaker_id = 0
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                
                if chunk.dBFS > energy_threshold:
                    if current_segment is None:
                        current_segment = {
                            'start': i / 1000.0,
                            'speaker': f'SPEAKER_{speaker_id}'
                        }
                else:
                    if current_segment is not None:
                        current_segment['end'] = i / 1000.0
                        current_segment['duration'] = current_segment['end'] - current_segment['start']
                        
                        if current_segment['duration'] >= min_duration:
                            segments.append(current_segment)
                            speaker_id += 1
                        
                        current_segment = None
            
            # 마지막 구간 처리
            if current_segment is not None:
                current_segment['end'] = len(audio) / 1000.0
                current_segment['duration'] = current_segment['end'] - current_segment['start']
                if current_segment['duration'] >= min_duration:
                    segments.append(current_segment)
            
            return segments
            
        except Exception as e:
            print(f"VAD 처리 실패: {e}")
            return []
    
    def transcribe_segments(self, video_path, segments):
        """각 화자 구간의 음성을 텍스트로 변환"""
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return segments
        
        try:
            audio = AudioSegment.from_wav(audio_path)
            
            for segment in segments:
                start_ms = int(segment['start'] * 1000)
                end_ms = int(segment['end'] * 1000)
                
                # 구간 오디오 추출
                segment_audio = audio[start_ms:end_ms]
                temp_path = tempfile.mktemp(suffix=".wav")
                segment_audio.export(temp_path, format="wav")
                
                # 음성 인식
                try:
                    with sr.AudioFile(temp_path) as source:
                        audio_data = self.recognizer.record(source)
                        text = self.recognizer.recognize_google(audio_data, language='ko-KR')
                        segment['text'] = text
                except:
                    segment['text'] = "[인식 불가]"
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return segments
            
        except Exception as e:
            print(f"음성 인식 실패: {e}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return segments
    
    def merge_close_segments(self, segments, max_gap=1.0):
        """가까운 화자 구간 병합"""
        if not segments:
            return segments
        
        merged = []
        current = segments[0].copy()
        
        for segment in segments[1:]:
            if (segment['speaker'] == current['speaker'] and 
                segment['start'] - current['end'] <= max_gap):
                # 같은 화자의 가까운 구간 병합
                current['end'] = segment['end']
                current['duration'] = current['end'] - current['start']
                if 'text' in current and 'text' in segment:
                    current['text'] = current['text'] + ' ' + segment['text']
            else:
                merged.append(current)
                current = segment.copy()
        
        merged.append(current)
        return merged