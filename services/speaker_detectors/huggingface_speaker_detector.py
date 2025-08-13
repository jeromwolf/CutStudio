import os
import tempfile
import time
from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio import Audio
import warnings
warnings.filterwarnings('ignore')

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

class HuggingFaceSpeakerDetector:
    def __init__(self, hf_token=None):
        """
        허깅페이스 기반 화자 분리 감지기
        
        Args:
            hf_token: 허깅페이스 액세스 토큰 (pyannote 모델 사용을 위해 필요)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        self.audio_loader = Audio(sample_rate=16000, mono=True)
        
    def initialize_pipeline(self):
        """Pyannote 화자 분리 파이프라인 초기화"""
        if self.pipeline is not None:
            return True
            
        try:
            if not self.hf_token:
                print("허깅페이스 토큰이 필요합니다. https://huggingface.co/settings/tokens 에서 토큰을 생성하세요.")
                print("토큰을 생성한 후, 환경변수 HUGGINGFACE_TOKEN으로 설정하거나 초기화 시 전달하세요.")
                return False
                
            print(f"토큰 확인됨 (길이: {len(self.hf_token)})")
            
            # 최신 pyannote 3.1 파이프라인 로드
            print("Pyannote 파이프라인 로드 중...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
                cache_dir="./pretrained_models"
            )
            
            # 파이프라인이 제대로 로드되었는지 확인
            if self.pipeline is None:
                raise ValueError("파이프라인이 None입니다. 모델 로드에 실패했습니다.")
            
            # GPU 사용 가능 시 GPU로 이동
            if torch.cuda.is_available():
                self.pipeline.to(self.device)
            
            print(f"화자 분리 파이프라인 로드 완료 (Device: {self.device})")
            return True
            
        except Exception as e:
            import traceback
            print(f"파이프라인 초기화 실패: {e}")
            print("상세 에러:")
            traceback.print_exc()
            print("\n토큰이 유효한지, pyannote 모델에 대한 액세스 권한이 있는지 확인하세요.")
            print("모델 페이지에서 사용 조건에 동의했는지 확인: https://huggingface.co/pyannote/speaker-diarization-3.1")
            return False
    
    def extract_audio(self, video_path):
        """미디어 파일에서 오디오 추출"""
        try:
            # 파일 확장자 확인
            ext = video_path.lower().split('.')[-1]
            audio_path = tempfile.mktemp(suffix=".wav")
            
            if ext in ['m4a', 'mp3', 'wav', 'aac', 'flac', 'ogg', 'wma']:
                # 오디오 파일인 경우
                audio = AudioFileClip(video_path)
                audio.write_audiofile(audio_path, verbose=False, logger=None)
                audio.close()
            else:
                # 비디오 파일인 경우
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                video.close()
                
            return audio_path
        except Exception as e:
            print(f"오디오 추출 실패: {e}")
            return None
    
    def detect_speakers(self, video_path, min_duration=2.0, num_speakers=None, 
                       min_speakers=None, max_speakers=None):
        """
        허깅페이스 모델을 사용한 화자 감지
        
        Args:
            video_path: 비디오 파일 경로
            min_duration: 최소 세그먼트 길이 (초)
            num_speakers: 정확한 화자 수 (옵션)
            min_speakers: 최소 화자 수 (옵션)
            max_speakers: 최대 화자 수 (옵션)
        
        Returns:
            화자 세그먼트 리스트
        """
        # 파이프라인 초기화
        if not self.initialize_pipeline():
            return None
            
        # 오디오 추출
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return None
        
        try:
            # 화자 분리 실행
            print("화자 분리 중... (처음 실행 시 시간이 걸릴 수 있습니다)")
            
            # 오디오 길이 확인
            from moviepy.editor import AudioFileClip
            with AudioFileClip(audio_path) as audio:
                duration = audio.duration
                print(f"오디오 길이: {duration:.1f}초")
                
            estimated_time = duration * 0.3  # 대략 오디오 길이의 30% 시간 소요
            print(f"예상 처리 시간: {estimated_time:.0f}초 ~ {estimated_time*2:.0f}초")
            
            # 파이프라인 매개변수 설정
            pipeline_params = {}
            if num_speakers is not None:
                pipeline_params["num_speakers"] = num_speakers
            elif min_speakers is not None or max_speakers is not None:
                pipeline_params["min_speakers"] = min_speakers
                pipeline_params["max_speakers"] = max_speakers
            
            print("\n처리 진행 중...")
            start_time = time.time()
            
            # 화자 분리 실행
            diarization = self.pipeline(audio_path, **pipeline_params)
            
            elapsed_time = time.time() - start_time
            print(f"\n처리 완료! 소요 시간: {elapsed_time:.1f}초")
            
            # 결과를 세그먼트 리스트로 변환
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                
                # 최소 길이 필터링
                if duration >= min_duration:
                    segments.append({
                        'start': turn.start,
                        'end': turn.end,
                        'speaker': speaker,
                        'duration': duration,
                        'confidence': 0.95  # pyannote는 일반적으로 높은 정확도를 가짐
                    })
            
            # 화자 라벨 정규화 (SPEAKER_0, SPEAKER_1 등으로 변경)
            speaker_mapping = {}
            for i, seg in enumerate(segments):
                speaker = seg['speaker']
                if speaker not in speaker_mapping:
                    speaker_mapping[speaker] = f'SPEAKER_{len(speaker_mapping)}'
                seg['speaker'] = speaker_mapping[speaker]
            
            # 시간순 정렬
            segments = sorted(segments, key=lambda x: x['start'])
            
            # 인접한 같은 화자 세그먼트 병합
            segments = self.merge_adjacent_segments(segments)
            
            print(f"화자 분리 완료: {len(speaker_mapping)}명의 화자, {len(segments)}개 세그먼트")
            
            # 임시 파일 삭제
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return segments
            
        except Exception as e:
            import traceback
            print(f"화자 감지 실패: {e}")
            print("상세 에러:")
            traceback.print_exc()
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return None
    
    def merge_adjacent_segments(self, segments, max_gap=0.5):
        """인접한 같은 화자 구간 병합"""
        if not segments:
            return segments
        
        merged = []
        current = segments[0].copy()
        
        for segment in segments[1:]:
            if (segment['speaker'] == current['speaker'] and 
                segment['start'] - current['end'] <= max_gap):
                # 병합
                current['end'] = segment['end']
                current['duration'] = current['end'] - current['start']
                # 신뢰도는 평균값 사용
                if 'confidence' in current and 'confidence' in segment:
                    current['confidence'] = (current['confidence'] + segment['confidence']) / 2
            else:
                merged.append(current)
                current = segment.copy()
        
        merged.append(current)
        return merged
    
    def get_speaker_statistics(self, segments):
        """화자별 통계 정보 반환"""
        if not segments:
            return {}
        
        stats = {}
        for seg in segments:
            speaker = seg['speaker']
            if speaker not in stats:
                stats[speaker] = {
                    'total_duration': 0,
                    'segment_count': 0,
                    'segments': []
                }
            
            stats[speaker]['total_duration'] += seg['duration']
            stats[speaker]['segment_count'] += 1
            stats[speaker]['segments'].append({
                'start': seg['start'],
                'end': seg['end'],
                'duration': seg['duration']
            })
        
        # 각 화자의 평균 세그먼트 길이 계산
        for speaker in stats:
            stats[speaker]['avg_segment_duration'] = (
                stats[speaker]['total_duration'] / stats[speaker]['segment_count']
            )
        
        return stats
    
    def export_to_rttm(self, segments, output_path, recording_id="RECORDING"):
        """
        세그먼트를 RTTM (Rich Transcription Time Marked) 형식으로 내보내기
        화자 분리 평가에 표준으로 사용되는 형식
        """
        with open(output_path, 'w') as f:
            for seg in segments:
                # RTTM 형식: SPEAKER file 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
                line = f"SPEAKER {recording_id} 1 {seg['start']:.3f} {seg['duration']:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>\n"
                f.write(line)
        
        print(f"RTTM 파일 저장됨: {output_path}")