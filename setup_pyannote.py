"""
Pyannote.audio 설정 및 사용 가이드

1. Hugging Face 토큰 받기:
   - https://huggingface.co/settings/tokens 에서 토큰 생성
   - pyannote/speaker-diarization-3.1 모델 사용 동의 필요

2. 환경 변수 설정:
   export HF_TOKEN="your_token_here"

3. 필요한 패키지 설치:
   pip install pyannote.audio torch torchaudio

4. 사용 예시:
"""

import os
from pyannote.audio import Pipeline
import torch

class PyannoteDetector:
    def __init__(self, hf_token=None):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.pipeline = None
        
    def initialize(self):
        """Pyannote pipeline 초기화"""
        if not self.hf_token:
            print("HF_TOKEN이 필요합니다. https://huggingface.co/settings/tokens 에서 토큰을 생성하세요.")
            return False
            
        try:
            # 최신 pyannote 3.1 모델 사용
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            
            # GPU 사용 가능시 GPU로 이동
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
            elif torch.backends.mps.is_available():
                self.pipeline.to(torch.device("mps"))
                
            return True
            
        except Exception as e:
            print(f"Pipeline 초기화 실패: {e}")
            return False
    
    def diarize(self, audio_path, num_speakers=None):
        """화자 분리 수행"""
        if not self.pipeline:
            if not self.initialize():
                return None
                
        # 화자 수를 알고 있는 경우
        if num_speakers:
            diarization = self.pipeline(audio_path, num_speakers=num_speakers)
        else:
            # 자동 감지
            diarization = self.pipeline(audio_path)
            
        # 결과 변환
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
                'duration': turn.duration
            })
            
        return segments
    
    def optimize_parameters(self, audio_path):
        """파라미터 최적화"""
        # 다양한 파라미터로 테스트
        best_params = {
            'min_duration': 1.0,  # 최소 발화 길이
            'min_speakers': 2,    # 최소 화자 수
            'max_speakers': 6     # 최대 화자 수
        }
        
        # 파라미터 튜닝
        for min_dur in [0.5, 1.0, 1.5, 2.0]:
            self.pipeline.params['min_duration'] = min_dur
            # 테스트 및 평가
            
        return best_params