#!/usr/bin/env python3
"""
오디오 세그먼트 테스트
"""
from moviepy.editor import AudioFileClip
import numpy as np

audio_file = "temp/(Raw)스타트업 성장 컨설팅 23년 2학기 사업계획 발표.m4a"

try:
    print("오디오 파일 로드 중...")
    audio = AudioFileClip(audio_file)
    
    print(f"오디오 정보:")
    print(f"- 길이: {audio.duration}초")
    print(f"- FPS: {audio.fps}")
    print(f"- 채널: {audio.nchannels if hasattr(audio, 'nchannels') else 'Unknown'}")
    
    # 첫 30초 테스트
    print("\n30초 구간 테스트...")
    segment = audio.subclip(0, 30)
    
    # 오디오 데이터 확인
    audio_array = segment.to_soundarray()
    print(f"- 오디오 배열 shape: {audio_array.shape}")
    print(f"- 최대값: {np.max(np.abs(audio_array))}")
    print(f"- 평균값: {np.mean(np.abs(audio_array))}")
    
    # 무음 체크
    if np.max(np.abs(audio_array)) < 0.01:
        print("⚠️ 오디오가 너무 조용합니다!")
    else:
        print("✓ 오디오 신호가 정상입니다.")
    
    audio.close()
    segment.close()
    
except Exception as e:
    print(f"오류: {e}")
    import traceback
    traceback.print_exc()