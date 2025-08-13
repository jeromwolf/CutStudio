#!/usr/bin/env python3
"""
간단한 오디오 파일 테스트
"""
from moviepy.editor import AudioFileClip
import numpy as np

audio_file = "temp/(Raw)스타트업 성장 컨설팅 23년 2학기 사업계획 발표.m4a"

print("오디오 파일 정보:")
try:
    audio = AudioFileClip(audio_file)
    print(f"- 길이: {audio.duration}초 ({audio.duration/60:.1f}분)")
    print(f"- FPS: {audio.fps}")
    print(f"- 채널: {audio.nchannels if hasattr(audio, 'nchannels') else 'Unknown'}")
    
    # 첫 10초만 테스트
    print("\n첫 10초 오디오 추출 테스트...")
    short_clip = audio.subclip(0, 10)
    
    # WAV로 저장
    test_output = "temp/test_10sec.wav"
    short_clip.write_audiofile(test_output, verbose=False, logger=None)
    print(f"테스트 파일 저장: {test_output}")
    
    audio.close()
    short_clip.close()
    
except Exception as e:
    print(f"오류: {e}")
    import traceback
    traceback.print_exc()