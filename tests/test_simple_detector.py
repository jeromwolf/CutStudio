#!/usr/bin/env python3
"""
간단한 화자 감지 테스트
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.speaker_detectors.speaker_detector import SpeakerDetector
import time

audio_file = "temp/(Raw)스타트업 성장 컨설팅 23년 2학기 사업계획 발표.m4a"

print("기본 화자 감지기로 테스트...")
detector = SpeakerDetector()

print("\n오디오 추출 중...")
start = time.time()
audio_path = detector.extract_audio(audio_file)
print(f"오디오 추출 완료: {time.time() - start:.2f}초")

if audio_path:
    print(f"추출된 파일: {audio_path}")
    
    # 짧은 구간만 테스트
    print("\n짧은 구간으로 테스트...")
    from pydub import AudioSegment
    
    audio = AudioSegment.from_wav(audio_path)
    print(f"전체 길이: {len(audio)/1000:.1f}초")
    
    # 첫 30초만 테스트
    short_audio = audio[:30000]  # 30초
    test_path = "temp/test_30sec.wav"
    short_audio.export(test_path, format="wav")
    
    print(f"\n30초 구간으로 화자 감지 테스트...")
    start = time.time()
    segments = detector.detect_speakers(test_path)
    print(f"화자 감지 완료: {time.time() - start:.2f}초")
    print(f"감지된 구간: {len(segments)}개")
    
    import os
    os.remove(audio_path)
    os.remove(test_path)