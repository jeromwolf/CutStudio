#!/usr/bin/env python3
"""
간단한 화자 감지 테스트
"""
from speaker_detector import SpeakerDetector
import time

audio_file = "temp/(Raw)스타트업 성장 컨설팅 23년 2학기 사업계획 발표.m4a"

try:
    print("기본 화자 감지기로 테스트...")
    detector = SpeakerDetector()
    
    print("화자 감지 시작...")
    start_time = time.time()
    
    # 간단한 화자 감지 (화자 수 지정)
    segments = detector.detect_speakers(audio_file, num_speakers=2)
    
    end_time = time.time()
    
    print(f"\n결과:")
    print(f"- 처리 시간: {end_time - start_time:.2f}초")
    print(f"- 감지된 구간: {len(segments)}개")
    
    if segments:
        print("\n구간별 정보:")
        for i, seg in enumerate(segments[:5]):  # 처음 5개만 표시
            print(f"  {i+1}. {seg['speaker']}: {seg['start']:.2f}s ~ {seg['end']:.2f}s ({seg['duration']:.2f}s)")
        
        # 화자별 총 시간
        speakers = {}
        for seg in segments:
            speaker = seg['speaker']
            if speaker not in speakers:
                speakers[speaker] = 0
            speakers[speaker] += seg['duration']
        
        print(f"\n화자별 발화 시간:")
        for speaker, duration in speakers.items():
            print(f"  {speaker}: {duration:.1f}초 ({duration/60:.1f}분)")
    else:
        print("화자가 감지되지 않았습니다.")
        
except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()