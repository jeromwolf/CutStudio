#!/usr/bin/env python3
"""
화자 감지 테스트 스크립트
"""
import sys
import time
from practical_speaker_detector import PracticalSpeakerDetector

def test_speaker_detection():
    print("화자 감지 테스트 시작...")
    
    # M4A 파일 경로
    audio_file = "temp/(Raw)스타트업 성장 컨설팅 23년 2학기 사업계획 발표.m4a"
    
    print(f"파일: {audio_file}")
    
    try:
        # 감지기 초기화
        detector = PracticalSpeakerDetector()
        
        # 시작 시간
        start_time = time.time()
        
        # 화자 감지 실행
        print("화자 감지 중...")
        segments = detector.detect_speakers(audio_file, num_speakers=0)
        
        # 종료 시간
        end_time = time.time()
        
        print(f"\n화자 감지 완료!")
        print(f"소요 시간: {end_time - start_time:.2f}초")
        print(f"감지된 구간 수: {len(segments)}")
        
        # 화자별 통계
        speakers = {}
        for seg in segments:
            speaker = seg['speaker']
            if speaker not in speakers:
                speakers[speaker] = 0
            speakers[speaker] += seg['end'] - seg['start']
        
        print(f"\n화자별 발화 시간:")
        for speaker, duration in speakers.items():
            print(f"  {speaker}: {duration:.2f}초 ({duration/60:.1f}분)")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_speaker_detection()