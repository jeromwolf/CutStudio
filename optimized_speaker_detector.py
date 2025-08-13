"""
최적화된 화자 감지기 - 긴 오디오 파일 처리용
"""
import os
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import numpy as np
from speaker_detector import SpeakerDetector
from practical_speaker_detector import PracticalSpeakerDetector
import streamlit as st


class OptimizedSpeakerDetector:
    """긴 오디오 파일을 위한 최적화된 화자 감지기"""
    
    def __init__(self, base_detector='practical'):
        """
        Args:
            base_detector: 사용할 기본 감지기 ('simple' 또는 'practical')
        """
        if base_detector == 'practical':
            self.detector = PracticalSpeakerDetector()
        else:
            self.detector = SpeakerDetector()
            
        self.chunk_duration_ms = 10 * 60 * 1000  # 10분 청크
        self.overlap_ms = 5000  # 5초 오버랩
    
    def extract_audio(self, media_path):
        """미디어 파일에서 오디오 추출"""
        return self.detector.extract_audio(media_path)
    
    def detect_speakers_chunked(self, audio_path, num_speakers=0, progress_callback=None):
        """청크 단위로 화자 감지 수행"""
        try:
            # 오디오 로드
            audio = AudioSegment.from_wav(audio_path)
            total_duration_ms = len(audio)
            total_duration_sec = total_duration_ms / 1000
            
            print(f"오디오 길이: {total_duration_sec:.1f}초 ({total_duration_sec/60:.1f}분)")
            
            # 전체 세그먼트 저장
            all_segments = []
            
            # 청크 단위로 처리
            chunk_count = 0
            for start_ms in range(0, total_duration_ms, self.chunk_duration_ms - self.overlap_ms):
                chunk_count += 1
                end_ms = min(start_ms + self.chunk_duration_ms, total_duration_ms)
                
                # 진행률 표시
                progress = start_ms / total_duration_ms
                if progress_callback:
                    progress_callback(progress)
                
                print(f"\n청크 {chunk_count} 처리 중: {start_ms/1000:.1f}초 ~ {end_ms/1000:.1f}초")
                
                # 청크 추출
                chunk = audio[start_ms:end_ms]
                
                # 임시 파일로 저장
                chunk_path = tempfile.mktemp(suffix=".wav")
                chunk.export(chunk_path, format="wav")
                
                try:
                    # 청크에서 화자 감지
                    if hasattr(self.detector, 'detect_speakers'):
                        segments = self.detector.detect_speakers(chunk_path, num_speakers=num_speakers)
                    else:
                        segments = []
                    
                    # 시간 오프셋 조정
                    offset_sec = start_ms / 1000
                    for seg in segments:
                        seg['start'] += offset_sec
                        seg['end'] += offset_sec
                    
                    all_segments.extend(segments)
                    
                finally:
                    # 임시 파일 삭제
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
            
            # 중복 제거 및 병합
            merged_segments = self._merge_overlapping_segments(all_segments)
            
            # 화자 ID 재할당
            final_segments = self._reassign_speakers(merged_segments, num_speakers)
            
            return final_segments
            
        except Exception as e:
            print(f"청크 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _merge_overlapping_segments(self, segments):
        """오버랩되는 세그먼트 병합"""
        if not segments:
            return []
        
        # 시작 시간으로 정렬
        segments = sorted(segments, key=lambda x: x['start'])
        
        merged = []
        current = segments[0].copy()
        
        for seg in segments[1:]:
            # 같은 화자이고 시간이 겹치거나 가까운 경우
            if (seg['speaker'] == current['speaker'] and 
                seg['start'] <= current['end'] + 2.0):  # 2초 이내면 병합
                current['end'] = max(current['end'], seg['end'])
                current['duration'] = current['end'] - current['start']
            else:
                merged.append(current)
                current = seg.copy()
        
        merged.append(current)
        return merged
    
    def _reassign_speakers(self, segments, num_speakers=0):
        """화자 ID 재할당 (일관성 유지)"""
        if not segments:
            return []
        
        # 화자별 총 발화 시간 계산
        speaker_durations = {}
        for seg in segments:
            speaker = seg['speaker']
            duration = seg['end'] - seg['start']
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0
            speaker_durations[speaker] += duration
        
        # 발화 시간이 긴 순서로 정렬
        sorted_speakers = sorted(speaker_durations.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # 새로운 화자 ID 매핑
        speaker_map = {}
        for i, (old_speaker, _) in enumerate(sorted_speakers):
            if num_speakers > 0 and i >= num_speakers:
                # 지정된 화자 수를 초과하면 마지막 화자로 할당
                speaker_map[old_speaker] = f'SPEAKER_{num_speakers-1}'
            else:
                speaker_map[old_speaker] = f'SPEAKER_{i}'
        
        # 세그먼트 업데이트
        for seg in segments:
            seg['speaker'] = speaker_map.get(seg['speaker'], seg['speaker'])
        
        return segments
    
    def detect_speakers(self, video_path, num_speakers=0, min_duration=1.0, progress_callback=None):
        """화자 감지 메인 메서드"""
        try:
            # 오디오 추출
            print("오디오 추출 중...")
            audio_path = self.extract_audio(video_path)
            if not audio_path:
                return []
            
            # 오디오 길이 확인
            audio = AudioSegment.from_wav(audio_path)
            duration_minutes = len(audio) / 1000 / 60
            
            if duration_minutes > 30:  # 30분 이상이면 청크 처리
                print(f"긴 오디오 파일 ({duration_minutes:.1f}분) - 청크 단위로 처리합니다...")
                segments = self.detect_speakers_chunked(audio_path, num_speakers, progress_callback)
            else:
                # 짧은 파일은 기존 방식으로 처리
                print(f"일반 처리 ({duration_minutes:.1f}분)")
                segments = self.detector.detect_speakers(video_path, num_speakers=num_speakers)
            
            # 임시 파일 삭제
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return segments
            
        except Exception as e:
            print(f"화자 감지 실패: {e}")
            import traceback
            traceback.print_exc()
            return []