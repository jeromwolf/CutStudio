"""
음성 처리 서비스
"""
import streamlit as st
from typing import List, Dict, Any, Optional, Callable
import tempfile
import os
from datetime import datetime


class SpeechProcessor:
    """음성 인식 처리를 담당하는 서비스 클래스"""
    
    def __init__(self, speech_recognizer=None):
        """
        Args:
            speech_recognizer: SpeechRecognizer 인스턴스
        """
        self.speech_recognizer = speech_recognizer
    
    def execute_recognition(
        self,
        video_path: str,
        segments: List[Dict[str, Any]],
        model_size: str = "tiny",
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        음성 인식을 실행합니다.
        
        Args:
            video_path: 비디오 파일 경로
            segments: 화자 세그먼트 리스트
            model_size: Whisper 모델 크기
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            인식된 세그먼트 리스트
        """
        if not self.speech_recognizer:
            raise ValueError("SpeechRecognizer가 초기화되지 않았습니다.")
        
        recognized_segments = []
        total_segments = len(segments)
        
        # 진행 상황 컨테이너
        if progress_callback:
            progress_callback(0, total_segments, "음성 인식 시작...")
        
        # 세그먼트별 음성 인식
        for idx, segment in enumerate(segments):
            try:
                # 오디오 추출
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    audio_path = tmp_file.name
                
                # 세그먼트의 오디오 추출 (video_editor 필요)
                if hasattr(st.session_state, 'video_editor') and st.session_state.video_editor:
                    audio_clip = st.session_state.video_editor.video_clip.subclip(
                        segment['start'], 
                        segment['end']
                    ).audio
                    
                    if audio_clip:
                        audio_clip.write_audiofile(audio_path, logger=None)
                        
                        # 음성 인식 실행
                        result = self.speech_recognizer.transcribe_audio(
                            audio_path
                        )
                        
                        if result and result.get('text', '').strip():
                            recognized_segment = {
                                'speaker': segment['speaker'],
                                'start': segment['start'],
                                'end': segment['end'],
                                'duration': segment['duration'],
                                'text': result['text'].strip(),
                                'language': result.get('language', 'unknown')
                            }
                            recognized_segments.append(recognized_segment)
                    
                    # 임시 파일 정리
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
                
                # 진행 상황 업데이트
                if progress_callback:
                    progress = (idx + 1) / total_segments
                    progress_callback(
                        idx + 1, 
                        total_segments, 
                        f"처리 중... {segment['speaker']} ({idx + 1}/{total_segments})"
                    )
                    
            except Exception as e:
                st.warning(f"세그먼트 {idx + 1} 처리 실패: {str(e)}")
                continue
        
        return recognized_segments
    
    def map_whisper_to_segments(
        self,
        whisper_segments: List[Dict[str, Any]],
        speaker_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Whisper 결과를 화자 세그먼트에 매핑합니다.
        
        Args:
            whisper_segments: Whisper 인식 결과
            speaker_segments: 화자 세그먼트
            
        Returns:
            매핑된 세그먼트 리스트
        """
        recognized_segments = []
        
        for speaker_seg in speaker_segments:
            speaker_text = []
            
            for whisper_seg in whisper_segments:
                # 시간 기반 매칭 (오버랩 확인)
                if (whisper_seg['start'] < speaker_seg['end'] and 
                    whisper_seg['end'] > speaker_seg['start']):
                    speaker_text.append(whisper_seg['text'].strip())
            
            if speaker_text:
                recognized_segment = {
                    'speaker': speaker_seg['speaker'],
                    'start': speaker_seg['start'],
                    'end': speaker_seg['end'],
                    'duration': speaker_seg['duration'],
                    'text': ' '.join(speaker_text),
                    'language': whisper_segments[0].get('language', 'unknown') if whisper_segments else 'unknown'
                }
                recognized_segments.append(recognized_segment)
        
        return recognized_segments
    
    def process_recognition_results(
        self,
        recognized_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        인식 결과를 처리하고 통계를 생성합니다.
        
        Args:
            recognized_segments: 인식된 세그먼트 리스트
            
        Returns:
            처리 결과 및 통계
        """
        if not recognized_segments:
            return {
                'success': False,
                'total_segments': 0,
                'recognized_segments': 0,
                'total_duration': 0,
                'languages': set()
            }
        
        total_duration = sum(seg['duration'] for seg in recognized_segments)
        languages = set(seg.get('language', 'unknown') for seg in recognized_segments)
        
        # 화자별 통계
        speaker_stats = {}
        for seg in recognized_segments:
            speaker = seg['speaker']
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'count': 0,
                    'total_duration': 0,
                    'total_words': 0
                }
            
            stats = speaker_stats[speaker]
            stats['count'] += 1
            stats['total_duration'] += seg['duration']
            stats['total_words'] += len(seg['text'].split())
        
        return {
            'success': True,
            'total_segments': len(recognized_segments),
            'recognized_segments': len([s for s in recognized_segments if s.get('text', '').strip()]),
            'total_duration': total_duration,
            'languages': languages,
            'speaker_stats': speaker_stats,
            'segments': recognized_segments
        }
    
    def export_transcription(
        self,
        recognized_segments: List[Dict[str, Any]],
        format: str = "txt"
    ) -> str:
        """
        음성 인식 결과를 특정 형식으로 내보냅니다.
        
        Args:
            recognized_segments: 인식된 세그먼트 리스트
            format: 출력 형식 ("txt", "srt", "vtt")
            
        Returns:
            포맷된 텍스트
        """
        if format == "txt":
            return self._export_as_text(recognized_segments)
        elif format == "srt":
            return self._export_as_srt(recognized_segments)
        elif format == "vtt":
            return self._export_as_vtt(recognized_segments)
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
    
    def _export_as_text(self, segments: List[Dict[str, Any]]) -> str:
        """텍스트 형식으로 내보내기"""
        lines = []
        for seg in segments:
            time_str = f"[{self._format_time(seg['start'])} - {self._format_time(seg['end'])}]"
            lines.append(f"{time_str} {seg['speaker']}: {seg['text']}")
        return "\n\n".join(lines)
    
    def _export_as_srt(self, segments: List[Dict[str, Any]]) -> str:
        """SRT 자막 형식으로 내보내기"""
        lines = []
        for idx, seg in enumerate(segments, 1):
            lines.append(str(idx))
            lines.append(f"{self._format_srt_time(seg['start'])} --> {self._format_srt_time(seg['end'])}")
            lines.append(f"{seg['speaker']}: {seg['text']}")
            lines.append("")
        return "\n".join(lines)
    
    def _export_as_vtt(self, segments: List[Dict[str, Any]]) -> str:
        """VTT 자막 형식으로 내보내기"""
        lines = ["WEBVTT", ""]
        for seg in segments:
            lines.append(f"{self._format_vtt_time(seg['start'])} --> {self._format_vtt_time(seg['end'])}")
            lines.append(f"{seg['speaker']}: {seg['text']}")
            lines.append("")
        return "\n".join(lines)
    
    def _format_time(self, seconds: float) -> str:
        """시간을 MM:SS 형식으로 변환"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _format_srt_time(self, seconds: float) -> str:
        """시간을 SRT 형식으로 변환 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _format_vtt_time(self, seconds: float) -> str:
        """시간을 VTT 형식으로 변환 (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"