"""
Export Module - 분석 결과 내보내기
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import tempfile


class ExportModule:
    """
    내보내기 모듈
    
    Features:
    - Speaker-specific video extraction
    - Analysis report generation
    - Multiple export formats
    - Batch processing
    """
    
    def __init__(self):
        """모듈 초기화"""
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)
    
    def export_speaker_videos(
        self,
        media_path: Path,
        analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        화자별 비디오 추출
        
        Args:
            media_path: 원본 미디어 파일
            analysis_results: 분석 결과
            
        Returns:
            추출된 파일 정보 리스트
        """
        results = []
        speakers = analysis_results.get("speakers", {})
        
        # 미디어 타입 확인
        is_video = media_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
        
        if is_video:
            video_clip = VideoFileClip(str(media_path))
        else:
            # 오디오 파일
            audio_clip = AudioFileClip(str(media_path))
        
        try:
            for speaker_id, speaker_data in speakers.items():
                segments = speaker_data.get("segments", [])
                
                if not segments:
                    continue
                
                # 세그먼트를 클립으로 변환
                clips = []
                for segment in segments:
                    start = segment["start"]
                    end = segment["end"]
                    
                    if is_video:
                        clip = video_clip.subclip(start, end)
                    else:
                        clip = audio_clip.subclip(start, end)
                    
                    clips.append(clip)
                
                if not clips:
                    continue
                
                # 클립 병합
                if len(clips) > 1:
                    final_clip = concatenate_videoclips(clips)
                else:
                    final_clip = clips[0]
                
                # 파일명 생성
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if is_video:
                    filename = f"{speaker_id}_{timestamp}.mp4"
                else:
                    filename = f"{speaker_id}_{timestamp}.m4a"
                
                output_path = self.export_dir / filename
                
                # 파일 저장
                if is_video:
                    final_clip.write_videofile(
                        str(output_path),
                        codec='libx264',
                        audio_codec='aac',
                        logger=None
                    )
                else:
                    final_clip.write_audiofile(
                        str(output_path),
                        codec='aac',
                        logger=None
                    )
                
                # 메모리 정리
                final_clip.close()
                for clip in clips:
                    clip.close()
                
                # 결과에 추가
                with open(output_path, 'rb') as f:
                    data = f.read()
                
                results.append({
                    "speaker": speaker_id,
                    "filename": filename,
                    "path": str(output_path),
                    "data": data,
                    "duration": speaker_data.get("duration", 0),
                    "summary": speaker_data.get("summary", "")
                })
            
        finally:
            # 리소스 정리
            if is_video and 'video_clip' in locals():
                video_clip.close()
            elif 'audio_clip' in locals():
                audio_clip.close()
        
        return results
    
    def generate_report(
        self,
        media_info: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> bytes:
        """
        분석 보고서 생성 (HTML → Markdown)
        
        Args:
            media_info: 미디어 정보
            analysis_results: 분석 결과
            
        Returns:
            보고서 텍스트 바이트
        """
        # Markdown 형식으로 보고서 생성
        report_lines = []
        
        # 제목
        report_lines.append("# CutStudio Pro - 분석 보고서")
        report_lines.append("")
        
        # 메타데이터
        report_lines.append("## 파일 정보")
        report_lines.append(f"- **파일명**: {media_info.get('filename', 'Unknown')}")
        report_lines.append(f"- **길이**: {self._format_duration(media_info.get('duration', 0))}")
        report_lines.append(f"- **형식**: {media_info.get('format', 'Unknown')}")
        report_lines.append(f"- **분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 화자별 분석
        speakers = analysis_results.get("speakers", {})
        
        if speakers:
            report_lines.append("## 화자별 분석 결과")
            report_lines.append("")
            
            # 전체 발화 시간 계산
            total_duration = sum(s.get("duration", 0) for s in speakers.values())
            
            # 화자별 정보
            for speaker_id, data in speakers.items():
                duration = data.get("duration", 0)
                ratio = (duration / total_duration * 100) if total_duration > 0 else 0
                
                report_lines.append(f"### {speaker_id}")
                report_lines.append(f"- **발화 시간**: {self._format_duration(duration)} ({ratio:.1f}%)")
                report_lines.append(f"- **세그먼트 수**: {len(data.get('segments', []))}")
                
                if summary := data.get("summary"):
                    report_lines.append(f"- **요약**: {summary}")
                
                if keywords := data.get("keywords"):
                    report_lines.append(f"- **키워드**: {', '.join(keywords)}")
                
                report_lines.append("")
        
        # 전체 요약
        if "summary" in analysis_results:
            report_lines.append("## 전체 요약")
            
            if full_summary := analysis_results["summary"].get("full"):
                report_lines.append(full_summary)
                report_lines.append("")
            
            if keywords := analysis_results["summary"].get("keywords"):
                report_lines.append(f"**주요 키워드**: {', '.join(keywords)}")
                report_lines.append("")
        
        # 화자별 발화 내용 (옵션)
        if speakers:
            report_lines.append("## 상세 발화 내용")
            report_lines.append("")
            
            for speaker_id, data in speakers.items():
                segments = data.get("segments", [])
                if segments and any(seg.get("text") for seg in segments):
                    report_lines.append(f"### {speaker_id}")
                    
                    for seg in segments:
                        if text := seg.get("text"):
                            time_str = f"[{self._format_time(seg['start'])} - {self._format_time(seg['end'])}]"
                            report_lines.append(f"- {time_str} {text}")
                    
                    report_lines.append("")
        
        # 텍스트를 바이트로 변환
        report_text = "\n".join(report_lines)
        return report_text.encode('utf-8')
    
    def export_transcription(
        self,
        analysis_results: Dict[str, Any],
        format: str = "txt"
    ) -> str:
        """
        음성 인식 결과 내보내기
        
        Args:
            analysis_results: 분석 결과
            format: 출력 형식 (txt, srt, vtt)
            
        Returns:
            포맷된 텍스트
        """
        speakers = analysis_results.get("speakers", {})
        all_segments = []
        
        # 모든 세그먼트 수집 및 정렬
        for speaker_data in speakers.values():
            segments = speaker_data.get("segments", [])
            all_segments.extend(segments)
        
        all_segments.sort(key=lambda x: x["start"])
        
        if format == "txt":
            return self._export_as_text(all_segments)
        elif format == "srt":
            return self._export_as_srt(all_segments)
        elif format == "vtt":
            return self._export_as_vtt(all_segments)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_as_text(self, segments: List[Dict[str, Any]]) -> str:
        """텍스트 형식으로 내보내기"""
        lines = []
        for seg in segments:
            time_str = f"[{self._format_time(seg['start'])} - {self._format_time(seg['end'])}]"
            text = seg.get("text", "")
            lines.append(f"{time_str} {seg['speaker']}: {text}")
        return "\n\n".join(lines)
    
    def _export_as_srt(self, segments: List[Dict[str, Any]]) -> str:
        """SRT 자막 형식으로 내보내기"""
        lines = []
        for idx, seg in enumerate(segments, 1):
            lines.append(str(idx))
            lines.append(f"{self._format_srt_time(seg['start'])} --> {self._format_srt_time(seg['end'])}")
            text = seg.get("text", "")
            lines.append(f"{seg['speaker']}: {text}")
            lines.append("")
        return "\n".join(lines)
    
    def _export_as_vtt(self, segments: List[Dict[str, Any]]) -> str:
        """WebVTT 자막 형식으로 내보내기"""
        lines = ["WEBVTT", ""]
        for seg in segments:
            lines.append(f"{self._format_vtt_time(seg['start'])} --> {self._format_vtt_time(seg['end'])}")
            text = seg.get("text", "")
            lines.append(f"{seg['speaker']}: {text}")
            lines.append("")
        return "\n".join(lines)
    
    def _format_duration(self, seconds: float) -> str:
        """초를 시:분:초 형식으로 변환"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}시간 {minutes}분 {secs}초"
        elif minutes > 0:
            return f"{minutes}분 {secs}초"
        else:
            return f"{secs}초"
    
    def _format_time(self, seconds: float) -> str:
        """시간을 MM:SS 형식으로 변환"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _format_srt_time(self, seconds: float) -> str:
        """시간을 SRT 형식으로 변환"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _format_vtt_time(self, seconds: float) -> str:
        """시간을 WebVTT 형식으로 변환"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"