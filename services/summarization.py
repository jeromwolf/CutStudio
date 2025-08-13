"""
요약 서비스
"""
import streamlit as st
from typing import List, Dict, Any, Optional, Union
from datetime import datetime


class SummarizationService:
    """AI 요약 서비스를 관리하는 클래스"""
    
    def __init__(self):
        self.gemini_summarizer = None
        self.claude_summarizer = None
        self.active_summarizer = 'gemini'  # Gemini만 사용
    
    def initialize_summarizers(self):
        """요약기들을 초기화합니다."""
        # Gemini 초기화
        try:
            from services.summarizers.gemini_summarizer import GeminiSummarizer
            self.gemini_summarizer = GeminiSummarizer()
            st.session_state.gemini_summarizer = self.gemini_summarizer
        except Exception as e:
            st.warning(f"Gemini 초기화 실패: {str(e)}")
        
        # Claude 초기화 (현재 비활성화 - Gemini만 사용)
        self.claude_summarizer = None
        print("ℹ️ Claude는 현재 비활성화됨 (Gemini 사용)")
    
    def get_active_summarizer(self):
        """현재 활성화된 요약기를 반환합니다."""
        # Gemini 우선 사용 (Claude는 현재 비활성화)
        if self.gemini_summarizer:
            return self.gemini_summarizer
        elif self.claude_summarizer:
            return self.claude_summarizer
        else:
            return None
    
    def smart_summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50
    ) -> str:
        """
        텍스트를 스마트하게 요약합니다.
        
        Args:
            text: 요약할 텍스트
            max_length: 최대 요약 길이
            min_length: 요약이 필요한 최소 텍스트 길이
            
        Returns:
            요약된 텍스트
        """
        if not text or len(text) < min_length:
            return text
        
        summarizer = self.get_active_summarizer()
        if not summarizer:
            return self._simple_summary(text, max_length)
        
        try:
            # AI 요약 시도
            summary = summarizer.summarize_text(text, max_length)
            return summary
        except Exception as e:
            # 실패 시 간단한 요약
            return self._simple_summary(text, max_length)
    
    def summarize_segments(
        self,
        segments: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        여러 세그먼트를 요약합니다.
        
        Args:
            segments: 요약할 세그먼트 리스트
            progress_callback: 진행 상황 콜백
            
        Returns:
            요약이 추가된 세그먼트 리스트
        """
        summarized_segments = []
        total = len(segments)
        
        for idx, segment in enumerate(segments):
            if progress_callback:
                progress_callback(idx + 1, total, f"요약 중... ({idx + 1}/{total})")
            
            # 텍스트가 있는 경우만 요약
            if 'text' in segment and segment['text']:
                summary = self.smart_summarize(segment['text'])
                segment['summary'] = summary
            
            summarized_segments.append(segment)
        
        return summarized_segments
    
    def generate_conversation_summary(
        self,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        전체 대화를 요약합니다.
        
        Args:
            segments: 인식된 세그먼트 리스트
            
        Returns:
            대화 요약 정보
        """
        print("🔍 generate_conversation_summary() 시작")
        print(f"📊 입력 세그먼트 수: {len(segments) if segments else 0}")
        
        summarizer = self.get_active_summarizer()
        print(f"📡 요약기: {type(summarizer).__name__ if summarizer else 'None'}")
        
        if not summarizer or not segments:
            print("❌ 요약기 또는 세그먼트 없음")
            return {
                'success': False,
                'summary': "요약을 생성할 수 없습니다."
            }
        
        try:
            # 대화 텍스트 준비
            conversation_text = self._prepare_conversation_text(segments)
            
            # 전체 요약 생성
            if hasattr(summarizer, 'summarize_conversation'):
                summary_result = summarizer.summarize_conversation(segments)
            else:
                # 기본 요약
                summary_result = {
                    'overall_summary': self.smart_summarize(conversation_text, 300),
                    'speaker_summaries': self._generate_speaker_summaries(segments),
                    'keywords': self._extract_keywords(conversation_text)
                }
            
            return {
                'success': True,
                'summary': summary_result.get('overall_summary', ''),
                'speaker_summaries': summary_result.get('speaker_summaries', {}),
                'keywords': summary_result.get('keywords', []),
                'total_segments': len(segments),
                'total_speakers': len(set(seg['speaker'] for seg in segments))
            }
            
        except Exception as e:
            return {
                'success': False,
                'summary': f"요약 생성 실패: {str(e)}"
            }
    
    def _prepare_conversation_text(self, segments: List[Dict[str, Any]]) -> str:
        """대화 텍스트를 준비합니다."""
        lines = []
        for seg in segments:
            if seg.get('text', '').strip():
                lines.append(f"{seg['speaker']}: {seg['text']}")
        return "\n".join(lines)
    
    def _generate_speaker_summaries(self, segments: List[Dict[str, Any]]) -> Dict[str, str]:
        """화자별 요약을 생성합니다."""
        speaker_texts = {}
        
        # 화자별 텍스트 수집
        for seg in segments:
            speaker = seg['speaker']
            if speaker not in speaker_texts:
                speaker_texts[speaker] = []
            if seg.get('text', '').strip():
                speaker_texts[speaker].append(seg['text'])
        
        # 화자별 요약 생성
        speaker_summaries = {}
        for speaker, texts in speaker_texts.items():
            combined_text = " ".join(texts)
            if len(combined_text) > 100:
                speaker_summaries[speaker] = self.smart_summarize(combined_text, 200)
            else:
                speaker_summaries[speaker] = combined_text
        
        return speaker_summaries
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """텍스트에서 키워드를 추출합니다."""
        summarizer = self.get_active_summarizer()
        
        if summarizer and hasattr(summarizer, 'extract_keywords'):
            try:
                return summarizer.extract_keywords(text, max_keywords)
            except:
                pass
        
        # 간단한 키워드 추출 (빈도 기반)
        import re
        from collections import Counter
        
        # 불용어 제거 및 단어 추출
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # 빈도 계산
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(max_keywords)]
    
    def _simple_summary(self, text: str, max_length: int) -> str:
        """간단한 요약을 생성합니다."""
        sentences = text.replace('?', '.').replace('!', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # 첫 몇 문장 반환
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) <= max_length:
                summary += sentence + ". "
            else:
                break
        
        return summary.strip() or sentences[0][:max_length] + "..."
    
    def export_summary_report(
        self,
        segments: List[Dict[str, Any]],
        conversation_summary: Dict[str, Any],
        format: str = "markdown"
    ) -> str:
        """
        요약 보고서를 생성합니다.
        
        Args:
            segments: 인식된 세그먼트
            conversation_summary: 대화 요약 정보
            format: 출력 형식
            
        Returns:
            포맷된 보고서
        """
        if format == "markdown":
            return self._export_markdown_report(segments, conversation_summary)
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
    
    def _export_markdown_report(
        self,
        segments: List[Dict[str, Any]],
        summary_info: Dict[str, Any]
    ) -> str:
        """마크다운 형식의 보고서 생성"""
        lines = [
            "# 대화 요약 보고서",
            f"\n생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## 개요",
            f"- 총 발화 수: {summary_info.get('total_segments', 0)}개",
            f"- 참여자 수: {summary_info.get('total_speakers', 0)}명",
            "\n## 전체 요약",
            summary_info.get('summary', '요약 없음'),
            "\n## 화자별 요약"
        ]
        
        # 화자별 요약 추가
        speaker_summaries = summary_info.get('speaker_summaries', {})
        for speaker, summary in speaker_summaries.items():
            lines.append(f"\n### {speaker}")
            lines.append(summary)
        
        # 키워드 추가
        keywords = summary_info.get('keywords', [])
        if keywords:
            lines.append("\n## 주요 키워드")
            lines.append(", ".join(keywords))
        
        return "\n".join(lines)