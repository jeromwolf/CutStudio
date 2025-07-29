import os
import google.generativeai as genai
from typing import List, Dict

class GeminiSummarizer:
    def __init__(self):
        """Gemini API 초기화"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
        genai.configure(api_key=api_key)
        # gemini-pro -> gemini-1.5-flash로 변경 (최신 모델)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """텍스트 요약"""
        if not text or len(text.strip()) < 10:
            return "내용이 너무 짧습니다."
        
        try:
            prompt = f"""
            다음 텍스트를 {max_length}자 이내로 요약해주세요. 
            핵심 내용만 간단명료하게 정리해주세요.
            
            텍스트: {text}
            
            요약:
            """
            
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            
            # 길이 제한
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return summary
            
        except Exception as e:
            print(f"Gemini 요약 실패: {e}")
            # 실패 시 간단한 요약 반환
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def summarize_conversation(self, segments: List[Dict]) -> Dict[str, str]:
        """화자별 대화 내용 요약 (화자 중심 분석)"""
        speaker_texts = {}
        
        # 화자별로 텍스트 수집
        for seg in segments:
            speaker = seg.get('speaker', 'UNKNOWN')
            text = seg.get('text', '').strip()
            
            if text and len(text) > 10:  # 최소 길이 체크
                if speaker not in speaker_texts:
                    speaker_texts[speaker] = []
                speaker_texts[speaker].append(text)
        
        # 화자별 요약 생성
        speaker_summaries = {}
        for speaker, texts in speaker_texts.items():
            combined_text = ' '.join(texts)
            if len(combined_text.strip()) >= 50:  # 최소 50자 이상일 때만 요약 시도
                try:
                    prompt = f"""
                    다음은 {speaker}이(가) 말한 내용들입니다. 
                    이 화자의 주요 발언 내용을 2-3문장으로 간단히 요약해주세요.
                    각 화자의 개별 발언에 집중하여 요약하세요.
                    
                    발화 내용: {combined_text}
                    
                    {speaker} 요약:
                    """
                    
                    response = self.model.generate_content(prompt)
                    summary = response.text.strip()
                    
                    # 길이 제한 (최대 200자)
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    
                    speaker_summaries[speaker] = summary
                    
                except Exception as e:
                    print(f"{speaker} 요약 실패: {e}")
                    speaker_summaries[speaker] = self._simple_summary(combined_text, 100)
            else:
                speaker_summaries[speaker] = "발화 내용이 너무 짧습니다"
        
        return speaker_summaries
    
    def generate_meeting_summary(self, segments: List[Dict]) -> str:
        """전체 대화/회의 요약"""
        all_texts = []
        speaker_order = []
        
        # 시간순으로 정렬
        sorted_segments = sorted(segments, key=lambda x: x.get('start', 0))
        
        for seg in sorted_segments:
            speaker = seg.get('speaker', 'UNKNOWN')
            text = seg.get('text', '').strip()
            
            if text:
                all_texts.append(f"{speaker}: {text}")
                if speaker not in speaker_order:
                    speaker_order.append(speaker)
        
        if not all_texts:
            return "대화 내용이 없습니다."
        
        try:
            conversation_text = '\n'.join(all_texts)
            
            prompt = f"""
            다음은 {len(speaker_order)}명이 참여한 대화 내용입니다.
            이 대화의 주요 내용을 다음 형식으로 정리해주세요:
            
            1. 대화 주제
            2. 주요 논의 사항 (3-5개)
            3. 각 참여자의 핵심 발언
            4. 결론 또는 합의사항 (있다면)
            
            대화 내용:
            {conversation_text}
            
            요약:
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"전체 요약 생성 실패: {e}")
            return "요약 생성에 실패했습니다."
    
    def _simple_summary(self, text: str, max_length: int = 150) -> str:
        """간단한 텍스트 요약 (API 실패 시 대체)"""
        if len(text) <= max_length:
            return text
        
        # 문장 단위로 자르기
        sentences = text.replace('?', '.').replace('!', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return text[:max_length] + "..."
        
        # 첫 몇 문장만 사용
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) < max_length:
                summary += sentence + ". "
            else:
                break
        
        return summary.strip() or text[:max_length] + "..."
    
    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """주요 키워드 추출"""
        try:
            prompt = f"""
            다음 텍스트에서 가장 중요한 키워드 {num_keywords}개를 추출해주세요.
            쉼표로 구분해서 나열해주세요.
            
            텍스트: {text}
            
            키워드:
            """
            
            response = self.model.generate_content(prompt)
            keywords = response.text.strip().split(',')
            keywords = [k.strip() for k in keywords if k.strip()]
            
            return keywords[:num_keywords]
            
        except Exception as e:
            print(f"키워드 추출 실패: {e}")
            # 간단한 키워드 추출 대체
            words = text.split()
            word_freq = {}
            for word in words:
                word = word.strip('.,!?()[]{}""\'')
                if len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:num_keywords]]