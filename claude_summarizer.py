import os
import anthropic
from typing import List, Dict

class ClaudeSummarizer:
    def __init__(self):
        """Claude API 초기화"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """텍스트 요약"""
        if not text or len(text.strip()) < 10:
            return "내용이 너무 짧습니다."
        
        try:
            prompt = f"""다음 텍스트를 {max_length}자 이내로 요약해주세요. 핵심 내용만 간단명료하게 정리해주세요.

텍스트: {text}

요약:"""
            
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            summary = message.content[0].text.strip()
            
            # 길이 제한 (문장 단위로)
            if len(summary) > max_length:
                sentences = summary.split('.')
                result = ""
                for sentence in sentences:
                    if len(result + sentence + ".") <= max_length:
                        result += sentence + "."
                    else:
                        break
                summary = result if result else summary[:max_length]
            
            return summary
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                print(f"Claude API 할당량/속도 제한: {e}")
            else:
                print(f"Claude 요약 실패: {e}")
            # API 실패 시 간단한 요약 반환
            return self._simple_summary(text, max_length)
    
    def summarize_conversation(self, segments: List[Dict]) -> Dict[str, str]:
        """화자별 대화 내용 요약"""
        speaker_texts = {}
        
        # 화자별로 텍스트 수집
        for seg in segments:
            speaker = seg.get('speaker', 'UNKNOWN')
            text = seg.get('text', '').strip()
            
            if text:
                if speaker not in speaker_texts:
                    speaker_texts[speaker] = []
                speaker_texts[speaker].append(text)
        
        speaker_summaries = {}
        
        for speaker, texts in speaker_texts.items():
            combined_text = ' '.join(texts)
            
            if len(combined_text) > 50:
                try:
                    prompt = f"""{speaker}이(가) 말한 내용들을 2-3문장으로 간단히 요약해주세요.

발화 내용: {combined_text}

{speaker} 요약:"""
                    
                    message = self.client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=300,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }]
                    )
                    
                    summary = message.content[0].text.strip()
                    
                    # 길이 제한 (최대 200자)
                    if len(summary) > 200:
                        sentences = summary.split('.')
                        result = ""
                        for sentence in sentences:
                            if len(result + sentence + ".") <= 200:
                                result += sentence + "."
                            else:
                                break
                        summary = result if result else summary[:200]
                    
                    speaker_summaries[speaker] = summary
                    
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                        print(f"{speaker} 요약 실패 (Claude API 할당량/속도 제한): {e}")
                        speaker_summaries[speaker] = f"[API 제한] {self._simple_summary(combined_text, 100)}"
                    else:
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
            
            prompt = f"""다음은 {len(speaker_order)}명이 참여한 대화 내용입니다.
이 대화의 주요 내용을 다음 형식으로 정리해주세요:

1. 대화 주제
2. 주요 논의 사항 (3-5개)
3. 각 참여자의 핵심 발언
4. 결론 또는 합의사항 (있다면)

대화 내용:
{conversation_text}

요약:"""
            
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                print(f"전체 요약 생성 실패 (Claude API 할당량/속도 제한): {e}")
                return f"대화 참여자: {', '.join(speaker_order)}\n총 {len(all_texts)}개 발언\n[Claude API 제한으로 요약 생성 불가]"
            else:
                print(f"전체 요약 생성 실패: {e}")
                return f"대화 참여자: {', '.join(speaker_order)}\n총 {len(all_texts)}개 발언\n[요약 생성 실패]"
    
    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """주요 키워드 추출"""
        try:
            prompt = f"""다음 텍스트에서 가장 중요한 키워드 {num_keywords}개를 추출해주세요.
쉼표로 구분해서 나열해주세요.

텍스트: {text}

키워드:"""
            
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            keywords = message.content[0].text.strip().split(',')
            keywords = [k.strip() for k in keywords if k.strip()]
            
            return keywords[:num_keywords]
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                print(f"키워드 추출 실패 (Claude API 할당량/속도 제한): {e}")
            else:
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
    
    def _simple_summary(self, text: str, max_length: int = 150) -> str:
        """간단한 텍스트 요약 (API 실패 시 대체)"""
        if len(text) <= max_length:
            return text
        
        # 문장 단위로 자르기
        sentences = text.replace('?', '.').replace('!', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return text[:max_length] + "..."
        
        # 첫 번째 문장부터 길이 제한까지 추가
        result = ""
        for sentence in sentences:
            if len(result + sentence + ". ") <= max_length:
                result += sentence + ". "
            else:
                break
        
        return result.strip() if result else text[:max_length] + "..."