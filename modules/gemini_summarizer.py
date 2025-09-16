"""
AI 요약 모듈 (AI Summary Module)
CutStudio v5.0 - Google Gemini 기반

핵심 기능:
1. 음성 인식 텍스트 자동 요약
2. 화자별 발언 요약
3. 핵심 키워드 추출
4. 교육 콘텐츠 특화 요약
"""

import os
import time
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Summary length definitions
SummaryLength = Literal["very_short", "short", "medium", "long", "very_long"]

@dataclass
class SummaryResult:
    """요약 결과 정보"""
    summary: str
    key_points: List[str]
    keywords: List[str]
    speaker_summaries: Dict[str, str] = None
    confidence: float = 1.0
    processing_time: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'summary': self.summary,
            'key_points': self.key_points,
            'keywords': self.keywords,
            'speaker_summaries': self.speaker_summaries or {},
            'confidence': round(self.confidence, 2),
            'processing_time': round(self.processing_time, 2)
        }


class GeminiSummarizer:
    """
    Google Gemini 기반 AI 요약기
    교육 콘텐츠에 최적화된 요약 생성
    """

    # 요약 길이 설정
    LENGTH_SETTINGS = {
        "very_short": {"words": 50, "points": 3},
        "short": {"words": 100, "points": 5},
        "medium": {"words": 200, "points": 7},
        "long": {"words": 300, "points": 10},
        "very_long": {"words": 500, "points": 15}
    }

    def __init__(self, api_key: Optional[str] = None, use_api: bool = False):
        """
        Args:
            api_key: Gemini API 키 (None이면 환경변수에서 읽음)
            use_api: API 사용 여부 (기본값 False로 비용 발생 안함)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.use_api = use_api  # 사용자가 명시적으로 활성화해야 API 사용
        self.model = None

        # API 사용 경고
        if self.use_api and self.api_key:
            print("⚠️ Gemini API 활성화됨 - 사용 시 비용이 발생할 수 있습니다!")
            self._initialize_model()
        else:
            print("ℹ️ 로컬 요약 모드 (API 비용 없음)")

    def _initialize_model(self):
        """Gemini 모델 초기화"""
        if not self.api_key:
            print("⚠️ GEMINI_API_KEY not found. Using fallback mode.")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini 모델 초기화 완료")
        except ImportError:
            print("⚠️ google-generativeai not installed. Using fallback mode.")
            self.model = None
        except Exception as e:
            print(f"⚠️ Gemini initialization failed: {e}")
            self.model = None

    def summarize_text(self,
                      text: str,
                      length: SummaryLength = "medium",
                      focus: str = "education") -> SummaryResult:
        """
        텍스트 요약 생성

        Args:
            text: 요약할 텍스트
            length: 요약 길이
            focus: 요약 초점 ("education", "business", "general")

        Returns:
            요약 결과
        """
        start_time = time.time()

        if not text or len(text) < 50:
            return self._generate_minimal_summary(text)

        # API 사용하지 않거나 모델이 없으면 로컬 요약
        if not self.use_api or self.model is None:
            return self._generate_smart_local_summary(text, length)

        try:
            # Gemini API 호출
            settings = self.LENGTH_SETTINGS[length]

            prompt = self._create_prompt(text, settings, focus)
            response = self.model.generate_content(prompt)

            # 응답 파싱
            result = self._parse_response(response.text)

            processing_time = time.time() - start_time

            return SummaryResult(
                summary=result.get('summary', ''),
                key_points=result.get('key_points', []),
                keywords=result.get('keywords', []),
                confidence=0.95,
                processing_time=processing_time
            )

        except Exception as e:
            print(f"❌ 요약 생성 실패: {e}")
            return self._generate_fallback_summary(text, length)

    def summarize_with_speakers(self,
                               transcript_segments: List[Dict],
                               speaker_info: Dict[str, List],
                               length: SummaryLength = "medium") -> SummaryResult:
        """
        화자별 음성 인식 결과 요약

        Args:
            transcript_segments: 음성 인식 세그먼트
            speaker_info: 화자별 정보
            length: 요약 길이

        Returns:
            화자별 요약 포함 결과
        """
        start_time = time.time()

        # 전체 텍스트 생성
        full_text = self._combine_segments(transcript_segments)

        # 화자별 텍스트 분리
        speaker_texts = self._separate_by_speaker(transcript_segments, speaker_info)

        if self.model is None:
            return self._generate_fallback_speaker_summary(speaker_texts, length)

        try:
            # 전체 요약
            overall_summary = self.summarize_text(full_text, length)

            # 화자별 요약
            speaker_summaries = {}
            for speaker_id, text in speaker_texts.items():
                if len(text) > 100:  # 최소 길이 체크
                    summary = self._generate_speaker_summary(speaker_id, text)
                    speaker_summaries[speaker_id] = summary

            processing_time = time.time() - start_time

            return SummaryResult(
                summary=overall_summary.summary,
                key_points=overall_summary.key_points,
                keywords=overall_summary.keywords,
                speaker_summaries=speaker_summaries,
                confidence=0.9,
                processing_time=processing_time
            )

        except Exception as e:
            print(f"❌ 화자별 요약 실패: {e}")
            return self._generate_fallback_speaker_summary(speaker_texts, length)

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        핵심 키워드 추출

        Args:
            text: 텍스트
            max_keywords: 최대 키워드 수

        Returns:
            키워드 리스트
        """
        if self.model is None:
            return self._extract_keywords_simple(text, max_keywords)

        try:
            prompt = f"""
            다음 텍스트에서 가장 중요한 키워드 {max_keywords}개를 추출해주세요.
            교육 콘텐츠임을 고려하여 학습에 중요한 개념 위주로 선택해주세요.

            텍스트: {text[:1000]}

            키워드만 쉼표로 구분하여 나열해주세요.
            """

            response = self.model.generate_content(prompt)
            keywords = [k.strip() for k in response.text.split(',')]

            return keywords[:max_keywords]

        except:
            return self._extract_keywords_simple(text, max_keywords)

    def generate_chapters(self, transcript_segments: List[Dict]) -> List[Dict]:
        """
        자동 챕터 생성

        Args:
            transcript_segments: 음성 인식 세그먼트

        Returns:
            챕터 정보 리스트
        """
        chapters = []
        current_chapter = None
        chapter_threshold = 60  # 60초 기준

        for i, segment in enumerate(transcript_segments):
            if current_chapter is None or segment['start'] - current_chapter['end'] > chapter_threshold:
                # 새 챕터 시작
                if current_chapter:
                    chapters.append(current_chapter)

                current_chapter = {
                    'index': len(chapters) + 1,
                    'title': self._generate_chapter_title(segment['text']),
                    'start': segment['start'],
                    'end': segment['end'],
                    'segments': [segment]
                }
            else:
                # 현재 챕터에 추가
                current_chapter['end'] = segment['end']
                current_chapter['segments'].append(segment)

        if current_chapter:
            chapters.append(current_chapter)

        return chapters

    def _create_prompt(self, text: str, settings: Dict, focus: str) -> str:
        """요약 프롬프트 생성"""
        focus_instructions = {
            "education": "교육적 가치가 있는 핵심 개념과 학습 포인트를 중심으로",
            "business": "비즈니스 인사이트와 실행 가능한 포인트를 중심으로",
            "general": "전반적인 내용을 균형있게"
        }

        return f"""
        다음 텍스트를 {settings['words']}자 이내로 요약해주세요.
        {focus_instructions.get(focus, focus_instructions['general'])} 요약해주세요.

        추가로 다음을 제공해주세요:
        1. 핵심 포인트 {settings['points']}개
        2. 중요 키워드 5-10개

        텍스트:
        {text[:3000]}  # 토큰 제한 고려

        형식:
        요약: [요약 내용]

        핵심 포인트:
        1.
        2.
        ...

        키워드: [쉼표로 구분]
        """

    def _parse_response(self, response_text: str) -> Dict:
        """Gemini 응답 파싱"""
        result = {
            'summary': '',
            'key_points': [],
            'keywords': []
        }

        lines = response_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith('요약:'):
                result['summary'] = line.replace('요약:', '').strip()
                current_section = 'summary'
            elif line.startswith('핵심 포인트:'):
                current_section = 'points'
            elif line.startswith('키워드:'):
                keywords_text = line.replace('키워드:', '').strip()
                result['keywords'] = [k.strip() for k in keywords_text.split(',')]
            elif current_section == 'points' and line and line[0].isdigit():
                point = line.split('.', 1)[1].strip() if '.' in line else line
                result['key_points'].append(point)
            elif current_section == 'summary' and line:
                result['summary'] += ' ' + line

        return result

    def _combine_segments(self, segments: List[Dict]) -> str:
        """세그먼트를 하나의 텍스트로 결합"""
        return ' '.join(seg.get('text', '') for seg in segments)

    def _separate_by_speaker(self, segments: List[Dict], speaker_info: Dict) -> Dict[str, str]:
        """화자별로 텍스트 분리"""
        speaker_texts = {}

        for segment in segments:
            speaker_id = segment.get('speaker_id', 'Unknown')
            if speaker_id not in speaker_texts:
                speaker_texts[speaker_id] = []
            speaker_texts[speaker_id].append(segment.get('text', ''))

        return {
            speaker: ' '.join(texts)
            for speaker, texts in speaker_texts.items()
        }

    def _generate_speaker_summary(self, speaker_id: str, text: str) -> str:
        """개별 화자 요약 생성"""
        if len(text) < 100:
            return text[:100]

        # 간단한 요약 (첫 2문장)
        sentences = text.split('.')[:2]
        return '. '.join(sentences) + '.'

    def _generate_chapter_title(self, text: str) -> str:
        """챕터 제목 생성"""
        # 첫 몇 단어를 제목으로
        words = text.split()[:5]
        return ' '.join(words) + '...'

    def _extract_keywords_simple(self, text: str, max_keywords: int) -> List[str]:
        """간단한 키워드 추출 (Fallback)"""
        import re
        from collections import Counter

        # 단어 추출
        words = re.findall(r'\b[가-힣]+\b|\b[a-zA-Z]+\b', text.lower())

        # 불용어 제거
        stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '도', '으로', '에서', '하다', '있다', '되다'}
        words = [w for w in words if w not in stopwords and len(w) > 1]

        # 빈도 계산
        word_counts = Counter(words)

        # 상위 N개 반환
        return [word for word, _ in word_counts.most_common(max_keywords)]

    def _generate_smart_local_summary(self, text: str, length: SummaryLength) -> SummaryResult:
        """향상된 로컬 요약 생성 (API 비용 없음)"""
        start_time = time.time()
        settings = self.LENGTH_SETTINGS[length]

        # 문장 분리
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        # 중요 문장 선택 (중요 키워드 포함)
        important_keywords = ['중요', '핵심', '필수', '기억', '정리', '요약', '결론', '첫째', '둘째', '마지막']
        scored_sentences = []

        for sent in sentences:
            score = 0
            # 위치 가중치 (처음과 끝 문장 중요)
            if sentences.index(sent) < 3:
                score += 2
            if sentences.index(sent) >= len(sentences) - 3:
                score += 1
            # 키워드 가중치
            for keyword in important_keywords:
                if keyword in sent:
                    score += 3
            # 길이 가중치 (너무 짧거나 긴 문장 제외)
            if 20 < len(sent) < 200:
                score += 1

            scored_sentences.append((sent, score))

        # 점수 기준 정렬
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # 요약 생성
        summary_sentences = []
        current_length = 0
        target_length = settings['words'] * 5  # 한글 기준 대략 변환

        for sent, score in scored_sentences:
            if current_length + len(sent) <= target_length:
                summary_sentences.append(sent)
                current_length += len(sent)
            if len(summary_sentences) >= settings['points'] // 2:
                break

        # 원래 순서대로 정렬
        summary_sentences = [s for s in sentences if s in summary_sentences]
        summary = '. '.join(summary_sentences[:5]) + '.'

        # 핵심 포인트 추출
        key_points = []
        for sent in scored_sentences[:settings['points']]:
            if sent[0] not in summary_sentences:
                key_points.append(sent[0])

        # 키워드 추출
        keywords = self._extract_keywords_simple(text, 10)

        return SummaryResult(
            summary=summary,
            key_points=key_points[:settings['points']],
            keywords=keywords,
            confidence=0.75,  # 로컬 요약이므로 신뢰도 조금 낮춤
            processing_time=time.time() - start_time
        )

    def _generate_fallback_summary(self, text: str, length: SummaryLength) -> SummaryResult:
        """Fallback 요약 생성"""
        settings = self.LENGTH_SETTINGS[length]

        # 간단한 요약 (첫 부분 추출)
        summary = text[:settings['words'] * 5]  # 대략적인 글자 수
        if len(text) > len(summary):
            summary += '...'

        # 문장 단위로 핵심 포인트 추출
        sentences = text.split('.')
        key_points = [s.strip() for s in sentences[:settings['points']] if s.strip()]

        # 키워드 추출
        keywords = self._extract_keywords_simple(text, 10)

        return SummaryResult(
            summary=summary,
            key_points=key_points,
            keywords=keywords,
            confidence=0.6,
            processing_time=0.1
        )

    def _generate_fallback_speaker_summary(self, speaker_texts: Dict, length: SummaryLength) -> SummaryResult:
        """Fallback 화자별 요약"""
        # 전체 텍스트
        full_text = ' '.join(speaker_texts.values())

        # 기본 요약
        base_summary = self._generate_fallback_summary(full_text, length)

        # 화자별 간단 요약
        speaker_summaries = {}
        for speaker_id, text in speaker_texts.items():
            if text:
                summary = text[:200] + '...' if len(text) > 200 else text
                speaker_summaries[speaker_id] = summary

        base_summary.speaker_summaries = speaker_summaries
        return base_summary

    def _generate_minimal_summary(self, text: str) -> SummaryResult:
        """최소 텍스트 요약"""
        return SummaryResult(
            summary=text,
            key_points=[text] if text else [],
            keywords=[],
            confidence=1.0,
            processing_time=0.0
        )


def test_gemini_summarizer():
    """AI 요약 모듈 테스트"""
    print("🤖 AI 요약 모듈 테스트 시작...\n")
    print("=" * 50)

    # 테스트 텍스트
    test_text = """
    안녕하세요, 오늘은 파이썬 프로그래밍의 기초에 대해 알아보겠습니다.
    파이썬은 배우기 쉽고 강력한 프로그래밍 언어로, 데이터 분석, 웹 개발, 인공지능 등
    다양한 분야에서 활용됩니다.

    첫 번째로 변수와 자료형에 대해 설명하겠습니다. 파이썬에서 변수는 값을 저장하는
    공간이며, 다양한 자료형을 지원합니다. 정수, 실수, 문자열, 리스트, 딕셔너리 등이
    있습니다.

    두 번째로 조건문과 반복문을 학습하겠습니다. if문을 사용하여 조건에 따라
    다른 코드를 실행할 수 있고, for문과 while문으로 반복 작업을 수행할 수 있습니다.

    마지막으로 함수에 대해 알아보겠습니다. 함수는 재사용 가능한 코드 블록으로,
    def 키워드를 사용하여 정의합니다. 매개변수를 받아 처리하고 결과를 반환할 수 있습니다.
    """

    # 요약기 초기화
    summarizer = GeminiSummarizer()

    # 각 길이별 요약 테스트
    test_results = []

    for length in ["very_short", "short", "medium"]:
        print(f"\n📊 {length.upper()} 요약 테스트:")
        print("-" * 30)

        start_time = time.time()
        result = summarizer.summarize_text(test_text, length=length)
        processing_time = time.time() - start_time

        print(f"✓ 요약: {result.summary[:100]}...")
        print(f"✓ 핵심 포인트 수: {len(result.key_points)}")
        print(f"✓ 키워드 수: {len(result.keywords)}")
        print(f"✓ 처리 시간: {processing_time:.2f}초")

        if result.key_points:
            print(f"✓ 첫 번째 포인트: {result.key_points[0]}")

        test_results.append({
            'length': length,
            'summary_len': len(result.summary),
            'points': len(result.key_points),
            'time': processing_time
        })

    # 키워드 추출 테스트
    print("\n" + "=" * 50)
    print("🔑 키워드 추출 테스트:")
    print("-" * 30)

    keywords = summarizer.extract_keywords(test_text, max_keywords=5)
    print(f"추출된 키워드: {', '.join(keywords)}")

    # 성능 비교
    print("\n" + "=" * 50)
    print("📈 요약 길이별 비교:")
    print("-" * 50)
    print(f"{'길이':<15} {'요약 글자수':<12} {'포인트 수':<10} {'처리 시간'}")
    print("-" * 50)

    for result in test_results:
        print(f"{result['length']:<15} {result['summary_len']:<12} "
              f"{result['points']:<10} {result['time']:.2f}초")

    print("\n✅ 테스트 완료!")
    print("\n💡 교육 콘텐츠는 'medium' 길이 요약 추천")


if __name__ == "__main__":
    test_gemini_summarizer()