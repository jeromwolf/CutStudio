# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

CutStudio는 Streamlit을 기반으로 한 웹 동영상 편집기입니다. MoviePy를 사용하여 동영상 편집 기능을 구현하고, Streamlit으로 사용자 인터페이스를 제공합니다.

## 개발 명령어

### 애플리케이션 실행
```bash
streamlit run app.py
```

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 가상환경 활성화
```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

## 코드 구조

### 핵심 컴포넌트
- `app.py`: Streamlit UI 및 사용자 상호작용 처리
- `video_editor.py`: VideoEditor 클래스 - MoviePy를 사용한 동영상 편집 로직
- `utils.py`: 동영상 정보 추출, 시간 포맷팅 등 유틸리티 함수

### 주요 기능 구현 위치
- 동영상 자르기: `video_editor.py:cut_video()`
- 동영상 트림: `video_editor.py:trim_video()`
- 효과 적용: `video_editor.py:apply_*()` 메서드들
- 파일 업로드 처리: `app.py:22-32`

### 디렉토리 구조
- `temp/`: 업로드된 원본 동영상 임시 저장
- `processed/`: 편집된 동영상 출력 파일 저장

## 기술 스택
- **Frontend**: Streamlit (동영상 플레이어, 파일 업로드/다운로드)
- **Backend**: MoviePy (동영상 편집), OpenCV (썸네일 생성)
- **파일 형식**: MP4, AVI, MOV, MKV 지원

---

# 📝 최신 개발 현황 (2025-07-29)

## 🚀 v3.0 주요 업데이트 완료

### ✅ 구현된 새로운 기능들

#### 1. 음성 인식 시스템 (speech_transcriber.py)
- **OpenAI Whisper 모델 통합**: 세계급 음성 인식 정확도
- **SpeechRecognizer 클래스**: Whisper 모델 래핑 및 최적화
- **AdvancedSpeechAnalyzer 클래스**: 화자 구분과 음성 인식 통합
- **다국어 지원**: 100개 이상 언어 자동 감지
- **시간 동기화**: 정확한 구간별 타임스탬프 제공

#### 2. AI 요약 시스템 (gemini_summarizer.py)
- **Google Gemini 1.5-flash 모델**: 최신 생성형 AI 활용
- **GeminiSummarizer 클래스**: 종합적인 텍스트 분석 도구
- **화자별 요약**: 각 화자의 주요 발언 내용 정리
- **전체 대화 요약**: 대화 전체의 핵심 내용 추출
- **키워드 추출**: 중요한 키워드 자동 식별
- **길이 제한 및 실패 대응**: 안정적인 요약 생성

#### 3. 스마트 타임라인 UI
- **화자별 썸네일**: 각 발화 구간의 대표 이미지 자동 생성
- **시간순 정렬**: 전체 대화 흐름을 직관적으로 표시
- **음성 인식 연동**: 썸네일과 텍스트 내용 매핑
- **AI 요약 통합**: 각 구간별 Gemini 요약 표시
- **카드형 레이아웃**: 4개씩 그리드 형태로 배열

### 🔧 해결된 기술적 이슈들

#### 1. 중첩 Expander 오류 해결
- **문제**: "Expanders may not be nested inside other expanders" 에러
- **해결**: 메인 expander 외부로 중첩된 expander들 이동
- **위치**: app.py:1075-1200 라인 대폭 수정

#### 2. 구간재생 기능 제거
- **문제**: JavaScript iframe 통신 및 메모리 과부하 이슈
- **해결**: 복잡한 구간재생 기능 완전 제거
- **결과**: UI 단순화 및 안정성 크게 향상

#### 3. 모듈 분리 및 코드 최적화
- **speech_recognition.py → speech_transcriber.py**: 라이브러리 충돌 해결
- **gemini-pro → gemini-1.5-flash**: 모델 업데이트로 안정성 향상
- **요약 임계값 조정**: 100자 → 50자로 낮춰 더 많은 구간 요약

### 📁 주요 파일 변경사항

#### app.py (메인 애플리케이션)
- **825-909 라인**: 메인 플레이어 추가 → 제거 (구간재생 기능 삭제)
- **1070-1200 라인**: 중첩 expander 구조 해결
- **자동 음성 인식**: 화자 감지 완료 후 자동 실행
- **UI 최적화**: 깔끔한 타임라인 중심 인터페이스

#### speech_transcriber.py (신규)
- **SpeechRecognizer**: Whisper 모델 래핑
- **AdvancedSpeechAnalyzer**: 화자 구분과 음성 인식 통합
- **에러 처리**: 안정적인 음성 인식 파이프라인

#### gemini_summarizer.py (신규)
- **GeminiSummarizer**: 종합 AI 분석 도구
- **summarize_conversation()**: 화자별 요약 (회의 분석 아닌 화자 중심)
- **extract_keywords()**: 키워드 추출 기능
- **_simple_summary()**: API 실패 시 대체 요약

### 🎯 사용자 피드백 반영사항

1. **"화자별 분석이지 회의 분석은 아니야"**
   - Gemini 요약을 개별 화자 중심으로 변경
   - generate_meeting_summary() 사용 중단

2. **"구간재생은 잘 안되는 것 같아"**
   - 복잡한 JavaScript 구간재생 기능 완전 제거
   - 단순하고 안정적인 타임라인 중심 UI로 변경

3. **"타이머가 있으면 좋을 것 같아"**
   - 실시간 타이머는 기술적 제약으로 시작/예상종료 시간 표시로 대체

### 🛠 환경 설정

#### 필수 API 키
```bash
# .env 파일 설정
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACE_TOKEN=hf_your_token_here
```

#### 주요 의존성
- openai-whisper: 음성 인식
- google-generativeai: AI 요약
- streamlit: 웹 인터페이스
- moviepy: 동영상 처리

### 📊 성능 특성

#### 처리 시간 (10분 영상 기준)
- **화자 감지**: 1-6분 (방법에 따라)
- **음성 인식**: 2-3분 (Whisper)
- **AI 요약**: 30초-1분 (Gemini)

#### 메모리 사용량
- **기본 기능**: 1-2GB
- **허깅페이스 AI**: 4-6GB
- **음성 인식**: 추가 1-2GB

### 🚀 다음 개발 방향

#### v3.1 계획
- [ ] 구간별 비디오 재생 기능 재구현 (단순화된 방식)
- [ ] 실시간 처리 상태 표시 개선
- [ ] 배치 처리 기능 (여러 파일 동시 처리)
- [ ] 화자 이름 수동 지정 기능

#### 알려진 제한사항
- 대용량 파일(50MB+)에서 메모리 이슈 가능
- JavaScript 구간재생 기능 제거됨
- Streamlit 특성상 실시간 타이머 구현 어려움

### 💡 개발자 노트

#### 코드 품질
- PEP 8 준수
- 모듈화된 구조
- 포괄적인 에러 처리
- 사용자 친화적 UI

#### 테스트 상태
- 기본 기능: ✅ 안정적
- 음성 인식: ✅ 높은 정확도
- AI 요약: ✅ 품질 우수
- 타임라인 UI: ✅ 직관적

---

# 🎯 다음 세션 시작 가이드

## 즉시 확인 사항
1. **앱 상태**: `streamlit run app.py`로 정상 작동 확인
2. **환경 변수**: .env 파일의 API 키 설정 상태
3. **최신 커밋**: GitHub 동기화 상태 점검

## 자주 사용하는 명령어
```bash
# 앱 실행
streamlit run app.py

# Git 상태 확인
git status
git log --oneline -5

# 의존성 확인
pip list | grep -E "(streamlit|moviepy|whisper|google-generativeai)"
```

## 주요 파일 위치
- 메인 앱: `app.py`
- 음성 인식: `speech_transcriber.py`
- AI 요약: `gemini_summarizer.py`
- 프로젝트 문서: `README.md`
- 스크린샷: `screenshots/` (5개 파일)

## 최근 GitHub 커밋
- `feat: AI 기반 음성 인식 및 요약 기능 추가 - v3.0 업데이트`
- `docs: CutStudio v3.0 스크린샷 추가`

**다음 세션에서 바로 개발을 계속할 수 있도록 모든 컨텍스트가 저장되었습니다.** 🚀