# 🧹 CutStudio 프로젝트 정리 및 재시작 계획

## 📊 현재 프로젝트 상태 분석

### 1. 파일 구조 현황

#### 🔴 문제점들
1. **중복된 버전들**
   - `cutstudio_pro.py` (39,046 줄)
   - `cutstudio_pro_v4.py` (20,137 줄)
   - `legacy_apps/app.py`, `app_enhanced.py`, `app_refactored.py`
   - 같은 기능이 여러 파일에 산재

2. **일관성 없는 구조**
   - 구조 1: `core/`, `services/`, `ui/` (구버전)
   - 구조 2: `cutstudio/core/`, `cutstudio/modules/`, `cutstudio/ui/` (v4)
   - 두 구조가 혼재되어 혼란

3. **과도한 화자 감지 모듈** (7개)
   - `speaker_detector.py`
   - `advanced_speaker_detector.py`
   - `enhanced_speaker_detector.py`
   - `huggingface_speaker_detector.py`
   - `improved_speaker_detector.py`
   - `optimized_speaker_detector.py`
   - `practical_speaker_detector.py`

4. **테스트 부재**
   - `tests/` 폴더는 있으나 실제 테스트 코드 미비
   - 단위 테스트, 통합 테스트 없음

### 2. 핵심 기능 식별

#### ✅ 유지할 핵심 기능
1. **비디오 처리**: `video_editor.py` (MoviePy 기반)
2. **YouTube 다운로드**: `youtube_downloader.py`
3. **음성 인식**: `services/speech/speech_transcriber.py` (Whisper)
4. **AI 요약**: `services/summarizers/` (Gemini + Claude)
5. **화자 감지**: `UnifiedSpeakerDetector` (통합 버전)

#### ❌ 제거/정리 대상
- `legacy_apps/` 전체
- `backup_versions/` 전체
- 중복된 화자 감지 모듈들
- 사용하지 않는 임시 파일들 (`temp-audio.m4a`)
- 오래된 설정 파일들

### 3. 의존성 문제
- 너무 많은 라이브러리 (35개+)
- 일부는 충돌 가능성 (torch 버전)
- 사용하지 않는 의존성 포함

---

## 🎯 새로운 시작 전략

### Phase 1: 클린업 (즉시 실행)
```bash
# 1. 불필요한 파일 정리
- legacy_apps/ → archive로 이동
- backup_versions/ → archive로 이동
- 중복 화자 감지 모듈 → 하나로 통합
- __pycache__/ 삭제
- temp/ 파일들 정리

# 2. 구조 단순화
CutStudio/
├── src/              # 핵심 소스 코드
│   ├── core/        # 비즈니스 로직
│   ├── api/         # 외부 API 연동
│   ├── ui/          # UI 컴포넌트
│   └── utils/       # 유틸리티
├── tests/           # 테스트 코드
├── docs/            # 문서
├── config/          # 설정 파일
└── scripts/         # 스크립트
```

### Phase 2: 재설계 핵심 원칙

#### 1. **모놀리식 → 모듈화**
```python
# AS-IS: 하나의 거대한 파일
class CutStudioApp:  # 1000+ 줄

# TO-BE: 책임별 분리
class VideoProcessor:  # 비디오 처리만
class AudioAnalyzer:   # 오디오 분석만
class UIManager:       # UI 관리만
```

#### 2. **명확한 인터페이스**
```python
# 각 모듈은 명확한 인터페이스 제공
class IVideoProcessor(ABC):
    @abstractmethod
    def process(self, video_path: str) -> ProcessedVideo:
        pass
```

#### 3. **설정 중앙화**
```yaml
# config/settings.yaml
app:
  name: CutStudio
  version: 5.0

video:
  max_size: 2GB
  formats: [mp4, avi, mov]

ai:
  whisper_model: base
  gemini_api_key: ${GEMINI_API_KEY}
```

### Phase 3: MVP 재구축

#### 🎯 MVP 핵심 기능 (v5.0)
1. **동영상 업로드/다운로드**
2. **기본 편집** (자르기, 트림)
3. **화자 감지** (1개 통합 모듈)
4. **음성 인식** (Whisper)
5. **AI 요약** (Gemini만)

#### 🚫 MVP에서 제외
- 복잡한 타임라인 에디터
- 플러그인 시스템
- 실시간 협업
- 다중 AI 모델

---

## 📐 벤치마킹 분석

### 경쟁 제품 비교

| 제품 | 강점 | 약점 | 시사점 |
|------|------|------|--------|
| **Descript** | - 텍스트 기반 편집<br>- 직관적 UI | - 비싼 가격<br>- 한국어 지원 약함 | 텍스트 기반 편집 UI 참고 |
| **Kapwing** | - 웹 기반<br>- 협업 기능 | - 성능 제한<br>- 워터마크 | 심플한 웹 UI 참고 |
| **Runway ML** | - AI 기능 강력<br>- 다양한 모델 | - 복잡한 인터페이스<br>- 학습곡선 | AI 통합 방식 참고 |
| **Adobe Premiere** | - 전문가 기능<br>- 안정성 | - 높은 진입장벽<br>- 데스크톱 전용 | 타임라인 UI 참고 |

### CutStudio 차별화 전략
1. **교육 영상 특화** - 화자 중심 분석
2. **한국어 최적화** - 로컬 AI 모델
3. **원클릭 자동화** - 복잡한 작업 단순화
4. **무료 오픈소스** - 커뮤니티 기반

---

## 🚀 실행 계획

### 즉시 실행 (오늘)
```bash
# 1. 정리
mv legacy_apps archive/
mv backup_versions archive/
rm -rf __pycache__
rm temp-audio.m4a

# 2. 새 구조 생성
mkdir -p src/{core,api,ui,utils}
mkdir -p tests/{unit,integration}
mkdir -p config
mkdir -p docs/{api,user,dev}
```

### Week 1: 기초 재구축
- [ ] 프로젝트 구조 재설계
- [ ] 핵심 모듈만 이전
- [ ] 테스트 환경 구축
- [ ] CI/CD 파이프라인

### Week 2: MVP 개발
- [ ] 기본 UI (Streamlit)
- [ ] 비디오 업로드/처리
- [ ] 화자 감지 통합
- [ ] 음성 인식

### Week 3: 테스트 및 최적화
- [ ] 단위 테스트 작성
- [ ] 성능 최적화
- [ ] 문서화
- [ ] 배포 준비

---

## 💡 권장사항

### 1. **기술 스택 단순화**
```txt
AS-IS: Streamlit + MoviePy + 7개 AI 모델 + 35개 라이브러리
TO-BE: FastAPI + FFmpeg + Whisper + Gemini + 최소 의존성
```

### 2. **테스트 우선 개발**
- TDD 방식 도입
- 코드 커버리지 80% 목표
- E2E 테스트 자동화

### 3. **문서화 강화**
- API 문서 자동 생성 (Swagger)
- 사용자 가이드
- 개발자 문서

### 4. **성능 목표**
- 10분 영상 처리: 3분 이내
- 메모리 사용: 1GB 이하
- 동시 사용자: 10명+

---

## 🎬 결론

CutStudio는 좋은 아이디어와 기능을 가지고 있지만, 구조적 문제로 인해 유지보수와 확장이 어려운 상태입니다.

**핵심 제안**:
1. 기존 코드는 참고용으로만 활용
2. 깨끗한 구조로 처음부터 재구축
3. MVP 먼저 완성 후 점진적 기능 추가
4. 테스트와 문서화를 처음부터 포함

이렇게 하면 더 안정적이고 확장 가능한 CutStudio v5.0을 만들 수 있을 것입니다.