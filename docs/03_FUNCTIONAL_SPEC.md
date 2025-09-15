# 🔧 기능 설계서 (Functional Specification)
## CutStudio v5.0

### 문서 정보
- **버전**: 1.0
- **작성일**: 2025.09.13
- **작성자**: Kelly
- **검토자**: Dev Team
- **승인자**: Tech Lead

---

## 1. 시스템 아키텍처

### 1.1 전체 구조도
```
┌──────────────────────────────────────────────────┐
│                  Frontend (UI Layer)              │
│                  Streamlit v1.29.0                │
├──────────────────────────────────────────────────┤
│                 Business Logic Layer              │
│  ┌────────────┬──────────────┬────────────────┐ │
│  │  Speaker   │  Smart Cut   │  Transcriber   │ │
│  │  Detector  │   Module     │    Module      │ │
│  └────────────┴──────────────┴────────────────┘ │
├──────────────────────────────────────────────────┤
│                   AI/ML Layer                     │
│  ┌────────────┬──────────────┬────────────────┐ │
│  │  PyAnnote  │   Librosa    │    Whisper     │ │
│  │            │              │    Gemini      │ │
│  └────────────┴──────────────┴────────────────┘ │
├──────────────────────────────────────────────────┤
│                 Infrastructure Layer              │
│  ┌────────────┬──────────────┬────────────────┐ │
│  │   FFmpeg   │  File System │   Database     │ │
│  │            │              │   (SQLite)     │ │
│  └────────────┴──────────────┴────────────────┘ │
└──────────────────────────────────────────────────┘
```

### 1.2 데이터 흐름도
```
[비디오 업로드] → [임시 저장] → [오디오 추출]
                                      ↓
[화자 감지] ← [음성 특징 추출] ← [오디오 전처리]
     ↓
[스마트 컷] → [침묵 감지] → [구간 제거]
     ↓
[음성 인식] → [텍스트 변환] → [자막 생성]
     ↓
[AI 요약] → [키워드 추출] → [요약 생성]
     ↓
[결과 통합] → [비디오 재인코딩] → [다운로드]
```

---

## 2. 모듈별 상세 설계

### 2.1 비디오 업로드 모듈

#### 2.1.1 기능 정의
```python
class VideoUploader:
    """비디오 업로드 및 검증"""

    def validate_file(self, file) -> bool:
        """
        파일 검증
        - 형식: MP4, AVI, MOV, MKV
        - 크기: 최대 2GB
        - 코덱: H.264, H.265
        """

    def save_temp_file(self, file) -> str:
        """
        임시 파일 저장
        - 경로: /tmp/cutstudio/uploads/
        - 명명: UUID_timestamp.ext
        - 권한: 644
        """

    def extract_metadata(self, file_path) -> dict:
        """
        메타데이터 추출
        - duration: 영상 길이 (초)
        - resolution: 해상도 (width, height)
        - fps: 프레임레이트
        - codec: 비디오/오디오 코덱
        - size: 파일 크기 (bytes)
        """
```

#### 2.1.2 입출력 명세
| 항목 | 타입 | 설명 | 예시 |
|------|------|------|------|
| **입력** |
| file | UploadedFile | Streamlit 업로드 객체 | video.mp4 |
| **출력** |
| file_path | str | 저장된 파일 경로 | /tmp/cutstudio/abc123.mp4 |
| metadata | dict | 비디오 메타데이터 | {duration: 120, ...} |
| **예외** |
| InvalidFormatError | Exception | 지원하지 않는 형식 | "Not supported: .wmv" |
| FileSizeError | Exception | 파일 크기 초과 | "File > 2GB" |

---

### 2.2 화자 감지 모듈

#### 2.2.1 기능 정의
```python
class SpeakerDetector:
    """화자 감지 및 분리"""

    def __init__(self, model_type="pyannote"):
        self.model = self.load_model(model_type)
        self.max_speakers = 10

    def detect_speakers(self, audio_path: str) -> List[Speaker]:
        """
        화자 감지 실행

        알고리즘:
        1. 오디오 로드 (16kHz 리샘플링)
        2. VAD로 음성 구간 추출
        3. 음성 특징 추출 (MFCC, embeddings)
        4. 클러스터링 (HDBSCAN)
        5. 화자 레이블 할당

        Returns:
            List[Speaker]: 화자 리스트
        """

    def merge_segments(self, segments, threshold=0.5):
        """
        인접 세그먼트 병합
        - threshold: 병합 임계값 (초)
        """

    def calculate_overlap(self, speakers):
        """
        화자 겹침 구간 계산
        """
```

#### 2.2.2 알고리즘 상세
```
1. 전처리
   - 16kHz 리샘플링
   - 노이즈 제거 (Wiener filter)
   - 정규화 (-23 LUFS)

2. VAD (Voice Activity Detection)
   - WebRTC VAD 사용
   - 프레임: 30ms
   - 임계값: -35dB

3. 특징 추출
   - MFCC: 13 coefficients
   - Embeddings: 512 dimensions
   - Window: 25ms, Hop: 10ms

4. 클러스터링
   - HDBSCAN
   - min_cluster_size: 15
   - min_samples: 5
   - metric: cosine

5. 후처리
   - 0.5초 이하 세그먼트 제거
   - 인접 세그먼트 병합
   - 화자 레이블 정규화
```

#### 2.2.3 성능 명세
| 지표 | 목표 | 측정 방법 |
|------|------|-----------|
| DER (Diarization Error Rate) | < 15% | NIST RT-09 |
| 처리 시간 | < 0.3 × 영상 길이 | Wall clock time |
| 메모리 사용 | < 2GB | Peak RSS |
| 정확도 | > 85% | F1 Score |

---

### 2.3 스마트 컷 모듈

#### 2.3.1 기능 정의
```python
class SmartCutter:
    """침묵 및 불필요 구간 제거"""

    def __init__(self, mode="balanced"):
        self.thresholds = self.get_thresholds(mode)

    def detect_silence(self, audio) -> List[Segment]:
        """
        침묵 구간 감지

        알고리즘:
        1. RMS 에너지 계산
        2. dB 변환 (20 * log10(rms))
        3. 임계값 이하 구간 감지
        4. 최소 지속 시간 필터링
        """

    def detect_fillers(self, transcript) -> List[Segment]:
        """
        간투사 감지
        패턴: ["음", "어", "그", "아", "에", "um", "uh"]
        """

    def remove_segments(self, video_path, segments):
        """
        구간 제거 및 재인코딩
        - Codec: libx264
        - Quality: CRF 23
        - Audio: AAC 128kbps
        """
```

#### 2.3.2 처리 모드별 파라미터
| 모드 | 침묵 임계값 | 최소 지속 시간 | 간투사 제거 | 예상 단축률 |
|------|-------------|---------------|------------|------------|
| **Conservative** | -40dB | 3.0초 | 미포함 | 20-30% |
| **Balanced** | -35dB | 2.0초 | 기본 패턴 | 30-40% |
| **Aggressive** | -30dB | 1.0초 | 확장 패턴 | 40-50% |

#### 2.3.3 침묵 감지 알고리즘
```python
def detect_silence_detailed(audio, sr=16000):
    # 1. 프레임 단위 에너지 계산
    hop_length = int(0.01 * sr)  # 10ms
    frame_length = int(0.025 * sr)  # 25ms

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # 2. dB 변환
    rms_db = 20 * np.log10(rms + 1e-10)

    # 3. 이동 평균 스무딩
    window_size = 10
    rms_smooth = np.convolve(
        rms_db,
        np.ones(window_size)/window_size,
        mode='same'
    )

    # 4. 임계값 적용
    threshold = -35  # dB
    silence_mask = rms_smooth < threshold

    # 5. 연속 구간 추출
    segments = []
    start = None

    for i, is_silent in enumerate(silence_mask):
        time = i * hop_length / sr

        if is_silent and start is None:
            start = time
        elif not is_silent and start is not None:
            if time - start >= 2.0:  # 최소 2초
                segments.append((start, time))
            start = None

    return segments
```

---

### 2.4 음성 인식 모듈

#### 2.4.1 기능 정의
```python
class Transcriber:
    """음성 인식 및 자막 생성"""

    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path, language="auto"):
        """
        음성 인식 실행

        처리 과정:
        1. 오디오 로드 (16kHz)
        2. 30초 청크로 분할
        3. 각 청크 전사
        4. 타임스탬프 정렬
        5. 후처리 (구두점, 대문자)
        """

    def generate_subtitles(self, segments):
        """
        자막 파일 생성
        - SRT: SubRip 형식
        - VTT: WebVTT 형식
        - 최대 2줄, 42자/줄
        """

    def align_with_speakers(self, transcript, speakers):
        """
        화자별 텍스트 할당
        """
```

#### 2.4.2 Whisper 모델 사양
| 모델 | 파라미터 | 크기 | VRAM | 속도 | WER (한국어) |
|------|---------|------|------|------|-------------|
| tiny | 39M | 75MB | 1GB | 32x | 15% |
| **base** | **74M** | **145MB** | **1GB** | **16x** | **10%** |
| small | 244M | 483MB | 2GB | 6x | 7% |
| medium | 769M | 1.5GB | 5GB | 2x | 5% |

#### 2.4.3 자막 생성 규칙
```
1. 타이밍
   - 최소 표시 시간: 1초
   - 최대 표시 시간: 7초
   - 씬 전환 시 분할

2. 텍스트
   - 최대 2줄
   - 줄당 42자 (한글 21자)
   - 문장 중간 분할 금지

3. 형식
   - 숫자: 아라비아 숫자
   - 외래어: 원어 병기
   - 전문용어: 첫 출현 시 설명

예시 (SRT):
1
00:00:00,000 --> 00:00:03,500
안녕하세요. 오늘은 CutStudio에 대해
소개해드리겠습니다.

2
00:00:03,500 --> 00:00:07,000
CutStudio는 AI 기반의
교육 영상 편집 도구입니다.
```

---

### 2.5 AI 요약 모듈

#### 2.5.1 기능 정의
```python
class Summarizer:
    """AI 기반 요약 생성"""

    def __init__(self):
        self.gemini = genai.GenerativeModel('gemini-pro')

    def summarize(self, text, max_length=500):
        """
        텍스트 요약

        프롬프트 전략:
        1. 컨텍스트 제공 (교육 영상)
        2. 요약 길이 명시
        3. 핵심 포인트 추출 요청
        4. 형식 지정 (불릿 포인트)
        """

    def extract_keywords(self, text, top_n=10):
        """
        키워드 추출
        - TF-IDF 기반
        - 명사 우선
        - 복합어 처리
        """

    def generate_chapters(self, transcript, speakers):
        """
        자동 챕터 생성
        - 주제 변화 감지
        - 화자 전환 고려
        - 5-10개 챕터
        """
```

#### 2.5.2 요약 생성 프롬프트
```python
SUMMARY_PROMPT = """
당신은 교육 콘텐츠 전문가입니다.
다음 강의 내용을 {max_words}자 이내로 요약해주세요.

요구사항:
1. 핵심 개념과 학습 목표 포함
2. 시간 순서대로 정리
3. 전문 용어는 쉽게 설명
4. 불릿 포인트 5개 이하

강의 내용:
{transcript}

형식:
## 요약
(요약 내용)

## 핵심 포인트
• 포인트 1
• 포인트 2
...

## 키워드
키워드1, 키워드2, ...
"""
```

---

## 3. API 명세

### 3.1 내부 API

#### 3.1.1 비디오 처리 API
```python
# POST /api/process
{
    "video_path": "/tmp/video.mp4",
    "options": {
        "detect_speakers": true,
        "max_speakers": 3,
        "smart_cut": true,
        "cut_mode": "balanced",
        "transcribe": true,
        "language": "auto",
        "summarize": true
    }
}

# Response
{
    "status": "success",
    "job_id": "abc123",
    "results": {
        "speakers": [...],
        "segments_removed": [...],
        "transcript": {...},
        "summary": {...},
        "output_path": "/tmp/output.mp4"
    },
    "processing_time": 180.5
}
```

#### 3.1.2 상태 조회 API
```python
# GET /api/status/{job_id}

# Response
{
    "job_id": "abc123",
    "status": "processing",  # queued|processing|completed|failed
    "progress": 65,
    "current_step": "transcribing",
    "eta": 120  # seconds
}
```

---

## 4. 데이터베이스 스키마

### 4.1 테이블 정의

#### 4.1.1 jobs 테이블
```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    video_path TEXT NOT NULL,
    status TEXT DEFAULT 'queued',
    options JSON,
    results JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
```

#### 4.1.2 videos 테이블
```sql
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    job_id TEXT REFERENCES jobs(id),
    original_path TEXT,
    processed_path TEXT,
    duration REAL,
    size_before INTEGER,
    size_after INTEGER,
    metadata JSON
);
```

#### 4.1.3 speakers 테이블
```sql
CREATE TABLE speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT REFERENCES jobs(id),
    speaker_id TEXT,
    name TEXT,
    total_duration REAL,
    segment_count INTEGER,
    segments JSON
);
```

---

## 5. 에러 처리

### 5.1 에러 코드 정의
| 코드 | 이름 | 설명 | HTTP Status |
|------|------|------|------------|
| E001 | INVALID_FORMAT | 지원하지 않는 파일 형식 | 400 |
| E002 | FILE_TOO_LARGE | 파일 크기 초과 | 413 |
| E003 | PROCESSING_FAILED | 처리 중 오류 | 500 |
| E004 | MODEL_LOAD_ERROR | AI 모델 로드 실패 | 503 |
| E005 | TIMEOUT | 처리 시간 초과 | 504 |
| E006 | QUOTA_EXCEEDED | API 할당량 초과 | 429 |

### 5.2 에러 처리 전략
```python
try:
    result = process_video(video_path)
except InvalidFormatError as e:
    return {"error": "E001", "message": str(e)}, 400
except ProcessingError as e:
    # 재시도 로직
    retry_count = 3
    for i in range(retry_count):
        try:
            result = process_video(video_path)
            break
        except:
            if i == retry_count - 1:
                raise
            time.sleep(2 ** i)  # Exponential backoff
```

---

## 6. 성능 최적화

### 6.1 캐싱 전략
```python
# 모델 캐싱
MODEL_CACHE = {}

def get_model(model_type):
    if model_type not in MODEL_CACHE:
        MODEL_CACHE[model_type] = load_model(model_type)
    return MODEL_CACHE[model_type]

# 결과 캐싱 (Redis)
def cache_result(job_id, result, ttl=3600):
    redis_client.setex(
        f"result:{job_id}",
        ttl,
        json.dumps(result)
    )
```

### 6.2 병렬 처리
```python
from concurrent.futures import ThreadPoolExecutor

def process_parallel(video_path):
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 병렬 실행
        future_speaker = executor.submit(detect_speakers, video_path)
        future_transcript = executor.submit(transcribe, video_path)
        future_silence = executor.submit(detect_silence, video_path)

        # 결과 수집
        speakers = future_speaker.result()
        transcript = future_transcript.result()
        silence = future_silence.result()

    return combine_results(speakers, transcript, silence)
```

---

## 7. 테스트 명세

### 7.1 단위 테스트
```python
def test_speaker_detection():
    """화자 감지 정확도 테스트"""
    audio_path = "test_data/3_speakers.wav"
    detector = SpeakerDetector()
    speakers = detector.detect_speakers(audio_path)

    assert len(speakers) == 3
    assert speakers[0].duration > 0
    assert sum(s.duration for s in speakers) <= audio_duration

def test_silence_detection():
    """침묵 감지 테스트"""
    # 10초 오디오: 3초 음성, 4초 침묵, 3초 음성
    audio = generate_test_audio()
    cutter = SmartCutter()
    segments = cutter.detect_silence(audio)

    assert len(segments) == 1
    assert segments[0] == (3.0, 7.0)
```

### 7.2 통합 테스트
```python
def test_end_to_end_processing():
    """전체 워크플로우 테스트"""
    video_path = "test_data/sample_lecture.mp4"

    result = process_video(video_path, {
        "detect_speakers": True,
        "smart_cut": True,
        "transcribe": True
    })

    assert result["status"] == "success"
    assert len(result["speakers"]) >= 1
    assert result["transcript"]["text"] != ""
    assert os.path.exists(result["output_path"])
```

---

## 8. 배포 명세

### 8.1 Docker 컨테이너
```dockerfile
FROM python:3.11-slim

# FFmpeg 설치
RUN apt-get update && apt-get install -y ffmpeg

# Python 의존성
COPY requirements.txt .
RUN pip install -r requirements.txt

# 앱 코드
COPY . /app
WORKDIR /app

# Whisper 모델 사전 다운로드
RUN python -c "import whisper; whisper.load_model('base')"

# 실행
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

### 8.2 환경 변수
```bash
# .env
GEMINI_API_KEY=xxx
TEMP_DIR=/tmp/cutstudio
MAX_UPLOAD_SIZE=2000
MODEL_CACHE_DIR=/models
LOG_LEVEL=INFO
```

---

## 9. 모니터링

### 9.1 로깅
```python
import logging

logging.config.dictConfig({
    'version': 1,
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'cutstudio.log',
            'formatter': 'detailed'
        }
    },
    'formatters': {
        'detailed': {
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file']
    }
})
```

### 9.2 메트릭스
| 메트릭 | 임계값 | 알림 조건 |
|--------|--------|----------|
| CPU 사용률 | 80% | 5분 이상 지속 |
| 메모리 사용률 | 90% | 즉시 |
| 처리 시간 | 영상 길이 × 0.5 | 초과 시 |
| 에러율 | 5% | 10분 평균 |

---

**작성일**: 2025.09.13
**다음 리뷰**: 2025.09.20