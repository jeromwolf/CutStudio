# 🛠️ CutStudio 기술 스택 결정

## 📊 현재 상황
- **기존**: Streamlit으로 v1-v4 개발 완료
- **경험**: Streamlit 개발 노하우 보유
- **문제**: 확장성, 성능, UX 한계 도달

## 🎯 핵심 요구사항
1. **대용량 비디오 처리** (2GB+)
2. **실시간 처리** (화자 감지, 음성 인식)
3. **반응형 UI** (드래그앤드롭, 타임라인)
4. **동시 사용자** (100명+)
5. **배포 용이성**

---

## 🔍 옵션 분석

### Option 1: Streamlit 유지 (현실적)
```python
# 장점
✅ 이미 구현된 코드 재사용 가능
✅ 빠른 프로토타이핑
✅ Python 생태계 직접 활용
✅ 배포 간단 (Streamlit Cloud)

# 단점
❌ 파일 업로드 제한 (200MB 기본)
❌ 실시간 업데이트 어려움
❌ 커스텀 UI 제한적
❌ 동시 사용자 처리 약함
```

**적합한 경우**: MVP, 빠른 검증, 소규모 사용자

### Option 2: FastAPI + React (확장성)
```python
# Backend: FastAPI
@app.post("/process-video")
async def process_video(file: UploadFile):
    # 비동기 처리
    result = await detect_speakers(file)
    return {"speakers": result}

# Frontend: React
const VideoEditor = () => {
  // 인터랙티브 타임라인
  return <Timeline segments={speakers} />
}
```

**장점**:
- ✅ 완벽한 UI 자유도
- ✅ 확장성 최고
- ✅ 실시간 WebSocket
- ✅ 마이크로서비스 가능

**단점**:
- ❌ 개발 시간 3배
- ❌ Frontend 개발자 필요
- ❌ 복잡한 아키텍처

### Option 3: Gradio (AI 특화)
```python
import gradio as gr

def process_video(video, mode):
    speakers = detect_speakers(video)
    return speakers, transcript, summary

interface = gr.Interface(
    fn=process_video,
    inputs=["video", "dropdown"],
    outputs=["json", "text", "text"]
)
```

**장점**:
- ✅ AI/ML 워크플로우 최적화
- ✅ Hugging Face 통합
- ✅ 빠른 개발

**단점**:
- ❌ UI 커스터마이징 제한
- ❌ 엔터프라이즈 기능 부족

### Option 4: Hybrid 접근 (추천) 🎯
```
Phase 1 (MVP): Streamlit
└── 빠른 출시, 기능 검증

Phase 2 (Growth): Streamlit + FastAPI
├── Streamlit: UI
└── FastAPI: Heavy Processing API

Phase 3 (Scale): Full Migration
└── FastAPI + React/Vue
```

---

## 💡 최종 추천: Streamlit + FastAPI Hybrid

### 아키텍처
```
┌─────────────────────────────────────┐
│         Streamlit Frontend          │
│    (UI, 간단한 상호작용, 미리보기)      │
└────────────┬────────────────────────┘
             │ REST API
┌────────────▼────────────────────────┐
│         FastAPI Backend             │
│   (무거운 처리, 비동기, 스케일링)        │
├─────────────────────────────────────┤
│ • Video Processing (FFmpeg)         │
│ • Speaker Detection (PyAnnote)      │
│ • Speech Recognition (Whisper)      │
│ • AI Summarization (Gemini)         │
└─────────────────────────────────────┘
```

### 구현 예시
```python
# streamlit_app.py
import streamlit as st
import requests

st.title("CutStudio v5")

video = st.file_uploader("Upload Video", type=['mp4'])

if video and st.button("Process"):
    # FastAPI로 처리 위임
    response = requests.post(
        "http://api:8000/process",
        files={"video": video}
    )
    result = response.json()

    # UI에 결과 표시
    st.success("Processing complete!")
    for speaker in result["speakers"]:
        st.write(f"Speaker {speaker['id']}: {speaker['duration']}s")

# api.py (FastAPI)
from fastapi import FastAPI, UploadFile
import asyncio

app = FastAPI()

@app.post("/process")
async def process_video(video: UploadFile):
    # 무거운 작업 비동기 처리
    tasks = [
        detect_speakers(video),
        transcribe_audio(video),
        generate_summary(video)
    ]
    results = await asyncio.gather(*tasks)

    return {
        "speakers": results[0],
        "transcript": results[1],
        "summary": results[2]
    }
```

---

## 📋 단계별 실행 계획

### Phase 1: MVP (2주)
```bash
기술: Streamlit Only
├── 기존 코드 재사용 80%
├── 3가지 핵심 기능만
├── 로컬 처리
└── 목표: 100명 베타 테스터
```

### Phase 2: API 분리 (1개월)
```bash
기술: Streamlit + FastAPI
├── 무거운 처리 → FastAPI
├── UI는 Streamlit 유지
├── 비동기 처리 도입
└── 목표: 1,000명 사용자
```

### Phase 3: 프로덕션 (3개월)
```bash
기술: FastAPI + React (선택적)
├── 필요시 React 마이그레이션
├── 또는 Streamlit 고도화
├── 클라우드 네이티브
└── 목표: 10,000명 사용자
```

---

## 🎯 즉시 실행 액션

### 1. Streamlit 최적화 (당장)
```python
# config.toml
[server]
maxUploadSize = 2000  # 2GB
maxMessageSize = 2000

[browser]
serverAddress = "0.0.0.0"
serverPort = 8501

# 세션 상태 활용
if 'processing_queue' not in st.session_state:
    st.session_state.processing_queue = []
```

### 2. FastAPI 준비 (1주차)
```python
# requirements.txt
streamlit==1.29.0
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6

# 프로젝트 구조
cutstudio/
├── frontend/
│   └── streamlit_app.py
├── backend/
│   ├── api.py
│   └── processors/
└── docker-compose.yml
```

### 3. 점진적 마이그레이션
```python
# Step 1: 무거운 기능만 API로
def process_video_heavy(video_path):
    if USE_API:
        return api_client.process(video_path)
    else:
        return local_process(video_path)

# Step 2: 성능 측정
@measure_performance
def detect_speakers(video):
    # 로컬 vs API 성능 비교
    pass
```

---

## 📊 기술 스택 비교표

| 항목 | Streamlit | FastAPI+React | Hybrid(추천) |
|------|-----------|---------------|--------------|
| 개발 속도 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| 확장성 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 성능 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 유지보수 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 비용 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **총점** | **17/25** | **17/25** | **22/25** |

---

## ✅ 결론

### 추천 전략
1. **지금**: Streamlit 그대로 사용 (기존 코드 활용)
2. **MVP 후**: FastAPI 백엔드 추가 (성능 개선)
3. **성장 후**: 필요시 React 고려 (UX 개선)

### 핵심 원칙
> "Perfect is the enemy of good"
> 완벽한 기술 스택보다 빠른 출시가 중요

### 당장 할 일
```bash
# 1. Streamlit 설정 최적화
streamlit config show > .streamlit/config.toml

# 2. FastAPI 뼈대 준비
pip install fastapi uvicorn
touch backend/api.py

# 3. 기존 코드 정리
mv *.py frontend/
mkdir backend/processors
```

**Kelly님, Streamlit으로 시작하되 FastAPI를 준비해두는 것이 최선입니다!**