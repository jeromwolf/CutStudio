# 🚀 CutStudio v5.0 설치 가이드

## 📋 사전 요구사항

### 필수 설치
- **Python 3.11+**
- **FFmpeg** (시스템에 설치 필요)
  ```bash
  # Mac
  brew install ffmpeg

  # Windows (chocolatey)
  choco install ffmpeg

  # Linux
  sudo apt install ffmpeg
  ```

## 🎯 설치 옵션 선택

### 옵션 1: 🚀 빠른 설치 (최소 기능)
> 기본 편집 기능만 필요한 경우

```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 최소 패키지 설치
pip install -r requirements_minimal.txt

# 3. 실행
streamlit run app_simple.py
```

**포함 기능:**
- ✅ YouTube 다운로드
- ✅ 침묵 구간 감지 (기본)
- ✅ 영상 편집
- ❌ 정확한 화자 구분
- ❌ 음성 인식/자막

---

### 옵션 2: 💪 전체 설치 (모든 기능)
> 모든 기능을 사용하고 싶은 경우

```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 전체 패키지 설치
pip install -r requirements_full.txt

# 3. 실행
streamlit run app_simple.py
```

**포함 기능:**
- ✅ YouTube 다운로드
- ✅ 정확한 화자 구분 (PyAnnote)
- ✅ 스마트 컷 (침묵 + 간투사)
- ✅ 음성 인식 & 자막 생성 (Whisper)
- ✅ AI 요약 (Gemini)

---

## 🔧 문제 해결

### 1. librosa 설치 오류
```bash
pip install --upgrade pip
pip install librosa --no-cache-dir
```

### 2. PyAnnote 설치 오류
```bash
# PyTorch 먼저 설치
pip install torch torchvision torchaudio
pip install pyannote.audio
```

### 3. Whisper 설치 오류
```bash
pip install -U openai-whisper
```

### 4. FFmpeg 오류
- FFmpeg가 PATH에 있는지 확인
- 터미널에서 `ffmpeg -version` 실행해보기

---

## 🏃‍♂️ 빠른 시작

```bash
# 이미 설치한 경우
source venv/bin/activate
streamlit run app_simple.py
```

브라우저에서 http://localhost:8501 접속

---

## 📦 패키지별 용도

| 패키지 | 용도 | 필수 여부 |
|--------|------|----------|
| streamlit | 웹 UI | ✅ 필수 |
| moviepy | 영상 편집 | ✅ 필수 |
| ffmpeg-python | 영상 처리 | ✅ 필수 |
| yt-dlp | YouTube 다운로드 | ✅ 필수 |
| librosa | 오디오 분석 | ✅ 필수 |
| pyannote.audio | 화자 감지 | ⭐ 권장 |
| openai-whisper | 음성 인식 | ⭐ 권장 |
| google-generativeai | AI 요약 | 선택 |

---

## 🎯 추천 설치

**일반 사용자**: `requirements_minimal.txt` (빠른 설치)
**교육 영상 편집**: `requirements_full.txt` (모든 기능)

---

## 📞 도움말

설치 중 문제가 있으면:
1. Python 버전 확인: `python --version` (3.11+ 필요)
2. FFmpeg 설치 확인: `ffmpeg -version`
3. 가상환경 활성화 확인

**Made with ❤️ by Kelly**