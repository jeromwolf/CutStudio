# 🎬 CutStudio v5.0 - MVP Edition

**교육 영상 편집 시간을 10시간에서 15분으로!**

## 🚀 Quick Start

```bash
# 1. 클론
git clone https://github.com/yourusername/cutstudio.git
cd cutstudio

# 2. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 API 키 입력

# 5. 실행
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속

## 🎯 3가지 핵심 기능

### 1️⃣ 화자 구분
- 최대 10명까지 자동 구분
- 화자별 타임라인 생성
- 특정 화자만 추출 가능

### 2️⃣ 스마트 컷
- 3초 이상 침묵 자동 제거
- "음...", "어..." 같은 간투사 제거
- 영상 길이 50% 단축

### 3️⃣ 음성→텍스트
- 95% 정확도 음성 인식
- SRT 자막 파일 생성
- AI 자동 요약

## 📊 성능

| 작업 | 기존 | CutStudio | 단축률 |
|------|------|-----------|--------|
| 화자 구분 | 2시간 | 5분 | 96% |
| 무음 제거 | 3시간 | 자동 | 100% |
| 자막 생성 | 5시간 | 10분 | 97% |
| **전체** | **10시간** | **15분** | **97.5%** |

## 🛠️ 기술 스택

- **Frontend**: Streamlit 1.29.0
- **Video**: MoviePy, FFmpeg
- **AI**: Whisper (음성인식), PyAnnote (화자감지), Gemini (요약)
- **Python**: 3.11+

## 📋 시스템 요구사항

- Python 3.11 이상
- 8GB RAM (권장 16GB)
- FFmpeg 설치 필수
- 인터넷 연결 (모델 다운로드)

## 🔑 API 키 설정

`.env` 파일에 다음 키 설정:

```env
# Google Gemini (필수)
GEMINI_API_KEY=your_key_here

# Hugging Face (선택)
HUGGINGFACE_TOKEN=your_token_here
```

## 📁 프로젝트 구조

```
cutstudio/
├── app.py                 # Streamlit 메인 앱
├── modules/
│   ├── speaker_detector.py  # 화자 감지
│   ├── smart_cutter.py      # 스마트 컷
│   └── transcriber.py       # 음성 인식 & 요약
├── .streamlit/
│   └── config.toml          # Streamlit 설정
└── requirements.txt         # 의존성
```

## 🐛 Known Issues

- 첫 실행 시 모델 다운로드로 시간 소요 (약 5분)
- 2GB 이상 파일은 메모리 부족 가능
- Windows에서 FFmpeg 경로 설정 필요

## 📞 Support

- Issues: [GitHub Issues](https://github.com/yourusername/cutstudio/issues)
- Email: contact@cutstudio.ai
- Discord: [Join Community](https://discord.gg/cutstudio)

## 📄 License

MIT License

---

**Made with ❤️ for Educators by Kelly**