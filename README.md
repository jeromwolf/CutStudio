# CutStudio - AI 기반 동영상 편집기

CutStudio는 **최신 AI 화자 인식 기능**을 갖춘 웹 기반 동영상 편집 도구입니다. YouTube 동영상 다운로드, 화자별 구간 자동 분리, 다양한 편집 기능을 제공합니다.

## 🆕 최신 업데이트 (v2.0)

- ✨ **허깅페이스 AI 화자분리** 추가 - Pyannote 3.1 모델 지원
- 🎯 **5가지 화자감지 방법** - 용도에 맞는 최적 선택
- ⚡ **성능 최적화** - 실용적 방법으로 빠른 처리
- 📊 **진행상황 표시** - 예상 시간 및 진행률 표시

## 🚀 주요 기능

### 📁 동영상 업로드 및 편집
- 다양한 형식 지원 (MP4, AVI, MOV, MKV)
- 실시간 미리보기
- 직관적인 편집 인터페이스

### 📺 YouTube 다운로드
- YouTube URL로 직접 다운로드
- 다양한 해상도 선택 가능
- 오디오만 추출 (MP3)
- 다운로드 진행률 표시

### ✂️ 편집 기능
- **자르기**: 특정 구간 추출
- **트림**: 앞뒤 불필요한 부분 제거
- **효과 적용**:
  - 흑백 변환
  - 페이드 인/아웃
  - 재생 속도 조절 (0.5x ~ 2.0x)

### 👥 AI 화자 분리 ⭐ **NEW**
- **자동 화자 감지**: 동영상에서 여러 화자 자동 구분
- **5가지 감지 방법**:
  - **허깅페이스 AI (최신)**: Pyannote 3.1 모델 사용, 최고 정확도 ⭐
  - **실용적 (권장)**: 속도와 정확도의 균형 (1-2분)
  - **고급 (향상된 특징 + 스펙트럴)**: 높은 정확도 (5-10분)
  - **자동 (MFCC + 클러스터링)**: 기본 성능
  - **간단 (에너지 기반)**: 빠르지만 정확도 낮음
- **화자별 동영상 분리**: 각 화자의 구간만 추출
- **수동 편집**: 화자 레이블 수정 가능
- **신뢰도 표시**: 각 구간의 화자 감지 신뢰도 확인

## 💻 설치 방법

### 🔧 시스템 요구사항
- **Python 3.8+**
- **FFmpeg** (필수)
- **8GB+ RAM** (허깅페이스 AI 사용 시)
- **인터넷 연결** (첫 실행 시 모델 다운로드)

### 📋 빠른 시작

#### 1. 저장소 클론
```bash
git clone https://github.com/jeromwolf/CutStudio.git
cd CutStudio
```

#### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\\Scripts\\activate
```

#### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

#### 4. FFmpeg 설치 (필수)
- **Mac**: `brew install ffmpeg`
- **Windows**: [FFmpeg 공식 사이트](https://ffmpeg.org/download.html)에서 다운로드
- **Linux**: `sudo apt-get install ffmpeg`

#### 5. 앱 실행
```bash
streamlit run app.py
```

브라우저에서 자동으로 열리거나 http://localhost:8501 접속

---

## 🤗 허깅페이스 AI 화자분리 설정 (선택사항)

### ⚠️ 중요: 필수 설정

허깅페이스 AI 화자분리 기능을 사용하려면 다음 단계가 **반드시** 필요합니다:

#### 1. 허깅페이스 계정 생성 및 토큰 발급
1. [Hugging Face](https://huggingface.co/) 가입
2. [토큰 생성 페이지](https://huggingface.co/settings/tokens) 접속
3. **"New token"** 클릭
4. Token type: **"Read"** 선택
5. 토큰 복사 (한 번만 표시됨!)

#### 2. 모델 사용 조건 동의 (필수!)
다음 **3개 모델** 페이지에 각각 접속하여 사용 조건에 동의해야 합니다:

1. 🔗 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. 🔗 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. 🔗 [pyannote/embedding](https://huggingface.co/pyannote/embedding)

각 페이지에서:
- 로그인하기
- **"Accept license"** 또는 **"Agree and access repository"** 클릭
- 약관 읽고 동의

#### 3. 토큰 설정
**방법 1: .env 파일 (권장)**
```bash
# .env 파일 생성
cp .env.example .env

# .env 파일을 열고 토큰 입력
# HUGGINGFACE_TOKEN=hf_your_token_here
```

**방법 2: 환경변수**
```bash
export HUGGINGFACE_TOKEN=hf_your_token_here
```

### ⏱️ 첫 실행 시 주의사항

#### 🔽 모델 다운로드 시간
- **첫 실행 시 약 1-2GB 모델 다운로드**
- **다운로드 시간: 10-30분** (인터넷 속도에 따라)
- **이후 실행부터는 빠름** (로컬에 캐시됨)

#### ⚡ 처리 시간 가이드
| 방법 | 10분 동영상 처리 시간 | 정확도 |
|------|---------------------|--------|
| 허깅페이스 AI | 3-6분 | 최고 (95%+) |
| 실용적 (권장) | 1-2분 | 높음 (85%+) |
| 고급 | 5-10분 | 높음 (80%+) |
| 자동 | 2-3분 | 보통 (70%+) |
| 간단 | 10-30초 | 낮음 (60%+) |

---

## 🎯 사용 가이드

### 1️⃣ 동영상 가져오기
- **파일 업로드**: 로컬 파일 직접 업로드
- **YouTube 다운로드**: URL 입력 후 다운로드

### 2️⃣ 편집하기
- 원하는 편집 도구 탭 선택
- 파라미터 조정
- 편집 실행

### 3️⃣ 화자 구분 (AI 기능)
1. **"👥 화자 구분"** 탭 선택
2. **감지 방법 선택**:
   - 처음 사용자: **"실용적 (권장)"**
   - 최고 정확도: **"허깅페이스 AI (최신)"**
   - 빠른 테스트: **"간단 (에너지 기반)"**
3. 예상 화자 수 설정 또는 자동 감지
4. **"화자 구간 감지"** 버튼 클릭
5. 결과 확인 및 수동 편집 (필요시)
6. 원하는 화자만 추출하여 저장

### 4️⃣ 저장하기
- 편집된 동영상 다운로드
- 원하는 위치에 저장

## 📁 프로젝트 구조

```
CutStudio/
├── app.py                          # Streamlit 메인 애플리케이션
├── video_editor.py                 # 동영상 편집 핵심 기능
├── speaker_detector.py             # 기본 화자 감지 기능
├── improved_speaker_detector.py    # 고급 화자 감지 기능
├── advanced_speaker_detector.py    # SpeechBrain 기반 화자 감지
├── practical_speaker_detector.py   # 실용적 화자 감지 기능
├── huggingface_speaker_detector.py # 허깅페이스 AI 화자 감지 ⭐
├── youtube_downloader.py           # YouTube 다운로드 기능
├── utils.py                        # 유틸리티 함수
├── requirements.txt                # Python 패키지 목록
├── .env.example                    # 환경변수 예제
├── .gitignore                      # Git 제외 파일
├── CLAUDE.md                       # Claude AI 가이드
└── README.md                       # 프로젝트 문서
```

## 🛠 기술 스택

- **Frontend**: Streamlit
- **Video Processing**: MoviePy, FFmpeg
- **Audio Processing**: PyDub, librosa
- **Machine Learning**: scikit-learn, scipy
- **Speech Detection**: Silero VAD
- **🆕 Speaker Diarization**: Pyannote 3.1 (허깅페이스)
- **Speaker Embedding**: SpeechBrain (선택사항)
- **YouTube Download**: yt-dlp

## ⚠️ 알려진 이슈 및 제한사항

### 🎯 화자 분리 기능 현황

#### ✅ 잘 작동하는 경우
- 깨끗한 음질의 대화
- 2-4명의 명확히 구분되는 화자
- 배경 소음이 적은 환경
- 각 화자가 충분한 시간 발화 (2초 이상)

#### ⚠️ 제한사항
- 배경 소음이 많거나 음질이 나쁜 경우 정확도 감소
- 동시 발화 구간은 분리가 어려움
- 화자 수가 많을수록 (5명 이상) 정확도 감소
- 실제 화자 신원 식별 불가 (SPEAKER_0, SPEAKER_1로만 구분)

#### 🚀 성능 최적화 팁
1. **긴 동영상**: 먼저 필요한 부분만 잘라서 처리
2. **화질 vs 속도**: 실용적 방법으로 시작 → 필요시 허깅페이스 AI 사용
3. **예상 화자 수 지정**: 자동 감지보다 정확
4. **후처리**: 결과 테이블에서 화자 레이블 수동 수정 가능

### 🔧 일반적인 문제 해결

#### 허깅페이스 관련
```bash
# 토큰 오류시
export HUGGINGFACE_TOKEN=your_token_here

# 모델 접근 권한 오류시
# → 위 3개 모델 페이지에서 사용 조건 동의 확인

# 느린 다운로드시
# → WiFi 연결 확인, 처음 1회만 오래 걸림
```

#### MoviePy 오류
```bash
# FFmpeg 경로 문제 시
export IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg

# 'NoneType' object has no attribute 'stdout' 오류 시
# → 앱이 자동으로 FFmpeg 직접 모드로 재시도합니다
```

#### YouTube 다운로드 실패
```bash
# 403 Forbidden 오류
pip install --upgrade yt-dlp

# 지역 제한 동영상
# → VPN 사용 또는 다른 동영상으로 시도
```

#### 메모리 부족
```bash
# 대용량 파일 처리 시
# → 동영상을 더 작은 부분으로 나누어 처리
# → 시스템 메모리 확인 및 다른 앱 종료
```

## 🚀 향후 개선 계획

### 단기 (v2.1)
- [ ] 실시간 진행률 표시 개선
- [ ] 배치 처리 기능 (여러 파일 동시 처리)
- [ ] 화자별 음성 특징 시각화

### 중기 (v3.0)
- [ ] GPU 가속 지원
- [ ] 실시간 스트리밍 처리
- [ ] 화자 이름 지정 및 저장 기능
- [ ] 음성 전사(STT) 기능 통합

### 장기 (v4.0)
- [ ] 웹 기반 완전 실시간 처리
- [ ] 다국어 화자 분리 지원
- [ ] API 서버 모드
- [ ] 클라우드 배포 지원

## 🤝 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

### 기여 방법
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 기여 가이드라인
- 코드 스타일: PEP 8 준수
- 커밋 메시지: 명확하고 설명적으로 작성
- 테스트: 새 기능에 대한 테스트 포함
- 문서: README 및 코드 주석 업데이트

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 👨‍💻 개발자

- **Kelly** - *Initial work & AI Integration* - [@jeromwolf](https://github.com/jeromwolf)

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들의 도움으로 만들어졌습니다:
- [Streamlit](https://streamlit.io/) - 웹 인터페이스
- [MoviePy](https://zulko.github.io/moviepy/) - 동영상 처리
- [Pyannote](https://github.com/pyannote/pyannote-audio) - AI 화자 분리 ⭐
- [Silero VAD](https://github.com/snakers4/silero-vad) - 음성 활동 감지
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube 다운로드
- [librosa](https://librosa.org/) - 오디오 분석
- [SpeechBrain](https://speechbrain.github.io/) - 화자 임베딩

---

## 🎬 스크린샷

*곧 추가 예정...*

## 📞 지원

문제가 발생하면 [Issues](https://github.com/jeromwolf/CutStudio/issues)에 보고해주세요.

**Happy Editing! 🎬✨**