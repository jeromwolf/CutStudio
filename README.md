# CutStudio - AI 기반 동영상 편집기

CutStudio는 AI 화자 인식 기능을 갖춘 웹 기반 동영상 편집 도구입니다. YouTube 동영상 다운로드, 화자별 구간 자동 분리, 다양한 편집 기능을 제공합니다.

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

### 👥 AI 화자 분리 (Beta)
- **자동 화자 감지**: 동영상에서 여러 화자 자동 구분
- **3가지 감지 방법**:
  - 간단 (에너지 기반): 빠르지만 정확도 낮음
  - 자동 (MFCC + 클러스터링): 균형잡힌 성능
  - 고급 (향상된 특징 + 스펙트럴): 가장 정확하지만 느림 (1-3분)
- **화자별 동영상 분리**: 각 화자의 구간만 추출
- **수동 편집**: 화자 레이블 수정 가능
- **신뢰도 표시**: 각 구간의 화자 감지 신뢰도 확인

## 💻 설치 방법

### 필수 요구사항
- Python 3.8+
- FFmpeg

### 1. 저장소 클론
```bash
git clone https://github.com/jeromwolf/CutStudio.git
cd CutStudio
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. FFmpeg 설치 (필수)
- Mac: `brew install ffmpeg`
- Windows: [FFmpeg 공식 사이트](https://ffmpeg.org/download.html)에서 다운로드
- Linux: `sudo apt-get install ffmpeg`

## 🎯 실행 방법

```bash
streamlit run app.py
```

브라우저에서 자동으로 열리거나 http://localhost:8501 접속

## 📖 사용 가이드

### 1. 동영상 가져오기
- **파일 업로드**: 로컬 파일 직접 업로드
- **YouTube 다운로드**: URL 입력 후 다운로드

### 2. 편집하기
- 원하는 편집 도구 탭 선택
- 파라미터 조정
- 편집 실행

### 3. 화자 구분 (선택사항)
- "화자 구분" 탭에서 분석 실행
- 예상 화자 수 설정 또는 자동 감지
- 감지 방법 선택
- 화자별 구간 확인 및 수동 편집
- 필요한 화자만 추출

### 4. 저장하기
- 편집된 동영상 다운로드
- 원하는 위치에 저장

## 📁 프로젝트 구조

```
CutStudio/
├── app.py                      # Streamlit 메인 애플리케이션
├── video_editor.py             # 동영상 편집 핵심 기능
├── speaker_detector.py         # 기본 화자 감지 기능
├── improved_speaker_detector.py # 고급 화자 감지 기능
├── advanced_speaker_detector.py # SpeechBrain 기반 화자 감지
├── youtube_downloader.py       # YouTube 다운로드 기능
├── utils.py                   # 유틸리티 함수
├── requirements.txt           # Python 패키지 목록
├── .gitignore                # Git 제외 파일
├── CLAUDE.md                 # Claude AI 가이드
└── README.md                 # 프로젝트 문서
```

## 🛠 기술 스택

- **Frontend**: Streamlit
- **Video Processing**: MoviePy, FFmpeg
- **Audio Processing**: PyDub, librosa
- **Machine Learning**: scikit-learn, scipy
- **Speech Detection**: Silero VAD
- **Speaker Embedding**: SpeechBrain (선택사항)
- **YouTube Download**: yt-dlp

## ⚠️ 알려진 이슈 및 제한사항

### 화자 분리 기능 (추가 튜닝 필요) 🚧
현재 화자 분리 기능은 베타 버전으로 다음과 같은 제한사항이 있습니다:

#### 정확도 관련
- 배경 소음이 많거나 음질이 나쁜 경우 정확도가 크게 떨어집니다
- 동시에 여러 명이 말하는 구간(오버랩)은 제대로 분리되지 않습니다
- 화자 수가 많을수록 (4명 이상) 정확도가 감소합니다
- 짧은 발화(2초 미만)는 화자 구분이 어렵습니다

#### 기술적 제한
- 실제 화자 신원은 알 수 없고, SPEAKER_0, SPEAKER_1 등으로만 구분됩니다
- 같은 화자라도 목소리 톤이 크게 달라지면 다른 화자로 인식될 수 있습니다
- 처리 시간이 동영상 길이에 비례하여 증가합니다

### 향후 개선 계획
- [ ] 더 정확한 화자 임베딩 모델 적용 (pyannote 3.1 정식 통합)
- [ ] 실시간 처리 지원
- [ ] 화자별 음성 특징 학습 및 저장 기능
- [ ] 배경 소음 제거 기능 강화
- [ ] GPU 가속 지원
- [ ] 동시 발화(오버랩) 처리 개선
- [ ] 화자 이름 지정 기능

### 기타 제한사항
- 대용량 동영상 처리 시 메모리 사용량이 높을 수 있습니다
- 일부 YouTube 동영상은 다운로드가 제한될 수 있습니다
- MoviePy의 일부 버그로 인해 특정 코덱 처리 시 오류가 발생할 수 있습니다

## 🐛 문제 해결

### 화자 감지 관련
```bash
# 화자 감지가 너무 느린 경우
- "간단" 또는 "자동" 방법을 사용하세요
- 최소 발화 시간을 늘려보세요 (예: 3초)
- 예상 화자 수를 직접 지정하세요

# 화자가 잘못 구분되는 경우
- 결과 테이블에서 화자 레이블을 수동으로 수정하세요
- "변경사항 적용" 버튼을 클릭하여 저장하세요
```

### MoviePy 오류
```bash
# FFmpeg 경로 문제 시
export IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg

# 'NoneType' object has no attribute 'stdout' 오류 시
- 앱이 자동으로 FFmpeg 직접 모드로 재시도합니다
```

### 메모리 부족
```bash
# 대용량 파일 처리 시
- 동영상을 더 작은 부분으로 나누어 처리
- 시스템 메모리 확인 및 다른 앱 종료
```

### YouTube 다운로드 실패
```bash
# 403 Forbidden 오류
pip install --upgrade yt-dlp

# 지역 제한 동영상
VPN 사용 또는 다른 동영상으로 시도
```

## 🤝 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

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

- GitHub: [@jeromwolf](https://github.com/jeromwolf)

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들의 도움으로 만들어졌습니다:
- [Streamlit](https://streamlit.io/) - 웹 인터페이스
- [MoviePy](https://zulko.github.io/moviepy/) - 동영상 처리
- [Silero VAD](https://github.com/snakers4/silero-vad) - 음성 활동 감지
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube 다운로드
- [librosa](https://librosa.org/) - 오디오 분석
- [SpeechBrain](https://speechbrain.github.io/) - 화자 임베딩