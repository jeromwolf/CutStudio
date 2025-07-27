# CutStudio - 동영상 편집기

Streamlit 기반의 강력한 웹 동영상 편집기입니다. YouTube 다운로드, 화자 구분, 다양한 편집 기능을 제공합니다.

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

### 👥 화자 구분 기능
- 자동 화자 감지
- 화자별 구간 분리
- 개별 화자 추출
- 화자별 통계 제공

## 💻 설치 방법

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
- 화자별 구간 확인
- 필요한 화자만 추출

### 4. 저장하기
- 편집된 동영상 다운로드
- 원하는 위치에 저장

## 📁 프로젝트 구조

```
CutStudio/
├── app.py                  # Streamlit 메인 애플리케이션
├── video_editor.py         # 동영상 편집 핵심 기능
├── speaker_detector.py     # 화자 감지 기능
├── youtube_downloader.py   # YouTube 다운로드 기능
├── utils.py               # 유틸리티 함수
├── requirements.txt       # Python 패키지 목록
├── .gitignore            # Git 제외 파일
├── CLAUDE.md             # Claude AI 가이드
└── README.md             # 프로젝트 문서
```

## 🛠 기술 스택

- **Frontend**: Streamlit
- **Video Processing**: MoviePy, OpenCV
- **Audio Processing**: PyDub, SpeechRecognition
- **YouTube Download**: yt-dlp
- **Speaker Detection**: pyannote.audio (선택사항)

## ⚠️ 주의사항

- 대용량 동영상은 처리 시간이 오래 걸릴 수 있습니다
- 임시 파일은 주기적으로 정리하세요 ("임시 파일 정리" 버튼)
- YouTube 다운로드는 저작권을 준수하여 사용하세요
- 화자 구분 기능은 음성 품질에 따라 정확도가 달라질 수 있습니다

## 🐛 문제 해결

### 일반적인 문제
1. **MoviePy 오류**: FFmpeg가 설치되어 있는지 확인
2. **메모리 부족**: 더 작은 동영상으로 시도하거나 시스템 메모리 확인
3. **YouTube 다운로드 실패**: URL이 올바른지 확인, yt-dlp 업데이트

### 의존성 문제
```bash
# numpy 버전 충돌 시
pip install numpy==1.24.3 --force-reinstall

# 전체 패키지 재설치
pip install -r requirements.txt --force-reinstall
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 👨‍💻 개발자

- GitHub: [@jeromwolf](https://github.com/jeromwolf)

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들의 도움으로 만들어졌습니다:
- [Streamlit](https://streamlit.io/)
- [MoviePy](https://zulko.github.io/moviepy/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)