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