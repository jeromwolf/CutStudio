# Windows에서 FFmpeg 설치 가이드

## 🎬 FFmpeg란?
FFmpeg는 동영상과 오디오를 처리하는 필수 프로그램입니다. CutStudio가 동영상을 편집하려면 반드시 필요합니다.

## 📥 설치 방법

### 방법 1: 공식 사이트에서 다운로드 (권장)

1. **FFmpeg 다운로드**
   - [FFmpeg 공식 다운로드 페이지](https://www.gyan.dev/ffmpeg/builds/) 접속
   - "release builds" 섹션에서 `ffmpeg-release-essentials.zip` 다운로드

2. **압축 해제**
   - 다운로드한 ZIP 파일을 `C:\ffmpeg`에 압축 해제
   - 폴더 구조: `C:\ffmpeg\bin\ffmpeg.exe`가 존재해야 함

3. **시스템 환경 변수 설정**
   - Windows 키 + R → `sysdm.cpl` 입력 → Enter
   - "고급" 탭 → "환경 변수" 클릭
   - 시스템 변수에서 `Path` 선택 → "편집" 클릭
   - "새로 만들기" → `C:\ffmpeg\bin` 입력
   - 모든 창에서 "확인" 클릭

4. **설치 확인**
   - 명령 프롬프트(cmd) 새로 열기
   - `ffmpeg -version` 입력
   - 버전 정보가 표시되면 성공!

### 방법 2: Chocolatey 사용 (개발자용)

```bash
# Chocolatey가 설치되어 있다면
choco install ffmpeg
```

### 방법 3: Scoop 사용 (개발자용)

```bash
# Scoop이 설치되어 있다면
scoop install ffmpeg
```

## ⚠️ 일반적인 문제 해결

### 1. "ffmpeg is not recognized" 오류
- **원인**: PATH 환경 변수가 제대로 설정되지 않음
- **해결**: 
  - 명령 프롬프트를 완전히 닫고 새로 열기
  - 시스템 재시작
  - PATH에 `C:\ffmpeg\bin`이 정확히 추가되었는지 확인

### 2. "[WinError 2] 지정된 파일을 찾을 수 없습니다" 오류
- **원인**: FFmpeg가 설치되지 않았거나 PATH에 없음
- **해결**: 위의 설치 단계를 다시 따라하기

### 3. Python에서 FFmpeg 경로 직접 지정
```python
import os
os.environ['IMAGEIO_FFMPEG_EXE'] = r'C:\ffmpeg\bin\ffmpeg.exe'
```

## ✅ 설치 성공 확인
1. 새 명령 프롬프트 열기
2. 다음 명령어 실행:
   ```
   ffmpeg -version
   where ffmpeg
   ```
3. 두 명령어 모두 정상적으로 결과를 표시하면 성공!

## 🔗 추가 리소스
- [FFmpeg 공식 문서](https://ffmpeg.org/documentation.html)
- [Windows용 FFmpeg 설치 영상 가이드](https://www.youtube.com/results?search_query=install+ffmpeg+windows)

---
설치 후에도 문제가 발생하면 GitHub Issues에 문의해주세요!