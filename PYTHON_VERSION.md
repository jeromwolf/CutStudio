# ⚠️ Python 버전 호환성 문제

## 문제
Python 3.13은 너무 최신 버전이라 많은 패키지가 호환되지 않습니다:
- torch (PyTorch)
- pandas
- moviepy

## 해결 방법

### 1. Python 3.12 설치 (권장)
```bash
# Homebrew로 Python 3.12 설치
brew install python@3.12

# 새로운 가상환경 생성
python3.12 -m venv venv_new

# 가상환경 활성화
source venv_new/bin/activate

# 패키지 설치
pip install -r requirements_stable.txt
```

### 2. 또는 Python 3.11 사용 (가장 안정적)
```bash
brew install python@3.11
python3.11 -m venv venv_new
source venv_new/bin/activate
pip install -r requirements_stable.txt
```

## 현재 상황
- Python 3.13 사용 중
- torch, pandas 등 핵심 패키지 호환성 문제
- Python 3.12 또는 3.11로 다운그레이드 필요