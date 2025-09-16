# Python 3.12 베이스 이미지 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY app.py .
COPY video_editor.py .
COPY youtube_downloader.py .
COPY utils.py .
COPY services/ ./services/
COPY modules/ ./modules/
COPY .streamlit/ ./.streamlit/

# 필요한 디렉토리 생성
RUN mkdir -p temp processed exports

# Streamlit 포트 설정
EXPOSE 8501

# 헬스체크 설정
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Streamlit 실행
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false", \
    "--server.maxUploadSize=500"]