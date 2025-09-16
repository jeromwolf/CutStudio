# Google Cloud Run 배포 가이드

## 사전 준비사항
1. Google Cloud 계정
2. gcloud CLI 설치
3. Docker Desktop 설치

## 1. 로컬에서 Docker 테스트

```bash
# Docker 이미지 빌드
docker build -t cutstudio .

# Docker 컨테이너 실행
docker run -p 8501:8501 cutstudio

# 또는 docker-compose 사용
docker-compose up --build
```

브라우저에서 http://localhost:8501 접속하여 테스트

## 2. Google Cloud 설정

```bash
# gcloud CLI 로그인
gcloud auth login

# 프로젝트 ID 설정 (your-project-id를 실제 프로젝트 ID로 변경)
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID

# Artifact Registry API 활성화
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com

# Artifact Registry 저장소 생성 (최초 1회)
gcloud artifacts repositories create cutstudio-repo \
    --repository-format=docker \
    --location=asia-northeast3 \
    --description="CutStudio Docker images"

# Docker 인증 설정
gcloud auth configure-docker asia-northeast3-docker.pkg.dev
```

## 3. Docker 이미지 빌드 및 푸시

```bash
# 이미지 태그 설정
export IMAGE_TAG=asia-northeast3-docker.pkg.dev/$PROJECT_ID/cutstudio-repo/cutstudio:latest

# 이미지 빌드
docker build -t $IMAGE_TAG .

# 이미지 푸시
docker push $IMAGE_TAG
```

## 4. Cloud Run 배포

```bash
# Cloud Run에 배포
gcloud run deploy cutstudio \
    --image=$IMAGE_TAG \
    --platform=managed \
    --region=asia-northeast3 \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=2 \
    --max-instances=10 \
    --min-instances=0 \
    --port=8501 \
    --timeout=3600
```

## 5. 추가 설정 (선택사항)

### 커스텀 도메인 연결
```bash
gcloud run domain-mappings create \
    --service=cutstudio \
    --domain=your-domain.com \
    --region=asia-northeast3
```

### 환경 변수 설정
```bash
gcloud run services update cutstudio \
    --update-env-vars="KEY=VALUE" \
    --region=asia-northeast3
```

### 메모리/CPU 조정
```bash
gcloud run services update cutstudio \
    --memory=4Gi \
    --cpu=4 \
    --region=asia-northeast3
```

## 6. 모니터링

- Cloud Console: https://console.cloud.google.com/run
- 로그 확인:
```bash
gcloud run services logs read cutstudio --region=asia-northeast3
```

## 7. 업데이트 배포

코드 수정 후:
```bash
# 이미지 재빌드
docker build -t $IMAGE_TAG .

# 이미지 푸시
docker push $IMAGE_TAG

# Cloud Run 업데이트 (자동으로 새 버전 배포)
gcloud run services update cutstudio \
    --image=$IMAGE_TAG \
    --region=asia-northeast3
```

## 주의사항

1. **비용**: Cloud Run은 사용한 만큼만 과금되지만, 대용량 비디오 처리 시 비용이 발생할 수 있음
2. **타임아웃**: 기본 타임아웃은 60초이므로 3600초(1시간)로 설정
3. **파일 크기**: Cloud Run의 기본 요청 크기 제한은 32MB, 필요시 조정 필요
4. **임시 저장소**: /tmp 디렉토리만 쓰기 가능, 최대 2GB

## 트러블슈팅

### 메모리 부족 에러
```bash
gcloud run services update cutstudio --memory=8Gi --region=asia-northeast3
```

### 타임아웃 에러
```bash
gcloud run services update cutstudio --timeout=3600 --region=asia-northeast3
```

### 권한 에러
```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$(gcloud iam service-accounts list --format='value(email)' | grep compute)" \
    --role="roles/run.admin"
```