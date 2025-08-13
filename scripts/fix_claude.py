#!/usr/bin/env python3
"""
Claude 초기화 문제 해결 스크립트
"""
import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프록시 관련 환경 변수 제거
proxy_vars = [
    'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
    'FTP_PROXY', 'ftp_proxy', 'NO_PROXY', 'no_proxy'
]

print("🧹 프록시 환경 변수 정리 중...")
for var in proxy_vars:
    if var in os.environ:
        print(f"  - {var}: {os.environ[var]} (제거됨)")
        del os.environ[var]
    else:
        print(f"  - {var}: 없음")

print("\n🔑 API 키 확인 중...")
api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key:
    print(f"  ✅ ANTHROPIC_API_KEY: {api_key[:10]}...{api_key[-4:]}")
else:
    print("  ❌ ANTHROPIC_API_KEY: 없음")
    sys.exit(1)

print("\n🧪 Claude 초기화 테스트 중...")
try:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    print("  ✅ Claude 클라이언트 초기화 성공")
    
    # 간단한 테스트 요청
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("  ✅ Claude API 테스트 성공")
    except Exception as e:
        print(f"  ⚠️ Claude API 테스트 실패: {e}")
        
except Exception as e:
    print(f"  ❌ Claude 초기화 실패: {e}")
    
    # 상세한 에러 정보
    if 'proxies' in str(e):
        print("\n💡 해결 방법:")
        print("1. 터미널에서 다음 명령어 실행:")
        for var in proxy_vars:
            print(f"   unset {var}")
        print("2. Streamlit 앱 재시작")
    
print("\n✅ 스크립트 완료")