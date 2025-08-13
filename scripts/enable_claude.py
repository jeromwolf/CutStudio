#!/usr/bin/env python3
"""
Claude 수동 활성화 스크립트
Claude를 사용하고 싶을 때 실행하는 스크립트
"""

import os
from pathlib import Path

def enable_claude():
    """Claude를 활성화합니다."""
    
    print("🔄 Claude 활성화 중...")
    
    # services/summarization.py 수정
    summarization_file = Path("services/summarization.py")
    
    if not summarization_file.exists():
        print("❌ services/summarization.py를 찾을 수 없습니다.")
        return False
    
    # 파일 읽기
    with open(summarization_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Claude 초기화 코드 복원
    old_code = """        # Claude 초기화 (현재 비활성화 - Gemini만 사용)
        self.claude_summarizer = None
        print("ℹ️ Claude는 현재 비활성화됨 (Gemini 사용)")"""
    
    new_code = """        # Claude 초기화
        try:
            from claude_summarizer import ClaudeSummarizer
            self.claude_summarizer = ClaudeSummarizer()
            st.session_state.claude_summarizer = self.claude_summarizer
            print("✅ Claude 초기화 성공")
        except Exception as e:
            print(f"❌ Claude 초기화 실패: {str(e)}")
            st.warning(f"Claude 초기화 실패: {str(e)}")
            self.claude_summarizer = None"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        # 파일 쓰기
        with open(summarization_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Claude 활성화 완료!")
        print("📍 Streamlit 앱을 재시작해야 변경사항이 적용됩니다.")
        print("   streamlit run app_refactored.py")
        return True
    else:
        print("ℹ️ Claude가 이미 활성화되어 있거나 코드를 찾을 수 없습니다.")
        return False

if __name__ == "__main__":
    enable_claude()