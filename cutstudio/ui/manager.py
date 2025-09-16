"""
UI Manager - UI 컴포넌트 관리
"""

import streamlit as st
from typing import Any, Optional, Callable, Dict
from contextlib import contextmanager
from .theme import Theme, get_theme_css


class UIManager:
    """
    UI 관리자
    
    Features:
    - Consistent UI components
    - Theme management
    - Layout helpers
    - Animation support
    """
    
    def __init__(self, theme: Theme = Theme.PROFESSIONAL):
        """
        UI 매니저 초기화
        
        Args:
            theme: UI 테마
        """
        self.theme = theme
        self._apply_theme()
    
    def _apply_theme(self):
        """테마 적용"""
        st.markdown(get_theme_css(self.theme), unsafe_allow_html=True)
    
    def render_header(self, title: str, subtitle: str = ""):
        """메인 헤더 렌더링"""
        st.markdown(
            f"""
            <div class="main-header">
                <h1>{title}</h1>
                {"<p>" + subtitle + "</p>" if subtitle else ""}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def render_progress(self, step: Any, progress: float):
        """프로그레스 바 렌더링"""
        # 단계별 진행률 계산
        step_progress = {
            "import": 0.0,
            "analyze": 0.33,
            "export": 0.66
        }
        
        base_progress = step_progress.get(step.value if hasattr(step, 'value') else str(step), 0)
        total_progress = min(base_progress + (progress * 0.33), 1.0)
        
        st.markdown(
            f"""
            <div class="progress-container">
                <div class="progress-bar" style="width: {total_progress * 100}%"></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @contextmanager
    def create_step_card(self, title: str, subtitle: str, active: bool = False, completed: bool = False):
        """
        단계 카드 생성 컨텍스트 매니저
        
        Args:
            title: 카드 제목
            subtitle: 카드 부제목
            active: 활성 상태
            completed: 완료 상태
        """
        classes = ["step-card"]
        if active:
            classes.append("active")
        if completed:
            classes.append("completed")
        
        class_str = " ".join(classes)
        
        st.markdown(
            f"""
            <div class="{class_str}">
                <h3>{title}</h3>
                <p>{subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 컨테이너 생성
        container = st.container()
        
        try:
            yield container
        finally:
            pass
    
    def show_loading(self, message: str = "처리 중..."):
        """로딩 표시"""
        with st.spinner(message):
            st.markdown(
                '<div class="processing">⏳ 잠시만 기다려주세요...</div>',
                unsafe_allow_html=True
            )
    
    def show_success(self, message: str):
        """성공 메시지"""
        st.success(f"✅ {message}")
    
    def show_error(self, message: str):
        """에러 메시지"""
        st.error(f"❌ {message}")
    
    def show_info(self, message: str):
        """정보 메시지"""
        st.info(f"ℹ️ {message}")
    
    def create_metric_card(self, label: str, value: Any, delta: Optional[Any] = None):
        """메트릭 카드 생성"""
        col = st.columns(1)[0]
        with col:
            st.metric(label=label, value=value, delta=delta)
    
    def create_download_button(
        self,
        label: str,
        data: bytes,
        filename: str,
        mime: str,
        key: Optional[str] = None
    ):
        """다운로드 버튼 생성"""
        return st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime=mime,
            key=key,
            use_container_width=True
        )
    
    def create_expander(self, label: str, expanded: bool = False):
        """확장 가능한 섹션 생성"""
        return st.expander(label, expanded=expanded)
    
    def create_tabs(self, tab_names: list):
        """탭 생성"""
        return st.tabs(tab_names)
    
    def create_columns(self, spec: list):
        """컬럼 레이아웃 생성"""
        return st.columns(spec)
    
    def render_speaker_card(self, speaker_id: str, data: Dict[str, Any]):
        """화자 정보 카드 렌더링"""
        with st.container():
            st.markdown(
                f"""
                <div class="result-card">
                    <h4>{speaker_id}</h4>
                    <p>발화 시간: {data.get('duration', 0):.1f}초</p>
                    <p>세그먼트 수: {len(data.get('segments', []))}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if summary := data.get('summary'):
                st.write(f"**요약**: {summary}")
            
            if keywords := data.get('keywords'):
                st.write(f"**키워드**: {', '.join(keywords)}")
    
    def render_empty_state(self, message: str, icon: str = "📭"):
        """빈 상태 표시"""
        st.markdown(
            f"""
            <div style="text-align: center; padding: 3rem; color: #999;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
                <p style="font-size: 1.2rem;">{message}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def create_sidebar_section(self, title: str):
        """사이드바 섹션 생성"""
        with st.sidebar:
            st.markdown(f"### {title}")
            return st.container()
    
    def render_tooltip(self, text: str, tooltip: str):
        """툴팁이 있는 텍스트 렌더링"""
        st.markdown(
            f'<span title="{tooltip}" style="cursor: help; border-bottom: 1px dashed #666;">{text}</span>',
            unsafe_allow_html=True
        )