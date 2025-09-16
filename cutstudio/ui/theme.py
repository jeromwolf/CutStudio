"""
UI Theme Configuration
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any


class Theme(Enum):
    """UI 테마 종류"""
    PROFESSIONAL = "professional"
    EDUCATIONAL = "educational"
    MINIMAL = "minimal"


@dataclass
class ThemeConfig:
    """테마 설정"""
    primary_color: str
    secondary_color: str
    background_color: str
    text_color: str
    accent_color: str
    font_family: str
    border_radius: str
    shadow: str


# 테마별 설정
THEME_CONFIGS = {
    Theme.PROFESSIONAL: ThemeConfig(
        primary_color="#1f77b4",
        secondary_color="#ff7f0e",
        background_color="#ffffff",
        text_color="#262730",
        accent_color="#2ca02c",
        font_family="'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif",
        border_radius="12px",
        shadow="0 2px 8px rgba(0,0,0,0.1)"
    ),
    Theme.EDUCATIONAL: ThemeConfig(
        primary_color="#4CAF50",
        secondary_color="#2196F3",
        background_color="#f5f5f5",
        text_color="#333333",
        accent_color="#FF9800",
        font_family="'Noto Sans KR', sans-serif",
        border_radius="8px",
        shadow="0 1px 4px rgba(0,0,0,0.1)"
    ),
    Theme.MINIMAL: ThemeConfig(
        primary_color="#000000",
        secondary_color="#666666",
        background_color="#ffffff",
        text_color="#000000",
        accent_color="#0066cc",
        font_family="'Inter', sans-serif",
        border_radius="4px",
        shadow="none"
    )
}


def get_theme_css(theme: Theme) -> str:
    """테마별 CSS 반환"""
    config = THEME_CONFIGS[theme]
    
    return f"""
    <style>
    /* CutStudio Pro Theme */
    .stApp {{
        font-family: {config.font_family};
    }}
    
    /* 헤더 스타일 */
    .main-header {{
        background: linear-gradient(135deg, {config.primary_color}ee, {config.secondary_color}ee);
        color: white;
        padding: 2rem;
        border-radius: {config.border_radius};
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: {config.shadow};
    }}
    
    .main-header h1 {{
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }}
    
    .main-header p {{
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }}
    
    /* 카드 스타일 */
    .step-card {{
        background: {config.background_color};
        border: 1px solid #e0e0e0;
        border-radius: {config.border_radius};
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: {config.shadow};
        transition: all 0.3s ease;
    }}
    
    .step-card.active {{
        border-color: {config.primary_color};
        box-shadow: 0 4px 16px rgba(31, 119, 180, 0.15);
    }}
    
    .step-card.completed {{
        border-color: {config.accent_color};
        background: #f0f9ff;
    }}
    
    .step-card h3 {{
        margin: 0 0 0.5rem 0;
        color: {config.text_color};
        font-size: 1.3rem;
        font-weight: 600;
    }}
    
    .step-card p {{
        margin: 0;
        color: #666;
        font-size: 0.95rem;
    }}
    
    /* 버튼 스타일 */
    .stButton > button {{
        border-radius: {config.border_radius};
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton > button[type="primary"] {{
        background: {config.primary_color};
        color: white;
    }}
    
    .stButton > button[type="primary"]:hover {{
        background: {config.secondary_color};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    /* 프로그레스 바 */
    .progress-container {{
        background: #f0f0f0;
        border-radius: {config.border_radius};
        height: 8px;
        margin: 2rem 0;
        overflow: hidden;
    }}
    
    .progress-bar {{
        background: linear-gradient(90deg, {config.primary_color}, {config.secondary_color});
        height: 100%;
        transition: width 0.5s ease;
        border-radius: {config.border_radius};
    }}
    
    /* 결과 카드 */
    .result-card {{
        background: #f8f9fa;
        border-left: 4px solid {config.primary_color};
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 {config.border_radius} {config.border_radius} 0;
    }}
    
    /* 애니메이션 */
    @keyframes pulse {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
        100% {{ opacity: 1; }}
    }}
    
    .processing {{
        animation: pulse 2s infinite;
    }}
    
    /* 반응형 디자인 */
    @media (max-width: 768px) {{
        .main-header h1 {{
            font-size: 2rem;
        }}
        
        .step-card {{
            padding: 1rem;
        }}
    }}
    </style>
    """