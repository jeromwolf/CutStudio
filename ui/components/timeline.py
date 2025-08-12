"""
타임라인 표시 컴포넌트
"""
import streamlit as st
from PIL import Image
import base64
import io
from typing import List, Dict, Any, Optional
from utils import format_time


# 화자별 색상 이모지 매핑
SPEAKER_COLORS = {
    'SPEAKER_0': '🔴',
    'SPEAKER_1': '🔵', 
    'SPEAKER_2': '🟢',
    'SPEAKER_3': '🟡',
    'SPEAKER_4': '🟣',
    'SPEAKER_5': '🟤'
}


def generate_thumbnail(video_clip, timestamp: float, size: tuple = (200, 150)) -> Optional[str]:
    """
    비디오에서 썸네일을 생성합니다.
    
    Args:
        video_clip: MoviePy VideoFileClip 객체
        timestamp: 썸네일을 생성할 시간 (초)
        size: 썸네일 크기 (width, height)
        
    Returns:
        Base64 인코딩된 이미지 문자열 또는 None
    """
    try:
        frame = video_clip.get_frame(timestamp)
        pil_image = Image.fromarray(frame)
        pil_image.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Base64 인코딩
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    except Exception:
        return None


def display_timeline_card(
    segment: Dict[str, Any],
    video_clip: Any,
    recognized_text: str = "",
    summarizer: Any = None,
    show_summary: bool = True
):
    """
    타임라인 카드 하나를 표시합니다.
    
    Args:
        segment: 세그먼트 정보
        video_clip: MoviePy VideoFileClip 객체
        recognized_text: 음성 인식된 텍스트
        summarizer: 요약기 객체 (Gemini/Claude)
        show_summary: 요약 표시 여부
    """
    # 썸네일 생성 및 표시
    mid_time = (segment['start'] + segment['end']) / 2
    thumbnail_base64 = generate_thumbnail(video_clip, mid_time) if video_clip else None
    
    if thumbnail_base64:
        st.image(f"data:image/jpeg;base64,{thumbnail_base64}", use_container_width=True)
    else:
        st.image("https://via.placeholder.com/200x150?text=No+Thumbnail", use_container_width=True)
    
    # 화자 정보
    speaker_emoji = SPEAKER_COLORS.get(segment['speaker'], '⚪')
    st.markdown(f"### {speaker_emoji} {segment['speaker']}")
    
    # 시간 정보
    st.caption(f"⏱️ {format_time(segment['start'])} - {format_time(segment['end'])}")
    st.caption(f"📏 길이: {format_time(segment['duration'])}")
    
    # 텍스트/요약 표시
    if recognized_text:
        text_length = len(recognized_text)
        
        if show_summary and summarizer and text_length > 50:
            try:
                # 요약 시도
                summary = summarizer.summarize_text(recognized_text, 150)
                st.info(f"💬 {summary}")
                st.caption(f"✅ AI 요약 완료 | 원본: {text_length}자")
            except Exception as e:
                # 요약 실패 시 원본 텍스트의 일부 표시
                display_text = recognized_text[:150] + "..." if text_length > 150 else recognized_text
                st.info(f"💬 {display_text}")
                st.caption(f"원본 텍스트 | {text_length}자")
        else:
            # 짧은 텍스트는 그대로 표시
            st.info(f"💬 {recognized_text}")
            st.caption(f"원본 텍스트 | {text_length}자")
    else:
        st.info("🔇 음성 인식 대기 중...")


def display_timeline(
    segments: List[Dict[str, Any]],
    video_clip: Any,
    recognized_segments: Optional[List[Dict[str, Any]]] = None,
    summarizer: Any = None,
    cols_per_row: int = 4
):
    """
    전체 타임라인을 표시합니다.
    
    Args:
        segments: 화자 세그먼트 리스트
        video_clip: MoviePy VideoFileClip 객체
        recognized_segments: 음성 인식된 세그먼트 리스트
        summarizer: 요약기 객체
        cols_per_row: 한 행에 표시할 카드 수
    """
    st.subheader("🎬 전체 대화 타임라인")
    
    # 시간순으로 정렬
    sorted_segments = sorted(segments, key=lambda x: x['start'])
    
    # 음성 인식 텍스트 매핑
    segment_texts = {}
    if recognized_segments:
        for rec_seg in recognized_segments:
            # 시간 기반으로 매칭
            for seg in sorted_segments:
                if (abs(rec_seg['start'] - seg['start']) < 1.0 and 
                    abs(rec_seg['end'] - seg['end']) < 1.0):
                    key = f"{seg['start']:.1f}_{seg['end']:.1f}"
                    segment_texts[key] = rec_seg.get('text', '').strip()
                    break
    
    # 전체 타임라인 정보
    st.info(f"📊 총 {len(sorted_segments)}개의 발화 구간")
    
    # 세그먼트를 행당 cols_per_row개씩 표시
    for i in range(0, len(sorted_segments), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, seg in enumerate(sorted_segments[i:i+cols_per_row]):
            if j < len(cols):
                with cols[j]:
                    seg_key = f"{seg['start']:.1f}_{seg['end']:.1f}"
                    text = segment_texts.get(seg_key, '')
                    
                    display_timeline_card(
                        segment=seg,
                        video_clip=video_clip,
                        recognized_text=text,
                        summarizer=summarizer
                    )