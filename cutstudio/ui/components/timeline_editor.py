"""
Timeline Editor - 전문가용 타임라인 편집 컴포넌트
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import base64
import cv2
import numpy as np


@dataclass
class TimelineSegment:
    """타임라인 세그먼트"""
    speaker_id: str
    start_time: float
    end_time: float
    transcript: str = ""
    summary: str = ""
    thumbnail_path: Optional[str] = None
    color: str = "#3498db"
    selected: bool = False


class TimelineEditor:
    """
    전문가용 타임라인 편집기
    
    Features:
    - 화자별 구간 시각화
    - 드래그로 구간 선택
    - 썸네일 미리보기
    - 원클릭 Cut 기능
    - 실시간 편집
    """
    
    def __init__(self):
        self.segments: List[TimelineSegment] = []
        self.selected_segments: List[str] = []
        self.speaker_colors = {
            'speaker_0': '#e74c3c',  # 빨강
            'speaker_1': '#3498db',  # 파랑
            'speaker_2': '#f39c12',  # 주황
            'speaker_3': '#27ae60',  # 초록
            'speaker_4': '#9b59b6',  # 보라
            'teacher': '#c0392b',    # 진한 빨강
            'student': '#2980b9',    # 진한 파랑
        }
        
        # CSS 스타일
        self._inject_css()
    
    def _inject_css(self):
        """커스텀 CSS 주입"""
        st.markdown("""
        <style>
        .timeline-container {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .segment-card {
            background: white;
            border-radius: 12px;
            padding: 15px;
            margin: 8px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 4px solid;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .segment-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .segment-card.selected {
            background: #e8f4fd;
            border-left-width: 6px;
        }
        
        .speaker-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            color: white;
            font-size: 12px;
            font-weight: bold;
            margin-right: 10px;
        }
        
        .time-badge {
            background: #ecf0f1;
            color: #2c3e50;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-family: monospace;
        }
        
        .cut-button {
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .cut-button:hover {
            background: linear-gradient(45deg, #c0392b, #a93226);
            transform: scale(1.05);
        }
        
        .thumbnail-preview {
            width: 80px;
            height: 60px;
            border-radius: 8px;
            object-fit: cover;
            border: 2px solid #ecf0f1;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def load_segments(self, speaker_data: Dict[str, List[Dict[str, Any]]]):
        """화자 데이터에서 세그먼트 로드"""
        self.segments = []
        
        for speaker_id, segments in speaker_data.items():
            color = self.speaker_colors.get(speaker_id, '#95a5a6')
            
            for segment in segments:
                self.segments.append(TimelineSegment(
                    speaker_id=speaker_id,
                    start_time=segment.get('start', 0),
                    end_time=segment.get('end', 0),
                    transcript=segment.get('transcript', ''),
                    summary=segment.get('summary', ''),
                    thumbnail_path=segment.get('thumbnail', None),
                    color=color
                ))
        
        # 시간순 정렬
        self.segments.sort(key=lambda x: x.start_time)
    
    def render_timeline_chart(self) -> Optional[go.Figure]:
        """타임라인 차트 렌더링"""
        if not self.segments:
            return None
        
        fig = go.Figure()
        
        # 화자별로 그룹화
        speakers = {}
        for segment in self.segments:
            if segment.speaker_id not in speakers:
                speakers[segment.speaker_id] = []
            speakers[segment.speaker_id].append(segment)
        
        # 각 화자별로 바 추가
        y_pos = 0
        y_labels = []
        
        for speaker_id, segments in speakers.items():
            y_labels.append(f"🎤 {speaker_id.replace('_', ' ').title()}")
            
            for segment in segments:
                duration = segment.end_time - segment.start_time
                
                fig.add_trace(go.Bar(
                    name=speaker_id,
                    x=[duration],
                    y=[y_pos],
                    base=[segment.start_time],
                    orientation='h',
                    marker=dict(
                        color=segment.color,
                        opacity=0.8 if not segment.selected else 1.0,
                        line=dict(
                            color='white' if not segment.selected else 'black',
                            width=2 if segment.selected else 1
                        )
                    ),
                    text=f"{segment.speaker_id}<br>{self._format_time(segment.start_time)} - {self._format_time(segment.end_time)}",
                    textposition='inside',
                    textfont=dict(color='white', size=10),
                    hovertemplate=(
                        f"<b>{segment.speaker_id}</b><br>"
                        f"시간: {self._format_time(segment.start_time)} - {self._format_time(segment.end_time)}<br>"
                        f"길이: {self._format_time(duration)}<br>"
                        f"내용: {segment.transcript[:50]}...<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False
                ))
            
            y_pos += 1
        
        # 레이아웃 설정
        fig.update_layout(
            title={
                'text': "🎬 타임라인 에디터 - 화자별 구간",
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title="시간 (초)",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(y_labels))),
                ticktext=y_labels,
                autorange='reversed'
            ),
            height=max(300, len(speakers) * 80),
            margin=dict(l=150, r=50, t=80, b=50),
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12),
            dragmode='select'
        )
        
        # 그리드 스타일
        fig.update_xaxes(
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=1,
            showgrid=True
        )
        fig.update_yaxes(
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=1,
            showgrid=True
        )
        
        return fig
    
    def render_segment_cards(self):
        """세그먼트 카드 렌더링"""
        if not self.segments:
            st.info("🎭 화자 분석을 먼저 실행해주세요!")
            return
        
        st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
        st.markdown("### 📋 세그먼트 목록")
        
        # 화자별 필터
        speakers = list(set(segment.speaker_id for segment in self.segments))
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            selected_speakers = st.multiselect(
                "👥 화자 선택",
                speakers,
                default=speakers,
                key="speaker_filter"
            )
        
        with col2:
            min_duration = st.slider(
                "⏱ 최소 길이 (초)",
                0.0, 30.0, 0.0,
                key="min_duration_filter"
            )
        
        with col3:
            if st.button("🔄 전체 선택/해제", key="toggle_all"):
                self._toggle_all_segments()
        
        # 필터링된 세그먼트
        filtered_segments = [
            seg for seg in self.segments
            if seg.speaker_id in selected_speakers
            and (seg.end_time - seg.start_time) >= min_duration
        ]
        
        # 세그먼트 카드들
        for i, segment in enumerate(filtered_segments):
            with st.container():
                self._render_segment_card(segment, i)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 선택된 세그먼트 요약
        selected_count = len([seg for seg in self.segments if seg.selected])
        if selected_count > 0:
            st.success(f"✂️ {selected_count}개 세그먼트가 선택되었습니다!")
    
    def _render_segment_card(self, segment: TimelineSegment, index: int):
        """개별 세그먼트 카드 렌더링"""
        duration = segment.end_time - segment.start_time
        segment_id = f"segment_{index}"
        
        # 카드 컨테이너
        card_class = "segment-card selected" if segment.selected else "segment-card"
        
        col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
        
        with col1:
            # 썸네일
            if segment.thumbnail_path and Path(segment.thumbnail_path).exists():
                st.image(segment.thumbnail_path, width=80)
            else:
                # 기본 아바타
                st.markdown(f"""
                <div style="width:80px;height:60px;background:{segment.color};
                           border-radius:8px;display:flex;align-items:center;
                           justify-content:center;color:white;font-size:24px;">
                    🎤
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # 세그먼트 정보
            st.markdown(f"""
            <div style="border-left: 4px solid {segment.color}; padding-left: 10px;">
                <div style="margin-bottom: 5px;">
                    <span class="speaker-badge" style="background-color: {segment.color};">
                        {segment.speaker_id.replace('_', ' ').title()}
                    </span>
                    <span class="time-badge">
                        {self._format_time(segment.start_time)} - {self._format_time(segment.end_time)}
                        ({self._format_time(duration)})
                    </span>
                </div>
                <div style="font-size: 13px; color: #7f8c8d; line-height: 1.4;">
                    {segment.transcript[:100]}{'...' if len(segment.transcript) > 100 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # 요약
            if segment.summary:
                with st.expander("📝 요약", expanded=False):
                    st.write(segment.summary)
            else:
                st.markdown("*요약 없음*")
        
        with col4:
            # 선택 체크박스
            segment.selected = st.checkbox(
                "선택",
                value=segment.selected,
                key=f"select_{segment_id}"
            )
            
            # Cut 버튼
            if st.button(
                "✂️ Cut",
                key=f"cut_{segment_id}",
                help="이 세그먼트를 개별 파일로 저장"
            ):
                self._cut_segment(segment)
    
    def _cut_segment(self, segment: TimelineSegment):
        """개별 세그먼트 Cut"""
        st.success(f"✂️ {segment.speaker_id} 세그먼트를 Cut했습니다!")
        # TODO: 실제 Cut 로직 연결
    
    def _toggle_all_segments(self):
        """모든 세그먼트 선택/해제"""
        all_selected = all(seg.selected for seg in self.segments)
        for segment in self.segments:
            segment.selected = not all_selected
    
    def _format_time(self, seconds: float) -> str:
        """시간 포맷팅"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def get_selected_segments(self) -> List[TimelineSegment]:
        """선택된 세그먼트 목록"""
        return [seg for seg in self.segments if seg.selected]
    
    def render_cut_panel(self):
        """Cut 패널 렌더링"""
        selected_segments = self.get_selected_segments()
        
        if not selected_segments:
            st.info("🎯 세그먼트를 선택한 후 Cut 작업을 진행하세요!")
            return
        
        st.markdown("### ✂️ Cut 작업 패널")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎬 개별 파일로 Cut", use_container_width=True):
                self._cut_individual_files(selected_segments)
        
        with col2:
            if st.button("📁 화자별로 묶어서 Cut", use_container_width=True):
                self._cut_by_speaker(selected_segments)
        
        with col3:
            if st.button("🔗 연결해서 Cut", use_container_width=True):
                self._cut_concatenated(selected_segments)
        
        # Cut 옵션
        with st.expander("⚙️ 고급 옵션"):
            output_format = st.selectbox(
                "출력 형식",
                ["MP4", "AVI", "MOV", "MP3 (오디오만)"],
                key="cut_format"
            )
            
            quality = st.select_slider(
                "품질",
                ["빠름", "보통", "고품질"],
                value="보통",
                key="cut_quality"
            )
            
            add_margins = st.checkbox(
                "구간 전후로 여유시간 추가",
                value=False,
                key="add_margins"
            )
            
            if add_margins:
                margin_seconds = st.number_input(
                    "여유시간 (초)",
                    0.0, 5.0, 0.5,
                    key="margin_time"
                )
    
    def _cut_individual_files(self, segments: List[TimelineSegment]):
        """개별 파일로 Cut"""
        st.success(f"✂️ {len(segments)}개 세그먼트를 개별 파일로 Cut합니다!")
        # TODO: 실제 Cut 로직
    
    def _cut_by_speaker(self, segments: List[TimelineSegment]):
        """화자별로 묶어서 Cut"""
        speakers = {}
        for seg in segments:
            if seg.speaker_id not in speakers:
                speakers[seg.speaker_id] = []
            speakers[seg.speaker_id].append(seg)
        
        st.success(f"📁 {len(speakers)}명 화자의 세그먼트를 각각 하나의 파일로 Cut합니다!")
        # TODO: 실제 Cut 로직
    
    def _cut_concatenated(self, segments: List[TimelineSegment]):
        """연결해서 Cut"""
        total_duration = sum(seg.end_time - seg.start_time for seg in segments)
        st.success(f"🔗 선택된 세그먼트들을 연결하여 하나의 파일로 Cut합니다! (총 {self._format_time(total_duration)})")
        # TODO: 실제 Cut 로직