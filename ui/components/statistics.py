"""
통계 표시 컴포넌트
"""
import streamlit as st
from typing import List, Dict, Any
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


def calculate_speaker_statistics(segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    화자별 통계를 계산합니다.
    
    Args:
        segments: 화자 세그먼트 리스트
        
    Returns:
        화자별 통계 딕셔너리
    """
    speaker_stats = {}
    
    for seg in segments:
        speaker = seg['speaker']
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'count': 0,
                'total_duration': 0,
                'segments': [],
                'first_appearance': seg['start'],
                'last_appearance': seg['end']
            }
        
        stats = speaker_stats[speaker]
        stats['count'] += 1
        stats['total_duration'] += seg['duration']
        stats['segments'].append(seg)
        stats['first_appearance'] = min(stats['first_appearance'], seg['start'])
        stats['last_appearance'] = max(stats['last_appearance'], seg['end'])
    
    return speaker_stats


def display_statistics(segments: List[Dict[str, Any]], video_duration: float = None):
    """
    화자별 통계를 표시합니다.
    
    Args:
        segments: 화자 세그먼트 리스트
        video_duration: 전체 비디오 길이 (초)
    """
    st.subheader("📊 화자별 요약 통계")
    
    # 화자별 통계 계산
    speaker_stats = calculate_speaker_statistics(segments)
    
    # 화자별 통계 카드 표시
    if speaker_stats:
        stats_cols = st.columns(len(speaker_stats))
        
        for idx, (speaker, stats) in enumerate(speaker_stats.items()):
            with stats_cols[idx]:
                speaker_emoji = SPEAKER_COLORS.get(speaker, '⚪')
                
                # 메인 메트릭
                st.metric(
                    label=f"{speaker_emoji} {speaker}",
                    value=f"{stats['count']}회",
                    delta=f"{format_time(stats['total_duration'])}"
                )
                
                # 참여율 계산
                if video_duration and video_duration > 0:
                    participation = (stats['total_duration'] / video_duration) * 100
                    st.caption(f"참여율: {participation:.1f}%")
                
                # 추가 정보
                st.caption(f"첫 발화: {format_time(stats['first_appearance'])}")
                st.caption(f"마지막: {format_time(stats['last_appearance'])}")
    else:
        st.info("통계 정보가 없습니다.")


def display_detailed_statistics(segments: List[Dict[str, Any]], video_duration: float = None):
    """
    화자별 상세 통계를 표시합니다.
    
    Args:
        segments: 화자 세그먼트 리스트
        video_duration: 전체 비디오 길이 (초)
    """
    import pandas as pd
    
    st.subheader("📊 상세 통계 분석")
    
    # 화자별 통계 계산
    speaker_stats = calculate_speaker_statistics(segments)
    
    # DataFrame 생성
    stats_data = []
    total_duration = sum(stats['total_duration'] for stats in speaker_stats.values())
    
    for speaker, stats in speaker_stats.items():
        participation = (stats['total_duration'] / video_duration * 100) if video_duration else 0
        avg_duration = stats['total_duration'] / stats['count'] if stats['count'] > 0 else 0
        
        stats_data.append({
            "화자": speaker,
            "발화 횟수": stats['count'],
            "총 발화 시간": format_time(stats['total_duration']),
            "평균 발화 길이": format_time(avg_duration),
            "참여율": f"{participation:.1f}%",
            "첫 발화": format_time(stats['first_appearance']),
            "마지막 발화": format_time(stats['last_appearance'])
        })
    
    # DataFrame 표시
    df = pd.DataFrame(stats_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # 시각화
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 참여율 분포")
        participation_data = {
            speaker: (stats['total_duration'] / video_duration * 100) if video_duration else 0
            for speaker, stats in speaker_stats.items()
        }
        st.bar_chart(participation_data)
    
    with col2:
        st.markdown("#### 💬 발화 횟수 분포")
        count_data = {
            speaker: stats['count']
            for speaker, stats in speaker_stats.items()
        }
        st.bar_chart(count_data)
    
    # 발화 패턴 분석
    st.markdown("#### 🔄 발화 패턴 분석")
    
    # 시간대별 발화 분포
    time_bins = 10  # 10개 구간으로 나누기
    if video_duration:
        bin_size = video_duration / time_bins
        time_distribution = {speaker: [0] * time_bins for speaker in speaker_stats.keys()}
        
        for seg in segments:
            bin_idx = min(int(seg['start'] / bin_size), time_bins - 1)
            time_distribution[seg['speaker']][bin_idx] += 1
        
        # 시간대별 분포 차트
        import plotly.graph_objects as go
        
        fig = go.Figure()
        for speaker, distribution in time_distribution.items():
            fig.add_trace(go.Scatter(
                x=list(range(time_bins)),
                y=distribution,
                mode='lines+markers',
                name=speaker,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="시간대별 발화 분포",
            xaxis_title="시간 구간",
            yaxis_title="발화 횟수",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)