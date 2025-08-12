"""
화자 프로필 표시 컴포넌트
"""
import streamlit as st
from utils import format_time
from typing import Dict, Any


def display_speaker_profile(speaker_profiles: Dict[str, Any]):
    """
    화자별 프로필 카드를 표시합니다.
    
    Args:
        speaker_profiles: 화자별 프로필 정보 딕셔너리
    """
    if not speaker_profiles:
        st.info("화자 프로필 정보가 없습니다.")
        return
    
    # 화자별 프로필 카드 표시
    speakers = list(speaker_profiles.keys())
    cols = st.columns(min(len(speakers), 3))  # 최대 3열로 표시
    
    for i, speaker_id in enumerate(speakers):
        with cols[i % 3]:
            profile = speaker_profiles[speaker_id]
            
            # 프로필 카드
            with st.container():
                st.markdown(f"### {speaker_id}")
                
                # 썸네일 표시
                if profile.get('has_thumbnail', False):
                    thumbnail = profile['thumbnail']
                    st.image(
                        f"data:image/jpeg;base64,{thumbnail['image_base64']}",
                        caption=f"타임스탬프: {format_time(thumbnail['timestamp'])}",
                        width=150
                    )
                else:
                    st.info("썸네일 생성 실패")
                
                # 요약 정보 표시
                if profile.get('has_summary', False):
                    summary = profile['summary']
                    st.metric("총 발화 시간", f"{format_time(summary['total_duration'])}")
                    st.metric("발화 횟수", f"{summary['segment_count']}회")
                    st.metric("참여율", f"{summary['participation_rate']}%")
                    
                    # 첫 등장 시간
                    st.write(f"**첫 등장:** {format_time(summary['first_appearance'])}")
                    st.write(f"**마지막 등장:** {format_time(summary['last_appearance'])}")
                
                st.markdown("---")


def display_speaker_statistics(speaker_profiles: Dict[str, Any]):
    """
    화자별 상세 통계를 표시합니다.
    
    Args:
        speaker_profiles: 화자별 프로필 정보 딕셔너리
    """
    st.subheader("📊 화자별 발화 통계")
    
    # 통계 데이터 집계
    stats_data = []
    for speaker_id, profile in speaker_profiles.items():
        if profile.get('has_summary', False):
            summary = profile['summary']
            stats_data.append({
                "화자": speaker_id,
                "총 발화 시간": format_time(summary['total_duration']),
                "발화 횟수": summary['segment_count'],
                "참여율": f"{summary['participation_rate']}%",
                "평균 발화 길이": format_time(summary['total_duration'] / summary['segment_count'])
            })
    
    if stats_data:
        import pandas as pd
        df = pd.DataFrame(stats_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # 시각화
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 참여율 분포")
            participation_data = {
                profile['summary']['speaker_id']: profile['summary']['participation_rate']
                for profile in speaker_profiles.values()
                if profile.get('has_summary', False)
            }
            st.bar_chart(participation_data)
        
        with col2:
            st.markdown("#### 발화 횟수 분포")
            segment_count_data = {
                profile['summary']['speaker_id']: profile['summary']['segment_count']
                for profile in speaker_profiles.values()
                if profile.get('has_summary', False)
            }
            st.bar_chart(segment_count_data)
    else:
        st.info("통계 정보가 없습니다.")