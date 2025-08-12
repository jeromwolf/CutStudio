"""
앱 상태 관리 모듈
"""
import streamlit as st
from typing import Optional, Dict, Any


class AppState:
    """CutStudio 앱의 중앙 상태 관리 클래스"""
    
    def __init__(self):
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        default_states = {
            'video_editor': None,
            'youtube_url': '',
            'video_path': None,
            'speaker_segments': None,
            'recognized_segments': None,
            'segment_profiles': None,
            'processing_active': False,
            'recognition_completed': False,
            'speech_recognizer': None,
            'selected_speaker': None,
            'recognition_model': 'tiny',
            'gemini_summarizer': None,
            'claude_summarizer': None,
            'active_summarizer': 'gemini'
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @property
    def video_editor(self):
        """비디오 에디터 인스턴스 반환"""
        return st.session_state.get('video_editor')
    
    @video_editor.setter
    def video_editor(self, value):
        """비디오 에디터 인스턴스 설정"""
        st.session_state.video_editor = value
    
    @property
    def video_path(self):
        """비디오 파일 경로 반환"""
        return st.session_state.get('video_path')
    
    @video_path.setter
    def video_path(self, value):
        """비디오 파일 경로 설정"""
        st.session_state.video_path = value
    
    @property
    def speaker_segments(self):
        """화자 세그먼트 반환"""
        return st.session_state.get('speaker_segments')
    
    @speaker_segments.setter
    def speaker_segments(self, value):
        """화자 세그먼트 설정"""
        st.session_state.speaker_segments = value
    
    @property
    def recognized_segments(self):
        """인식된 세그먼트 반환"""
        return st.session_state.get('recognized_segments')
    
    @recognized_segments.setter
    def recognized_segments(self, value):
        """인식된 세그먼트 설정"""
        st.session_state.recognized_segments = value
    
    @property
    def segment_profiles(self):
        """세그먼트 프로필 반환"""
        return st.session_state.get('segment_profiles')
    
    @segment_profiles.setter
    def segment_profiles(self, value):
        """세그먼트 프로필 설정"""
        st.session_state.segment_profiles = value
    
    @property
    def is_processing(self) -> bool:
        """처리 중 상태 반환"""
        return st.session_state.get('processing_active', False)
    
    @is_processing.setter
    def is_processing(self, value: bool):
        """처리 중 상태 설정"""
        st.session_state.processing_active = value
    
    def get_active_summarizer(self):
        """현재 활성화된 요약기 반환"""
        if st.session_state.active_summarizer == 'gemini':
            return st.session_state.gemini_summarizer
        else:
            return st.session_state.claude_summarizer
    
    def reset_speaker_data(self):
        """화자 관련 데이터 초기화"""
        st.session_state.speaker_segments = None
        st.session_state.recognized_segments = None
        st.session_state.segment_profiles = None
        st.session_state.recognition_completed = False
        st.session_state.selected_speaker = None
    
    def clear_all(self):
        """모든 상태 초기화"""
        keys_to_keep = ['gemini_summarizer', 'claude_summarizer', 'active_summarizer']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        self._initialize_session_state()