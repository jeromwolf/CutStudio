"""
State Management System for CutStudio Pro
중앙 집중식 상태 관리로 모든 컴포넌트 간 데이터 동기화
"""

import streamlit as st
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, field, asdict
import json
from datetime import datetime
from pathlib import Path


class StateManager:
    """
    애플리케이션 전역 상태 관리자
    
    Features:
    - Reactive state updates
    - Persistence support
    - State history tracking
    - Type-safe state access
    """
    
    def __init__(self, initial_state: Any = None):
        """
        상태 관리자 초기화
        
        Args:
            initial_state: 초기 상태 객체 (dataclass)
        """
        self._state_key = "cutstudio_state"
        self._history_key = "cutstudio_history"
        self._subscribers: List[Callable] = []
        
        # 초기 상태 설정
        if initial_state:
            self._initialize_state(initial_state)
    
    def _initialize_state(self, initial_state: Any):
        """초기 상태 설정"""
        if self._state_key not in st.session_state:
            st.session_state[self._state_key] = asdict(initial_state)
            st.session_state[self._history_key] = []
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        상태 값 가져오기
        
        Args:
            key: 상태 키
            default: 기본값
            
        Returns:
            상태 값
        """
        state = st.session_state.get(self._state_key, {})
        return state.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        단일 상태 값 설정
        
        Args:
            key: 상태 키
            value: 설정할 값
        """
        self.update({key: value})
    
    def update(self, updates: Dict[str, Any]):
        """
        여러 상태 값 한번에 업데이트
        
        Args:
            updates: 업데이트할 키-값 딕셔너리
        """
        # 현재 상태 가져오기
        state = st.session_state.get(self._state_key, {})
        
        # 히스토리에 현재 상태 저장
        self._save_to_history(state)
        
        # 상태 업데이트
        state.update(updates)
        st.session_state[self._state_key] = state
        
        # 구독자들에게 알림
        self._notify_subscribers(updates)
    
    def reset(self):
        """상태 초기화"""
        if self._state_key in st.session_state:
            del st.session_state[self._state_key]
        if self._history_key in st.session_state:
            del st.session_state[self._history_key]
    
    def get_all(self) -> Dict[str, Any]:
        """전체 상태 반환"""
        return st.session_state.get(self._state_key, {})
    
    def subscribe(self, callback: Callable):
        """
        상태 변경 구독
        
        Args:
            callback: 상태 변경 시 호출될 함수
        """
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """구독 해제"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def _notify_subscribers(self, changes: Dict[str, Any]):
        """구독자들에게 변경 사항 알림"""
        for subscriber in self._subscribers:
            try:
                subscriber(changes)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")
    
    def _save_to_history(self, state: Dict[str, Any]):
        """상태 히스토리 저장"""
        history = st.session_state.get(self._history_key, [])
        
        # 최대 10개 히스토리 유지
        if len(history) >= 10:
            history.pop(0)
        
        history.append({
            "timestamp": datetime.now().isoformat(),
            "state": state.copy()
        })
        
        st.session_state[self._history_key] = history
    
    def undo(self):
        """이전 상태로 되돌리기"""
        history = st.session_state.get(self._history_key, [])
        
        if history:
            previous = history.pop()
            st.session_state[self._state_key] = previous["state"]
            st.session_state[self._history_key] = history
    
    def save_to_file(self, filepath: Path):
        """상태를 파일로 저장"""
        state = self.get_all()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: Path):
        """파일에서 상태 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        st.session_state[self._state_key] = state
    
    def create_checkpoint(self, name: str):
        """체크포인트 생성"""
        checkpoints = st.session_state.get("checkpoints", {})
        checkpoints[name] = {
            "timestamp": datetime.now().isoformat(),
            "state": self.get_all().copy()
        }
        st.session_state["checkpoints"] = checkpoints
    
    def restore_checkpoint(self, name: str):
        """체크포인트 복원"""
        checkpoints = st.session_state.get("checkpoints", {})
        if name in checkpoints:
            st.session_state[self._state_key] = checkpoints[name]["state"]


# 싱글톤 인스턴스
_state_manager_instance = None


def get_state_manager() -> StateManager:
    """상태 관리자 싱글톤 인스턴스 반환"""
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = StateManager()
    return _state_manager_instance