"""
Workflow Management System
워크플로우 관리 및 단계별 처리 로직
"""

from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import time
from pathlib import Path


class WorkflowStep(Enum):
    """워크플로우 단계"""
    IMPORT = "import"
    ANALYZE = "analyze"
    EXPORT = "export"


@dataclass
class WorkflowState:
    """워크플로우 상태"""
    current_step: WorkflowStep = WorkflowStep.IMPORT
    completed_steps: List[WorkflowStep] = None
    progress: float = 0.0
    is_processing: bool = False
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []


class WorkflowManager:
    """
    워크플로우 매니저
    
    Features:
    - Step navigation
    - Progress tracking
    - State validation
    - Error handling
    """
    
    def __init__(self):
        """워크플로우 매니저 초기화"""
        self.state = WorkflowState()
        self.callbacks: Dict[str, List[Callable]] = {
            "step_changed": [],
            "progress_updated": [],
            "error_occurred": []
        }
    
    def get_current_step(self) -> WorkflowStep:
        """현재 단계 반환"""
        return self.state.current_step
    
    def get_progress(self) -> float:
        """진행률 반환"""
        return self.state.progress
    
    def is_processing(self) -> bool:
        """처리 중인지 확인"""
        return self.state.is_processing
    
    def move_to_next_step(self):
        """다음 단계로 이동"""
        current = self.state.current_step
        
        if current == WorkflowStep.IMPORT:
            self.state.current_step = WorkflowStep.ANALYZE
        elif current == WorkflowStep.ANALYZE:
            self.state.current_step = WorkflowStep.EXPORT
        
        # 완료된 단계에 추가
        if current not in self.state.completed_steps:
            self.state.completed_steps.append(current)
        
        # 콜백 실행
        self._notify("step_changed", self.state.current_step)
    
    def move_to_step(self, step: WorkflowStep):
        """특정 단계로 이동"""
        self.state.current_step = step
        self._notify("step_changed", step)
    
    def update_progress(self, progress: float, message: str = ""):
        """진행률 업데이트"""
        self.state.progress = max(0, min(1, progress))
        self._notify("progress_updated", {
            "progress": self.state.progress,
            "message": message
        })
    
    def start_processing(self):
        """처리 시작"""
        self.state.is_processing = True
        self.state.error = None
    
    def stop_processing(self, success: bool = True, error: str = None):
        """처리 중지"""
        self.state.is_processing = False
        
        if not success and error:
            self.state.error = error
            self._notify("error_occurred", error)
    
    def is_step_completed(self, step: WorkflowStep) -> bool:
        """단계 완료 여부 확인"""
        return step in self.state.completed_steps
    
    def can_proceed_to(self, step: WorkflowStep) -> bool:
        """특정 단계로 진행 가능한지 확인"""
        if step == WorkflowStep.IMPORT:
            return True
        elif step == WorkflowStep.ANALYZE:
            return self.is_step_completed(WorkflowStep.IMPORT)
        elif step == WorkflowStep.EXPORT:
            return self.is_step_completed(WorkflowStep.ANALYZE)
        return False
    
    def reset(self):
        """워크플로우 초기화"""
        self.state = WorkflowState()
    
    def get_step_progress(self) -> Dict[str, Any]:
        """단계별 진행 상황"""
        return {
            "current": self.state.current_step.value,
            "completed": [step.value for step in self.state.completed_steps],
            "progress": self.state.progress,
            "processing": self.state.is_processing,
            "error": self.state.error
        }
    
    def subscribe(self, event: str, callback: Callable):
        """이벤트 구독"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def unsubscribe(self, event: str, callback: Callable):
        """이벤트 구독 해제"""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
    
    def _notify(self, event: str, data: Any):
        """이벤트 알림"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                print(f"Workflow callback error: {e}")
    
    def validate_step_data(self, step: WorkflowStep, data: Dict[str, Any]) -> bool:
        """단계별 데이터 검증"""
        if step == WorkflowStep.IMPORT:
            return "media_file" in data and Path(data["media_file"]).exists()
        elif step == WorkflowStep.ANALYZE:
            return "analysis_results" in data and data["analysis_results"]
        elif step == WorkflowStep.EXPORT:
            return "export_options" in data
        return False
    
    def get_next_step_requirements(self) -> Dict[str, Any]:
        """다음 단계 요구사항"""
        current = self.state.current_step
        
        if current == WorkflowStep.IMPORT:
            return {
                "step": "analyze",
                "required": ["media_file"],
                "description": "미디어 파일이 필요합니다"
            }
        elif current == WorkflowStep.ANALYZE:
            return {
                "step": "export",
                "required": ["analysis_results"],
                "description": "분석 결과가 필요합니다"
            }
        else:
            return {
                "step": "complete",
                "required": [],
                "description": "모든 단계가 완료되었습니다"
            }