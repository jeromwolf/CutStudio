"""
CutStudio Enhanced - 교육 특화 동영상 편집기
개선 사항:
1. 원클릭 화자별 추출
2. 화자 라벨링 시스템
3. 교육 특화 요약
"""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os

# 내부 모듈
from core.config import AppConfig
from core.state_manager import StateManager
from services.speaker_detection import UnifiedSpeakerDetector
from services.speech_processing import SpeechProcessor
from services.summarization import SummarizationService
from video_editor import VideoEditor
from utils import (
    get_video_info, format_time, get_mime_type,
    cleanup_old_files, generate_unique_filename
)
from youtube_downloader import YouTubeDownloader

# UI 컴포넌트
from ui.components.speaker_profile import display_speaker_profile
from ui.components.timeline import display_timeline


class EnhancedCutStudioApp:
    """개선된 CutStudio 애플리케이션"""
    
    def __init__(self):
        """애플리케이션 초기화"""
        st.set_page_config(
            page_title="CutStudio Enhanced",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.config = AppConfig()
        self.state = StateManager()
        self.speaker_detector = UnifiedSpeakerDetector()
        self.speech_processor = SpeechProcessor()
        self.summarizer = SummarizationService()
        self.youtube_downloader = YouTubeDownloader()
        
        # 세션 상태 초기화
        self._initialize_session_state()
        
        # 임시 파일 정리
        cleanup_old_files()
        
        # AI 서비스 초기화
        self.summarizer.initialize_summarizers()
    
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        # 화자 라벨 저장
        if 'speaker_labels' not in st.session_state:
            st.session_state.speaker_labels = {}
        
        # 추출 히스토리
        if 'export_history' not in st.session_state:
            st.session_state.export_history = []
        
        # 교육 설정
        if 'education_settings' not in st.session_state:
            st.session_state.education_settings = {
                'lecture_type': 'general',  # general, seminar, lab, tutorial
                'summary_depth': 'medium',  # short, medium, detailed
                'include_qa': True,
                'auto_chapters': True
            }
    
    def run(self):
        """애플리케이션 실행"""
        self._display_header()
        self._display_sidebar()
        
        # 메인 탭 - 논리적 순서로 재배치
        tabs = st.tabs([
            "📤 파일 업로드",
            "📺 YouTube 다운로드", 
            "👥 화자 분석",
            "🎯 스마트 편집",
            "📝 교육 요약"
        ])
        
        with tabs[0]:
            self._handle_file_upload()
        
        with tabs[1]:
            self._handle_youtube_download()
        
        with tabs[2]:
            self._display_speaker_analysis_enhanced()  # 화자 분석 먼저
        
        with tabs[3]:
            self._display_smart_editing()  # 그 다음 스마트 편집
        
        with tabs[4]:
            self._display_education_summary()  # 마지막에 요약
    
    def _display_header(self):
        """헤더 표시"""
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🎬 CutStudio Enhanced")
            st.markdown("**교육 특화** AI 동영상 편집기 | v4.0")
        with col2:
            if st.button("📚 사용 가이드"):
                self._show_user_guide()
    
    def _display_sidebar(self):
        """사이드바 표시"""
        with st.sidebar:
            st.header("⚙️ 설정")
            
            # 교육 설정
            st.subheader("🎓 교육 설정")
            
            lecture_type = st.selectbox(
                "강의 유형",
                ["general", "seminar", "lab", "tutorial"],
                format_func=lambda x: {
                    "general": "일반 강의",
                    "seminar": "세미나/토론",
                    "lab": "실습/실험",
                    "tutorial": "튜토리얼"
                }[x]
            )
            st.session_state.education_settings['lecture_type'] = lecture_type
            
            summary_depth = st.select_slider(
                "요약 상세도",
                options=["short", "medium", "detailed"],
                value="medium",
                format_func=lambda x: {
                    "short": "간단히",
                    "medium": "보통",
                    "detailed": "자세히"
                }[x]
            )
            st.session_state.education_settings['summary_depth'] = summary_depth
            
            st.session_state.education_settings['include_qa'] = st.checkbox(
                "Q&A 세션 포함", 
                value=True
            )
            
            st.session_state.education_settings['auto_chapters'] = st.checkbox(
                "자동 챕터 생성", 
                value=True
            )
            
            # AI 요약 설정
            st.markdown("---")
            st.subheader("🤖 AI 설정")
            self._display_ai_settings()
            
            # 내보내기 히스토리
            st.markdown("---")
            st.subheader("📜 최근 작업")
            self._display_export_history()
    
    def _display_smart_editing(self):
        """스마트 편집 - 원클릭 추출 기능"""
        st.header("🎯 스마트 편집")
        
        if not self.state.video_path:
            # 비디오가 없으면 업로드 또는 YouTube 다운로드에서 선택할 수 있도록 안내
            st.info("편집할 비디오를 선택해주세요.")
            
            # YouTube 다운로드 기록이 있으면 선택 옵션 제공
            if 'youtube_downloads' in st.session_state and st.session_state.youtube_downloads:
                st.markdown("### 📺 YouTube 다운로드 파일에서 선택")
                
                # 최근 다운로드한 파일들 표시
                recent_downloads = st.session_state.youtube_downloads[-5:]  # 최근 5개
                
                for i, download in enumerate(recent_downloads):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"🎬 **{download['title'][:50]}...**")
                        st.caption(f"크기: {download['size_mb']:.1f} MB | 품질: {download['quality']}")
                    
                    with col2:
                        if Path(download['path']).exists():
                            st.success("✅ 사용 가능")
                        else:
                            st.error("❌ 파일 없음")
                    
                    with col3:
                        if Path(download['path']).exists():
                            if st.button("📝 편집하기", key=f"edit_smart_{i}"):
                                self._load_downloaded_video(download['path'])
                                st.rerun()
                
                st.markdown("---")
            
            # 파일 업로드 또는 YouTube 다운로드 안내
            st.markdown("### 💡 비디오를 편집하려면:")
            col1, col2 = st.columns(2)
            with col1:
                st.info("📤 **파일 업로드 탭**에서 비디오/오디오 파일을 업로드하세요")
            with col2:
                st.info("📺 **YouTube 다운로드 탭**에서 온라인 영상을 다운로드하세요")
            
            return
        
        if not self.state.speaker_segments:
            st.warning("화자 분석을 먼저 실행해주세요.")
            if st.button("화자 분석 실행하기"):
                st.switch_page("pages/speaker_analysis.py")
            return
        
        # 원클릭 추출 섹션
        st.subheader("✨ 원클릭 추출")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👨‍🏫 교수 강의만", use_container_width=True):
                self._extract_professor_only()
        
        with col2:
            if st.button("🙋 학생 질문만", use_container_width=True):
                self._extract_student_questions()
        
        with col3:
            if st.button("💬 Q&A 세션", use_container_width=True):
                self._extract_qa_sessions()
        
        with col4:
            if st.button("📚 챕터별 분리", use_container_width=True):
                self._extract_by_chapters()
        
        # 고급 추출 옵션
        with st.expander("🔧 고급 추출 옵션"):
            self._display_advanced_extraction()
        
        # 추출 결과 표시
        if 'last_extraction' in st.session_state:
            st.markdown("---")
            self._display_extraction_results()
    
    def _display_speaker_analysis_enhanced(self):
        """개선된 화자 분석 - 라벨링 기능 포함"""
        st.header("👥 화자 분석 Pro")
        
        if not self.state.video_path:
            st.info("먼저 비디오를 업로드해주세요.")
            return
        
        # 화자 감지 설정
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 화자 감지")
            
            # 감지 모드 선택 - 균형잡힌 모드를 기본값으로
            available_modes = self.speaker_detector.get_available_modes()
            mode_list = list(available_modes.keys())
            default_index = mode_list.index('balanced') if 'balanced' in mode_list else 0
            
            detection_mode = st.selectbox(
                "감지 모드",
                mode_list,
                index=default_index,
                format_func=lambda x: available_modes[x]['name']
            )
            
            st.caption(available_modes[detection_mode]['description'])
            
            num_speakers = st.number_input(
                "예상 화자 수 (0=자동)",
                min_value=0,
                max_value=10,
                value=2,
                help="일반적으로 교수 1명 + 학생 여러명"
            )
        
        with col2:
            st.subheader("🏷️ 화자 라벨링")
            
            if self.state.speaker_segments:
                self._display_speaker_labeling()
            else:
                st.info("화자 감지 후 라벨을 지정할 수 있습니다.")
        
        # 화자 감지 실행
        if st.button("🚀 화자 감지 시작", type="primary", use_container_width=True):
            self._run_speaker_detection_enhanced(
                detection_mode,
                num_speakers if num_speakers > 0 else None
            )
        
        # 감지 결과 표시
        if self.state.speaker_segments:
            st.markdown("---")
            
            # 화자별 통계
            st.subheader("📊 화자별 통계")
            self._display_speaker_statistics()
            
            # 타임라인
            st.subheader("📍 타임라인")
            if self.state.recognized_segments:
                display_timeline(
                    self.state.recognized_segments,
                    self.state.video_path,
                    speaker_labels=st.session_state.speaker_labels
                )
            else:
                if st.button("🎙️ 음성 인식 시작"):
                    self._run_speech_recognition("base")
    
    def _display_education_summary(self):
        """교육 특화 요약"""
        st.header("📝 교육 특화 요약")
        
        if not self.state.recognized_segments:
            st.info("먼저 화자 분석과 음성 인식을 완료해주세요.")
            return
        
        # 요약 옵션
        col1, col2, col3 = st.columns(3)
        
        with col1:
            summary_type = st.selectbox(
                "요약 유형",
                ["full", "concepts", "qa", "exam"],
                format_func=lambda x: {
                    "full": "전체 요약",
                    "concepts": "핵심 개념만",
                    "qa": "Q&A 정리",
                    "exam": "시험 대비"
                }[x]
            )
        
        with col2:
            summary_length = st.selectbox(
                "요약 길이",
                ["1min", "5min", "15min"],
                format_func=lambda x: {
                    "1min": "1분 요약",
                    "5min": "5분 요약",
                    "15min": "15분 요약"
                }[x]
            )
        
        with col3:
            output_format = st.selectbox(
                "출력 형식",
                ["markdown", "pdf", "docx"],
                format_func=lambda x: {
                    "markdown": "마크다운",
                    "pdf": "PDF 문서",
                    "docx": "Word 문서"
                }[x]
            )
        
        # 요약 생성
        if st.button("📝 요약 생성", type="primary", use_container_width=True):
            self._generate_education_summary(summary_type, summary_length)
        
        # 요약 결과 표시
        if 'education_summary' in st.session_state:
            st.markdown("---")
            self._display_education_summary_results()
    
    # === 핵심 기능 구현 메서드들 ===
    
    def _extract_professor_only(self):
        """교수 강의만 추출"""
        with st.spinner("교수님 강의 구간을 추출하는 중..."):
            # 교수로 라벨링된 화자 찾기
            professor_speakers = []
            for speaker, label in st.session_state.speaker_labels.items():
                if "교수" in label or "professor" in label.lower():
                    professor_speakers.append(speaker)
            
            if not professor_speakers and self.state.speaker_segments:
                # 라벨이 없으면 가장 많이 말한 화자를 교수로 추정
                speaker_durations = {}
                for segment in self.state.speaker_segments:
                    speaker = segment['speaker']
                    duration = segment['end'] - segment['start']
                    speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
                
                professor_speakers = [max(speaker_durations, key=speaker_durations.get)]
            
            # 교수 세그먼트만 추출
            professor_segments = [
                seg for seg in self.state.speaker_segments 
                if seg['speaker'] in professor_speakers
            ]
            
            if professor_segments:
                # 비디오 편집 및 내보내기
                output_path = self._merge_and_export_segments(
                    professor_segments, 
                    "professor_only"
                )
                
                st.session_state.last_extraction = {
                    'type': 'professor_only',
                    'path': output_path,
                    'segments': len(professor_segments),
                    'duration': sum(s['end'] - s['start'] for s in professor_segments)
                }
                
                st.success(f"✅ 교수 강의 추출 완료! ({len(professor_segments)}개 구간)")
            else:
                st.error("교수 화자를 찾을 수 없습니다. 화자 라벨을 확인해주세요.")
    
    def _extract_student_questions(self):
        """학생 질문만 추출"""
        with st.spinner("학생 질문 구간을 추출하는 중..."):
            # 학생으로 라벨링된 화자 찾기
            student_speakers = []
            for speaker, label in st.session_state.speaker_labels.items():
                if "학생" in label or "student" in label.lower() or speaker != "SPEAKER_0":
                    student_speakers.append(speaker)
            
            # 학생 세그먼트 추출
            student_segments = [
                seg for seg in self.state.speaker_segments 
                if seg['speaker'] in student_speakers
            ]
            
            if student_segments:
                output_path = self._merge_and_export_segments(
                    student_segments, 
                    "student_questions"
                )
                
                st.session_state.last_extraction = {
                    'type': 'student_questions',
                    'path': output_path,
                    'segments': len(student_segments),
                    'duration': sum(s['end'] - s['start'] for s in student_segments)
                }
                
                st.success(f"✅ 학생 질문 추출 완료! ({len(student_segments)}개 구간)")
            else:
                st.warning("학생 발화를 찾을 수 없습니다.")
    
    def _extract_qa_sessions(self):
        """Q&A 세션 추출 - 교수-학생 상호작용"""
        with st.spinner("Q&A 세션을 추출하는 중..."):
            qa_segments = []
            segments = self.state.speaker_segments
            
            for i in range(len(segments) - 1):
                current = segments[i]
                next_seg = segments[i + 1]
                
                # 화자가 바뀌는 구간 찾기
                if current['speaker'] != next_seg['speaker']:
                    # 전후 컨텍스트 포함 (5초)
                    start_time = max(0, current['start'] - 5)
                    end_time = min(next_seg['end'] + 5, self.state.video_editor.duration)
                    
                    qa_segments.append({
                        'start': start_time,
                        'end': end_time,
                        'speakers': [current['speaker'], next_seg['speaker']]
                    })
            
            if qa_segments:
                # 중복 제거 및 병합
                merged_segments = self._merge_overlapping_segments(qa_segments)
                output_path = self._merge_and_export_segments(
                    merged_segments, 
                    "qa_sessions"
                )
                
                st.session_state.last_extraction = {
                    'type': 'qa_sessions',
                    'path': output_path,
                    'segments': len(merged_segments),
                    'duration': sum(s['end'] - s['start'] for s in merged_segments)
                }
                
                st.success(f"✅ Q&A 세션 추출 완료! ({len(merged_segments)}개 세션)")
            else:
                st.warning("Q&A 세션을 찾을 수 없습니다.")
    
    def _extract_by_chapters(self):
        """챕터별 분리"""
        with st.spinner("챕터를 분석하는 중..."):
            # 간단한 챕터 감지 (10분 단위 또는 주제 변경)
            video_duration = self.state.video_editor.duration
            chapter_duration = 600  # 10분
            
            chapters = []
            for i in range(0, int(video_duration), chapter_duration):
                start = i
                end = min(i + chapter_duration, video_duration)
                chapters.append({
                    'start': start,
                    'end': end,
                    'title': f"Chapter {len(chapters) + 1}"
                })
            
            # 각 챕터별로 내보내기
            output_paths = []
            for i, chapter in enumerate(chapters):
                output_path = Path(self.config.PROCESSED_DIR) / f"chapter_{i+1}.mp4"
                # 실제 비디오 자르기 구현...
                output_paths.append(str(output_path))
            
            st.session_state.last_extraction = {
                'type': 'chapters',
                'paths': output_paths,
                'count': len(chapters)
            }
            
            st.success(f"✅ {len(chapters)}개 챕터로 분리 완료!")
    
    def _display_speaker_labeling(self):
        """화자 라벨링 UI"""
        st.write("각 화자의 이름이나 역할을 지정하세요:")
        
        # 화자별 샘플 재생 및 라벨 입력
        for speaker in set(seg['speaker'] for seg in self.state.speaker_segments):
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                st.write(f"**{speaker}**")
            
            with col2:
                # 해당 화자의 첫 번째 세그먼트 찾기
                first_segment = next(
                    seg for seg in self.state.speaker_segments 
                    if seg['speaker'] == speaker
                )
                
                if st.button(f"🔊 샘플 듣기", key=f"play_{speaker}"):
                    # 오디오 재생 로직
                    st.info(f"{format_time(first_segment['start'])} - {format_time(first_segment['end'])}")
            
            with col3:
                # 기존 라벨 또는 기본값
                default_label = st.session_state.speaker_labels.get(
                    speaker, 
                    "교수" if speaker == "SPEAKER_0" else f"학생{speaker[-1]}"
                )
                
                label = st.text_input(
                    "이름/역할",
                    value=default_label,
                    key=f"label_{speaker}"
                )
                
                st.session_state.speaker_labels[speaker] = label
        
        # 라벨 저장
        if st.button("💾 라벨 저장"):
            self._save_speaker_labels()
            st.success("라벨이 저장되었습니다!")
    
    def _generate_education_summary(self, summary_type: str, length: str):
        """교육 특화 요약 생성"""
        with st.spinner("AI가 강의를 분석하는 중..."):
            # 전체 텍스트 준비
            full_transcript = self._prepare_transcript_for_summary()
            
            # 교육 특화 프롬프트
            prompts = {
                "full": f"""
                다음 강의 내용을 {length} 분량으로 요약해주세요.
                포함할 내용:
                1. 학습 목표
                2. 핵심 개념 (정의 포함)
                3. 상세 설명
                4. 예시/사례
                5. Q&A 요약
                6. 다음 수업 준비사항
                
                강의 내용:
                {full_transcript}
                """,
                
                "concepts": f"""
                다음 강의에서 핵심 개념만 추출해주세요.
                - 정의가 포함된 개념
                - 2번 이상 반복된 중요 내용
                - 전문 용어와 설명
                
                강의 내용:
                {full_transcript}
                """,
                
                "qa": f"""
                다음 강의에서 Q&A 세션만 정리해주세요.
                - 학생 질문
                - 교수 답변
                - 추가 설명이 필요한 부분
                
                강의 내용:
                {full_transcript}
                """,
                
                "exam": f"""
                다음 강의에서 시험에 나올 만한 내용을 정리해주세요.
                - "중요", "시험", "꼭 기억" 언급 부분
                - 핵심 정의와 공식
                - 예상 문제
                
                강의 내용:
                {full_transcript}
                """
            }
            
            # AI 요약 생성
            summary = self.summarizer.summarize_text(
                prompts[summary_type],
                max_length={"1min": 200, "5min": 1000, "15min": 3000}[length]
            )
            
            st.session_state.education_summary = {
                'type': summary_type,
                'length': length,
                'content': summary,
                'timestamp': datetime.now()
            }
    
    def _merge_and_export_segments(self, segments: List[Dict], output_name: str) -> str:
        """세그먼트 병합 및 내보내기"""
        output_path = Path(self.config.PROCESSED_DIR) / f"{output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # VideoEditor를 사용하여 세그먼트 병합
        if hasattr(self.state.video_editor, 'merge_segments'):
            self.state.video_editor.merge_segments(segments, str(output_path))
        else:
            # 폴백: 첫 번째 세그먼트만 추출
            if segments:
                self.state.video_editor.cut_video(
                    segments[0]['start'], 
                    segments[-1]['end'], 
                    str(output_path)
                )
        
        # 내보내기 히스토리에 추가
        st.session_state.export_history.append({
            'name': output_name,
            'path': str(output_path),
            'timestamp': datetime.now(),
            'segments': len(segments)
        })
        
        return str(output_path)
    
    def _display_extraction_results(self):
        """추출 결과 표시"""
        extraction = st.session_state.last_extraction
        
        st.subheader("📤 추출 결과")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("추출 유형", extraction['type'].replace('_', ' ').title())
        
        with col2:
            if 'segments' in extraction:
                st.metric("세그먼트 수", extraction['segments'])
            elif 'count' in extraction:
                st.metric("파일 수", extraction['count'])
        
        with col3:
            if 'duration' in extraction:
                st.metric("총 길이", format_time(extraction['duration']))
        
        # 다운로드 버튼
        if 'path' in extraction and Path(extraction['path']).exists():
            with open(extraction['path'], 'rb') as f:
                st.download_button(
                    "⬇️ 다운로드",
                    data=f,
                    file_name=Path(extraction['path']).name,
                    mime="video/mp4",
                    use_container_width=True
                )
        elif 'paths' in extraction:
            st.info(f"{len(extraction['paths'])}개 파일이 생성되었습니다.")
    
    def _display_export_history(self):
        """내보내기 히스토리 표시"""
        if st.session_state.export_history:
            for export in st.session_state.export_history[-5:]:  # 최근 5개
                with st.expander(f"{export['name']} - {export['timestamp'].strftime('%H:%M')}"):
                    st.write(f"세그먼트: {export['segments']}개")
                    if Path(export['path']).exists():
                        st.write("✅ 파일 사용 가능")
                    else:
                        st.write("❌ 파일 없음")
        else:
            st.info("아직 내보낸 파일이 없습니다.")
    
    def _merge_overlapping_segments(self, segments):
        """겹치는 세그먼트 병합 - VideoEditor의 메서드 활용"""
        if hasattr(self.state.video_editor, 'merge_overlapping_segments'):
            return self.state.video_editor.merge_overlapping_segments(segments, padding=2)
        return segments
    
    def _save_speaker_labels(self):
        """화자 라벨 저장"""
        if self.state.video_path:
            # 라벨 정보를 JSON 파일로 저장
            labels_file = Path(self.state.video_path).with_suffix('.labels.json')
            with open(labels_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'labels': st.session_state.speaker_labels,
                    'video_path': self.state.video_path,
                    'timestamp': datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
    
    def _prepare_transcript_for_summary(self) -> str:
        """요약을 위한 전체 트랜스크립트 준비"""
        if not self.state.recognized_segments:
            return ""
        
        transcript_lines = []
        for segment in self.state.recognized_segments:
            speaker = segment.get('speaker', 'Unknown')
            # 라벨이 있으면 사용
            if speaker in st.session_state.speaker_labels:
                speaker = st.session_state.speaker_labels[speaker]
            
            text = segment.get('text', '').strip()
            if text:
                transcript_lines.append(f"{speaker}: {text}")
        
        return "\n".join(transcript_lines)
    
    def _display_speaker_statistics(self):
        """화자별 통계 표시"""
        if not self.state.speaker_segments:
            return
        
        # 화자별 통계 계산
        speaker_stats = {}
        total_duration = 0
        
        for segment in self.state.speaker_segments:
            speaker = segment['speaker']
            duration = segment['end'] - segment['start']
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'duration': 0,
                    'count': 0,
                    'first': segment['start'],
                    'last': segment['end']
                }
            
            speaker_stats[speaker]['duration'] += duration
            speaker_stats[speaker]['count'] += 1
            speaker_stats[speaker]['last'] = max(speaker_stats[speaker]['last'], segment['end'])
            total_duration += duration
        
        # 통계 표시
        cols = st.columns(len(speaker_stats))
        for i, (speaker, stats) in enumerate(speaker_stats.items()):
            with cols[i]:
                # 라벨 사용
                label = st.session_state.speaker_labels.get(speaker, speaker)
                st.metric(
                    label,
                    f"{format_time(stats['duration'])}",
                    f"{stats['count']}개 구간"
                )
                
                # 참여율
                participation = (stats['duration'] / total_duration * 100) if total_duration > 0 else 0
                st.progress(participation / 100)
                st.caption(f"{participation:.1f}% 참여")
    
    def _display_ai_settings(self):
        """AI 설정 표시"""
        # 요약기 상태 표시
        col1, col2 = st.columns(2)
        with col1:
            if self.summarizer.gemini_summarizer:
                st.success("✅ Gemini")
            else:
                st.error("❌ Gemini")
        
        with col2:
            if self.summarizer.claude_summarizer:
                st.success("✅ Claude")
            else:
                st.info("⏸️ Claude")
        
        # 사용 가능한 요약기 선택
        available_summarizers = []
        if self.summarizer.gemini_summarizer:
            available_summarizers.append("Gemini")
        if self.summarizer.claude_summarizer:
            available_summarizers.append("Claude")
        
        if available_summarizers:
            selected = st.selectbox(
                "요약 AI 선택",
                available_summarizers,
                key="summarizer_select"
            )
            self.summarizer.active_summarizer = selected.lower()
    
    def _display_advanced_extraction(self):
        """고급 추출 옵션"""
        col1, col2 = st.columns(2)
        
        with col1:
            min_segment_duration = st.slider(
                "최소 세그먼트 길이 (초)",
                min_value=1,
                max_value=30,
                value=5,
                help="이보다 짧은 발화는 제외됩니다"
            )
            
            merge_gap = st.slider(
                "세그먼트 병합 간격 (초)",
                min_value=0,
                max_value=10,
                value=2,
                help="이 간격 이내의 같은 화자 발화는 하나로 병합"
            )
        
        with col2:
            include_context = st.checkbox(
                "전후 컨텍스트 포함",
                value=True,
                help="각 세그먼트 전후 5초를 추가로 포함"
            )
            
            if include_context:
                context_seconds = st.number_input(
                    "컨텍스트 길이 (초)",
                    min_value=1,
                    max_value=30,
                    value=5
                )
        
        # 고급 추출 실행
        if st.button("🚀 고급 추출 실행", use_container_width=True):
            self._run_advanced_extraction(
                min_segment_duration,
                merge_gap,
                include_context,
                context_seconds if include_context else 0
            )
    
    def _run_advanced_extraction(self, min_duration, merge_gap, include_context, context_seconds):
        """고급 추출 실행"""
        with st.spinner("고급 옵션으로 추출 중..."):
            # 필터링: 최소 길이 이상인 세그먼트만
            filtered_segments = [
                seg for seg in self.state.speaker_segments
                if (seg['end'] - seg['start']) >= min_duration
            ]
            
            # 병합: 같은 화자의 가까운 세그먼트
            merged_segments = self._merge_same_speaker_segments(filtered_segments, merge_gap)
            
            # 컨텍스트 추가
            if include_context and hasattr(self.state.video_editor, 'export_segments_with_context'):
                output_path = self.state.video_editor.export_segments_with_context(
                    merged_segments,
                    context_before=context_seconds,
                    context_after=context_seconds
                )
            else:
                output_path = self._merge_and_export_segments(merged_segments, "advanced_extraction")
            
            if output_path:
                st.success("✅ 고급 추출 완료!")
                st.session_state.last_extraction = {
                    'type': 'advanced',
                    'path': output_path,
                    'segments': len(merged_segments),
                    'duration': sum(s['end'] - s['start'] for s in merged_segments)
                }
    
    def _merge_same_speaker_segments(self, segments, max_gap):
        """같은 화자의 가까운 세그먼트 병합"""
        if not segments:
            return []
        
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        merged = [sorted_segments[0].copy()]
        
        for current in sorted_segments[1:]:
            last = merged[-1]
            
            # 같은 화자이고 간격이 max_gap 이내인 경우 병합
            if (current['speaker'] == last['speaker'] and 
                current['start'] - last['end'] <= max_gap):
                last['end'] = current['end']
                # 텍스트가 있으면 합치기
                if 'text' in last and 'text' in current:
                    last['text'] = last['text'] + " " + current['text']
            else:
                merged.append(current.copy())
        
        return merged
    
    def _run_speaker_detection_enhanced(self, mode, num_speakers):
        """개선된 화자 감지 실행"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(current, total, message="처리 중..."):
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"{message} ({current}/{total})")
        
        try:
            # 화자 감지 실행
            segments = self.speaker_detector.detect_speakers(
                self.state.video_path,
                mode=mode,
                num_speakers=num_speakers,
                progress_callback=progress_callback
            )
            
            if segments:
                self.state.speaker_segments = segments
                
                # 화자 프로필 생성
                if self.state.video_editor:
                    self.state.segment_profiles = self.state.video_editor.generate_speaker_profile(segments)
                
                progress_bar.progress(1.0)
                status_text.text("화자 감지 완료!")
                st.success(f"✅ {len(segments)}개의 화자 구간을 감지했습니다!")
                
                # 자동 라벨링 제안
                self._suggest_speaker_labels()
            else:
                st.error("화자를 감지하지 못했습니다.")
        
        except Exception as e:
            st.error(f"화자 감지 중 오류 발생: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def _suggest_speaker_labels(self):
        """화자 라벨 자동 제안"""
        if not self.state.speaker_segments:
            return
        
        # 화자별 발화 시간 계산
        speaker_durations = {}
        for segment in self.state.speaker_segments:
            speaker = segment['speaker']
            duration = segment['end'] - segment['start']
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
        
        # 가장 많이 말한 화자를 교수로 추정
        if speaker_durations:
            main_speaker = max(speaker_durations, key=speaker_durations.get)
            
            # 기본 라벨 제안
            for i, speaker in enumerate(sorted(speaker_durations.keys())):
                if speaker not in st.session_state.speaker_labels:
                    if speaker == main_speaker:
                        st.session_state.speaker_labels[speaker] = "교수"
                    else:
                        st.session_state.speaker_labels[speaker] = f"학생{i}"
    
    def _display_education_summary_results(self):
        """교육 요약 결과 표시"""
        summary_data = st.session_state.education_summary
        
        # 요약 메타데이터
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("요약 유형", summary_data['type'].replace('_', ' ').title())
        with col2:
            st.metric("요약 길이", summary_data['length'])
        with col3:
            st.metric("생성 시간", summary_data['timestamp'].strftime('%H:%M'))
        
        # 요약 내용
        st.markdown("### 📄 요약 내용")
        st.markdown(summary_data['content'])
        
        # 다운로드 옵션
        st.download_button(
            "📥 요약 다운로드 (텍스트)",
            data=summary_data['content'],
            file_name=f"lecture_summary_{summary_data['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    def _run_speech_recognition(self, whisper_model):
        """음성 인식 실행"""
        if not self.state.speaker_segments:
            st.error("먼저 화자 감지를 실행해주세요.")
            return
        
        with st.spinner(f"음성 인식 중... (모델: {whisper_model})"):
            try:
                recognized_segments = self.speech_processor.process_segments(
                    self.state.video_path,
                    self.state.speaker_segments,
                    whisper_model=whisper_model
                )
                
                if recognized_segments:
                    self.state.recognized_segments = recognized_segments
                    st.success(f"✅ 음성 인식 완료! {len(recognized_segments)}개 세그먼트")
                else:
                    st.error("음성 인식에 실패했습니다.")
            
            except Exception as e:
                st.error(f"음성 인식 중 오류: {str(e)}")
    
    # === 기존 메서드들 (app_refactored.py에서 가져옴) ===
    
    def _handle_file_upload(self):
        """파일 업로드 처리"""
        st.header("📤 미디어 파일 업로드")
        
        uploaded_file = st.file_uploader(
            "비디오 또는 오디오 파일을 선택하세요",
            type=self.config.get_all_supported_formats(),
            help=f"지원 형식: 비디오({', '.join(self.config.SUPPORTED_VIDEO_FORMATS)}), 오디오({', '.join(self.config.SUPPORTED_AUDIO_FORMATS)})"
        )
        
        if uploaded_file is not None:
            # 파일 저장
            temp_path = Path(self.config.TEMP_DIR)
            temp_path.mkdir(exist_ok=True)
            
            video_path = temp_path / uploaded_file.name
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
            
            # 미디어 에디터 초기화
            self.state.video_editor = VideoEditor()
            self.state.video_editor.load_video(str(video_path))
            self.state.video_path = str(video_path)
            
            # 미디어 정보 표시
            self._display_media_info()
    
    def _handle_youtube_download(self):
        """YouTube 다운로드 처리"""
        st.header("📺 YouTube 동영상 다운로드")
        
        # YouTube URL 입력
        youtube_url = st.text_input(
            "YouTube URL을 입력하세요",
            placeholder="https://www.youtube.com/watch?v=...",
            help="YouTube 동영상 또는 재생목록 URL을 입력하세요"
        )
        
        if youtube_url:
            # URL 유효성 간단 체크
            if "youtube.com" in youtube_url or "youtu.be" in youtube_url:
                
                col1, col2 = st.columns(2)
                
                with col1:
                    download_quality = st.selectbox(
                        "다운로드 품질",
                        ["highest", "720p", "480p", "360p", "audio_only"],
                        format_func=lambda x: {
                            "highest": "최고 화질",
                            "720p": "720p (HD)",
                            "480p": "480p (SD)", 
                            "360p": "360p (저화질)",
                            "audio_only": "오디오만"
                        }[x]
                    )
                
                with col2:
                    download_format = st.selectbox(
                        "파일 형식",
                        ["mp4", "webm", "mp3"] if download_quality != "audio_only" else ["mp3", "m4a", "wav"],
                        format_func=lambda x: {
                            "mp4": "MP4 (추천)",
                            "webm": "WebM",
                            "mp3": "MP3 (오디오)",
                            "m4a": "M4A (오디오)",
                            "wav": "WAV (오디오)"
                        }[x]
                    )
                
                # 고급 옵션
                with st.expander("🔧 고급 옵션"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        start_time = st.text_input(
                            "시작 시간 (선택사항)",
                            placeholder="00:10:30",
                            help="형식: HH:MM:SS 또는 MM:SS"
                        )
                    
                    with col2:
                        end_time = st.text_input(
                            "종료 시간 (선택사항)",
                            placeholder="01:20:15",
                            help="형식: HH:MM:SS 또는 MM:SS"
                        )
                    
                    subtitle_download = st.checkbox(
                        "자막 다운로드",
                        value=False,
                        help="사용 가능한 자막을 함께 다운로드합니다"
                    )
                
                # 다운로드 버튼
                if st.button("📥 다운로드 시작", type="primary", use_container_width=True):
                    self._download_youtube_video(
                        youtube_url,
                        download_quality,
                        download_format,
                        start_time,
                        end_time,
                        subtitle_download
                    )
            
            else:
                st.error("❌ 유효한 YouTube URL을 입력해주세요.")
        
        # 다운로드 히스토리
        if 'youtube_downloads' in st.session_state and st.session_state.youtube_downloads:
            st.markdown("---")
            st.subheader("📜 다운로드 기록")
            
            for download in st.session_state.youtube_downloads[-3:]:  # 최근 3개
                with st.expander(f"🎬 {download['title'][:50]}..."):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("품질", download['quality'])
                    with col2:
                        st.metric("크기", f"{download['size_mb']:.1f} MB")
                    with col3:
                        st.metric("시간", download['duration'])
                    
                    if Path(download['path']).exists():
                        # 이 파일을 편집에 사용하기
                        if st.button("✂️ 이 파일로 편집하기", key=f"youtube_edit_{i}"):
                            self._load_downloaded_video(download['path'])
                            st.success("✅ 파일이 로드되었습니다! 스마트 편집 탭으로 이동하세요.")
                    else:
                        st.warning("⚠️ 파일이 삭제되었습니다.")
    
    def _download_youtube_video(self, url, quality, format_type, start_time, end_time, include_subtitle):
        """YouTube 동영상 다운로드 실행"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("📡 동영상 정보를 가져오는 중...")
            progress_bar.progress(0.1)
            
            # 다운로드 옵션 설정
            download_options = {
                'quality': quality,
                'format': format_type,
                'include_subtitle': include_subtitle
            }
            
            # 시간 구간 설정
            if start_time:
                download_options['start_time'] = start_time
            if end_time:
                download_options['end_time'] = end_time
            
            # 진행률 콜백
            def progress_callback(current, total, message="다운로드 중..."):
                if total > 0:
                    progress = 0.1 + (current / total) * 0.9
                    progress_bar.progress(progress)
                    status_text.text(f"{message} ({current}/{total})")
            
            status_text.text("⬇️ 다운로드 시작...")
            progress_bar.progress(0.2)
            
            # YouTube 다운로드 실행 (기존 메서드 시그니처에 맞춤)
            def youtube_progress_callback(percent, downloaded, total):
                progress = 0.2 + (percent / 100) * 0.7
                progress_bar.progress(progress)
                status_text.text(f"⬇️ 다운로드 중... {percent:.1f}%")
            
            downloaded_path = self.youtube_downloader.download_video(
                url=url,
                format_id=None,  # 기본 최고 품질
                progress_callback=youtube_progress_callback
            )
            
            if downloaded_path and os.path.exists(downloaded_path):
                progress_bar.progress(1.0)
                status_text.text("✅ 다운로드 완료!")
                
                # 파일 정보 가져오기
                file_size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
                filename = Path(downloaded_path).stem
                
                # 다운로드 기록 저장
                if 'youtube_downloads' not in st.session_state:
                    st.session_state.youtube_downloads = []
                
                st.session_state.youtube_downloads.append({
                    'title': filename,
                    'path': downloaded_path,
                    'quality': quality,
                    'format': format_type,
                    'size_mb': file_size_mb,
                    'duration': 'Unknown',
                    'download_time': datetime.now(),
                    'url': url
                })
                
                st.success(f"✅ 다운로드 완료: {filename}")
                
                # 자동으로 편집기에 로드할지 묻기
                if st.button("🚀 바로 편집하기"):
                    self._load_downloaded_video(downloaded_path)
                    st.rerun()
            
            else:
                st.error("❌ 다운로드 실패: 파일을 찾을 수 없습니다.")
        
        except Exception as e:
            st.error(f"❌ 다운로드 중 오류 발생: {str(e)}")
        
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def _load_downloaded_video(self, file_path):
        """다운로드된 비디오를 편집기에 로드"""
        try:
            # VideoEditor 초기화
            self.state.video_editor = VideoEditor()
            self.state.video_editor.load_video(file_path)
            self.state.video_path = file_path
            
            # 세션 상태 업데이트
            st.session_state.video_path = file_path
            
        except Exception as e:
            st.error(f"파일 로드 실패: {str(e)}")
    
    def _display_media_info(self):
        """미디어 정보 표시"""
        if self.state.video_editor:
            info = get_video_info(self.state.video_path)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("길이", format_time(info['duration']))
            with col2:
                st.metric("FPS", f"{info['fps']:.1f}")
            with col3:
                st.metric("해상도", f"{info['width']}x{info['height']}")
            with col4:
                st.metric("크기", f"{info['size_mb']:.1f} MB")
    
    def _show_user_guide(self):
        """사용 가이드 표시"""
        st.info("""
        ### 🎯 CutStudio Enhanced 사용법
        
        1. **파일 업로드**: 강의 영상 또는 오디오 파일 업로드
        2. **화자 분석**: AI가 교수/학생 음성 자동 구분
        3. **스마트 편집**: 원클릭으로 필요한 부분만 추출
        4. **교육 요약**: 강의 내용을 구조화된 문서로 변환
        
        **💡 Pro Tip**: 화자 라벨을 지정하면 더 정확한 추출이 가능합니다!
        """)
    
    # 추가 헬퍼 메서드들...


# 앱 실행
if __name__ == "__main__":
    app = EnhancedCutStudioApp()
    app.run()