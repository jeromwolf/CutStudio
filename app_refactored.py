"""
CutStudio - 리팩토링된 메인 애플리케이션
"""
import streamlit as st
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프록시 관련 환경 변수 정리 (Claude 초기화 문제 방지)
proxy_vars = [
    'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
    'FTP_PROXY', 'ftp_proxy', 'NO_PROXY', 'no_proxy',
    'ALL_PROXY', 'all_proxy'
]
for var in proxy_vars:
    os.environ.pop(var, None)

# Core 모듈
from core.state_manager import AppState
from core.config import AppConfig

# Services 모듈
from services.speaker_detection import UnifiedSpeakerDetector
from services.speech_processing import SpeechProcessor
from services.summarization import SummarizationService

# UI 모듈
from ui.components import display_speaker_profile, display_timeline, display_statistics

# 기존 모듈
from video_editor import VideoEditor
from youtube_downloader import YouTubeDownloader
from utils import format_time


# 페이지 설정
st.set_page_config(
    page_title="🎬 CutStudio - 스마트 동영상 편집기",
    page_icon="🎬",
    layout="wide"
)


class CutStudioApp:
    """CutStudio 메인 애플리케이션 클래스"""
    
    def __init__(self):
        self.state = AppState()
        self.config = AppConfig()
        self.speaker_detector = UnifiedSpeakerDetector()
        self.speech_processor = None
        self.summarizer = SummarizationService()
        self.youtube_downloader = YouTubeDownloader()
        
        # 요약기 초기화
        self.summarizer.initialize_summarizers()
    
    def run(self):
        """애플리케이션 실행"""
        self._display_header()
        self._display_sidebar()
        
        # 메인 탭
        tabs = st.tabs([
            "📤 파일 업로드",
            "📺 YouTube 다운로드",
            "✂️ 편집 도구"
        ])
        
        with tabs[0]:
            self._handle_file_upload()
        
        with tabs[1]:
            self._handle_youtube_download()
        
        with tabs[2]:
            self._display_editing_tools()
    
    def _display_header(self):
        """헤더 표시"""
        st.title("🎬 CutStudio v3.1")
        st.markdown("AI 기반 스마트 동영상 편집기")
    
    def _display_sidebar(self):
        """사이드바 표시"""
        with st.sidebar:
            st.header("⚙️ 설정")
            
            # 요약기 상태 및 선택
            st.subheader("AI 요약 설정")
            
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
                    st.info("⏸️ Claude (비활성화)")
            
            # 사용 가능한 요약기 선택
            available_summarizers = []
            if self.summarizer.gemini_summarizer:
                available_summarizers.append("Gemini")
            if self.summarizer.claude_summarizer:
                available_summarizers.append("Claude")
            
            if available_summarizers:
                selected = st.selectbox(
                    "요약 AI 선택",
                    available_summarizers
                )
                self.summarizer.active_summarizer = selected.lower()
                st.info(f"현재 사용 중: {selected}")
            else:
                st.warning("⚠️ AI 요약기가 초기화되지 않았습니다.")
                st.info("API 키를 확인하고 앱을 재시작해주세요.")
            
            # 파일 관리
            st.markdown("---")
            st.subheader("🗑️ 파일 관리")
            if st.button("임시 파일 정리"):
                self._cleanup_temp_files()
    
    def _handle_file_upload(self):
        """파일 업로드 처리"""
        st.header("📤 비디오 파일 업로드")
        
        uploaded_file = st.file_uploader(
            "비디오 파일을 선택하세요",
            type=self.config.SUPPORTED_VIDEO_FORMATS
        )
        
        if uploaded_file is not None:
            # 파일 저장
            temp_path = Path(self.config.TEMP_DIR)
            temp_path.mkdir(exist_ok=True)
            
            video_path = temp_path / uploaded_file.name
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
            
            # 비디오 에디터 초기화
            self.state.video_editor = VideoEditor()
            self.state.video_editor.load_video(str(video_path))
            self.state.video_path = str(video_path)
            
            # 비디오 정보 표시
            self._display_video_info()
    
    def _handle_youtube_download(self):
        """YouTube 다운로드 처리"""
        st.header("📺 YouTube 동영상 다운로드")
        
        youtube_url = st.text_input(
            "YouTube URL을 입력하세요",
            value=st.session_state.get('youtube_url', '')
        )
        
        if st.button("다운로드", type="primary"):
            if youtube_url:
                with st.spinner("다운로드 중..."):
                    try:
                        # 비디오 정보 가져오기
                        info = self.youtube_downloader.get_video_info(youtube_url)
                        
                        if info:
                            # 다운로드 실행
                            video_path = self.youtube_downloader.download_video(youtube_url)
                            
                            if video_path:
                                st.success("✅ 다운로드 완료!")
                                
                                # 비디오 에디터 초기화
                                self.state.video_editor = VideoEditor()
                                self.state.video_editor.load_video(video_path)
                                self.state.video_path = video_path
                                
                                # 비디오 정보 표시
                                self._display_video_info()
                        else:
                            st.error("비디오 정보를 가져올 수 없습니다.")
                    
                    except Exception as e:
                        st.error(f"다운로드 실패: {str(e)}")
    
    def _display_video_info(self):
        """비디오 정보 표시"""
        if self.state.video_editor and self.state.video_editor.video_clip:
            video_clip = self.state.video_editor.video_clip
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📏 길이", format_time(video_clip.duration))
            with col2:
                st.metric("📐 해상도", f"{video_clip.w}x{video_clip.h}")
            with col3:
                st.metric("🎞️ FPS", f"{video_clip.fps:.1f}")
            
            # 미리보기
            st.video(self.state.video_path)
    
    def _display_editing_tools(self):
        """편집 도구 표시"""
        if not self.state.video_editor:
            st.info("먼저 비디오를 업로드하거나 다운로드하세요.")
            return
        
        # 편집 도구 탭
        edit_tabs = st.tabs([
            "✂️ 자르기",
            "🎨 효과",
            "🔊 오디오",
            "👥 화자 분석"
        ])
        
        with edit_tabs[0]:
            self._display_cut_tools()
        
        with edit_tabs[1]:
            self._display_effects_tools()
        
        with edit_tabs[2]:
            self._display_audio_tools()
        
        with edit_tabs[3]:
            self._display_speaker_analysis()
    
    def _display_cut_tools(self):
        """자르기 도구 표시"""
        st.header("✂️ 비디오 자르기")
        
        video_duration = self.state.video_editor.video_clip.duration
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_time = st.slider(
                "시작 시간",
                0.0,
                video_duration,
                0.0,
                format="%.1f"
            )
        
        with col2:
            end_time = st.slider(
                "종료 시간",
                0.0,
                video_duration,
                video_duration,
                format="%.1f"
            )
        
        if st.button("자르기", type="primary"):
            if start_time < end_time:
                with st.spinner("처리 중..."):
                    # 임시 파일명 생성
                    output_path = Path(self.config.PROCESSED_DIR) / "trimmed_video.mp4"
                    output_path.parent.mkdir(exist_ok=True)
                    
                    # 자르기 실행
                    self.state.video_editor.trim_video(
                        start_time,
                        end_time,
                        str(output_path)
                    )
                    
                    st.success("✅ 자르기 완료!")
                    st.video(str(output_path))
                    
                    # 다운로드 버튼
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="📥 다운로드",
                            data=f,
                            file_name="trimmed_video.mp4",
                            mime="video/mp4"
                        )
            else:
                st.error("시작 시간이 종료 시간보다 앞서야 합니다.")
    
    def _display_effects_tools(self):
        """효과 도구 표시"""
        st.header("🎨 비디오 효과")
        
        effect_type = st.selectbox(
            "효과 선택",
            ["흑백", "페이드 인/아웃", "속도 조절", "크롭"]
        )
        
        if effect_type == "흑백":
            if st.button("흑백 효과 적용"):
                with st.spinner("처리 중..."):
                    output_path = Path(self.config.PROCESSED_DIR) / "grayscale_video.mp4"
                    self.state.video_editor.apply_grayscale(str(output_path))
                    st.success("✅ 흑백 효과 적용 완료!")
                    st.video(str(output_path))
        
        elif effect_type == "페이드 인/아웃":
            fade_duration = st.slider("페이드 시간 (초)", 0.5, 5.0, 1.0)
            if st.button("페이드 효과 적용"):
                with st.spinner("처리 중..."):
                    output_path = Path(self.config.PROCESSED_DIR) / "fade_video.mp4"
                    self.state.video_editor.apply_fade(fade_duration, str(output_path))
                    st.success("✅ 페이드 효과 적용 완료!")
                    st.video(str(output_path))
        
        elif effect_type == "속도 조절":
            speed_factor = st.slider("속도 배율", 0.5, 2.0, 1.0)
            if st.button("속도 조절 적용"):
                with st.spinner("처리 중..."):
                    output_path = Path(self.config.PROCESSED_DIR) / "speed_video.mp4"
                    self.state.video_editor.change_speed(speed_factor, str(output_path))
                    st.success("✅ 속도 조절 완료!")
                    st.video(str(output_path))
        
        elif effect_type == "크롭":
            st.info("크롭 기능은 준비 중입니다.")
    
    def _display_audio_tools(self):
        """오디오 도구 표시"""
        st.header("🔊 오디오 편집")
        
        audio_option = st.selectbox(
            "오디오 옵션",
            ["볼륨 조절", "음소거", "오디오 추출"]
        )
        
        if audio_option == "볼륨 조절":
            volume_factor = st.slider("볼륨 배율", 0.0, 2.0, 1.0)
            if st.button("볼륨 조절 적용"):
                with st.spinner("처리 중..."):
                    output_path = Path(self.config.PROCESSED_DIR) / "volume_adjusted.mp4"
                    self.state.video_editor.adjust_volume(volume_factor, str(output_path))
                    st.success("✅ 볼륨 조절 완료!")
                    st.video(str(output_path))
        
        elif audio_option == "음소거":
            if st.button("음소거 적용"):
                with st.spinner("처리 중..."):
                    output_path = Path(self.config.PROCESSED_DIR) / "muted_video.mp4"
                    self.state.video_editor.remove_audio(str(output_path))
                    st.success("✅ 음소거 적용 완료!")
                    st.video(str(output_path))
        
        elif audio_option == "오디오 추출":
            if st.button("오디오 추출"):
                with st.spinner("처리 중..."):
                    output_path = Path(self.config.PROCESSED_DIR) / "extracted_audio.mp3"
                    self.state.video_editor.extract_audio(str(output_path))
                    st.success("✅ 오디오 추출 완료!")
                    st.audio(str(output_path))
    
    def _display_speaker_analysis(self):
        """화자 분석 표시"""
        st.header("👥 AI 화자 분석")
        
        # 화자 감지 설정
        st.subheader("🎯 화자 감지 설정")
        
        # 사용 가능한 모드 가져오기
        available_modes = self.speaker_detector.get_available_modes()
        
        col1, col2 = st.columns(2)
        
        with col1:
            detection_mode = st.selectbox(
                "감지 모드",
                list(available_modes.keys()),
                format_func=lambda x: available_modes[x]['name']
            )
            
            st.caption(available_modes[detection_mode]['description'])
        
        with col2:
            num_speakers = st.number_input(
                "예상 화자 수 (0=자동)",
                min_value=0,
                max_value=10,
                value=0
            )
        
        # 고급 설정
        with st.expander("⚙️ 고급 설정"):
            min_duration = st.slider(
                "최소 세그먼트 길이 (초)",
                0.5,
                5.0,
                2.0
            )
            
            whisper_model = st.selectbox(
                "음성 인식 모델",
                self.config.WHISPER_MODELS,
                index=0
            )
        
        # 화자 감지 실행
        if st.button("🚀 화자 감지 시작", type="primary"):
            self._run_speaker_detection(
                detection_mode,
                num_speakers if num_speakers > 0 else None,
                min_duration
            )
        
        # 결과 표시
        if self.state.speaker_segments:
            st.markdown("---")
            
            # 화자별 프로필
            if self.state.segment_profiles:
                st.subheader("👥 화자별 프로필")
                display_speaker_profile(self.state.segment_profiles)
            
            # 음성 인식
            if st.button("🎙️ 음성 인식 시작"):
                self._run_speech_recognition(whisper_model)
            
            # 타임라인
            if self.state.recognized_segments:
                st.markdown("---")
                display_timeline(
                    self.state.speaker_segments,
                    self.state.video_editor.video_clip,
                    self.state.recognized_segments,
                    self.summarizer.get_active_summarizer()
                )
            
            # 통계
            st.markdown("---")
            display_statistics(
                self.state.speaker_segments,
                self.state.video_editor.video_clip.duration
            )
    
    def _run_speaker_detection(self, mode: str, num_speakers: Optional[int], min_duration: float):
        """화자 감지 실행"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(current, total, message):
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(message)
        
        try:
            # 예상 시간 계산
            video_duration = self.state.video_editor.video_clip.duration
            min_time, max_time = self.speaker_detector.estimate_processing_time(
                video_duration,
                mode
            )
            
            st.info(f"예상 처리 시간: {format_time(min_time)} ~ {format_time(max_time)}")
            
            # 화자 감지 실행
            segments = self.speaker_detector.detect_speakers(
                self.state.video_path,
                mode=mode,
                num_speakers=num_speakers,
                min_duration=min_duration,
                progress_callback=progress_callback
            )
            
            if segments:
                self.state.speaker_segments = segments
                
                # 프로필 생성
                self.state.segment_profiles = self.state.video_editor.generate_speaker_profile(
                    segments
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ 화자 감지 완료!")
                st.success(f"총 {len(segments)}개의 발화 구간을 감지했습니다.")
                
                # 자동으로 음성 인식 시작 옵션
                if st.checkbox("음성 인식 자동 시작", value=True):
                    self._run_speech_recognition("tiny")
            else:
                st.error("화자를 감지하지 못했습니다.")
        
        except Exception as e:
            st.error(f"화자 감지 실패: {str(e)}")
        
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def _run_speech_recognition(self, model_size: str):
        """음성 인식 실행"""
        if not self.state.speaker_segments:
            st.error("먼저 화자 감지를 실행하세요.")
            return
        
        # SpeechProcessor 초기화 (모델 크기가 변경되면 새로 생성)
        current_model = getattr(self, '_current_model_size', None)
        if not self.speech_processor or current_model != model_size:
            try:
                from speech_transcriber import SpeechRecognizer
                speech_recognizer = SpeechRecognizer(model_size=model_size)
                self.speech_processor = SpeechProcessor(speech_recognizer)
                self._current_model_size = model_size
                st.info(f"Whisper {model_size} 모델 로딩 완료")
            except Exception as e:
                st.error(f"음성 인식 초기화 실패: {str(e)}")
                return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(current, total, message):
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(message)
        
        try:
            # 음성 인식 실행
            recognized_segments = self.speech_processor.execute_recognition(
                self.state.video_path,
                self.state.speaker_segments,
                model_size=model_size,
                progress_callback=progress_callback
            )
            
            if recognized_segments:
                self.state.recognized_segments = recognized_segments
                
                # 결과 처리
                result = self.speech_processor.process_recognition_results(
                    recognized_segments
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ 음성 인식 완료!")
                
                st.success(
                    f"총 {result['recognized_segments']}개 구간에서 "
                    f"음성을 인식했습니다. (언어: {', '.join(result['languages'])})"
                )
                
                # 대화 요약 생성
                print("🔄 대화 요약 생성 시작...")
                self._generate_conversation_summary()
            else:
                st.warning("인식된 음성이 없습니다.")
        
        except Exception as e:
            st.error(f"음성 인식 실패: {str(e)}")
        
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def _generate_conversation_summary(self):
        """대화 요약 생성"""
        print("📝 _generate_conversation_summary() 시작")
        
        if not self.state.recognized_segments:
            print("❌ recognized_segments가 없음")
            st.warning("음성 인식 결과가 없어 요약을 생성할 수 없습니다.")
            return
        
        print(f"✅ recognized_segments 개수: {len(self.state.recognized_segments)}")
        
        # 요약기 확인
        active_summarizer = self.summarizer.get_active_summarizer()
        print(f"📡 active_summarizer: {active_summarizer}")
        
        if not active_summarizer:
            print("❌ active_summarizer가 None")
            st.error("AI 요약기가 초기화되지 않았습니다. API 키를 확인해주세요.")
            return
        
        try:
            with st.spinner("AI 요약 생성 중..."):
                summary_result = self.summarizer.generate_conversation_summary(
                    self.state.recognized_segments
                )
                
                st.markdown("---")
                st.subheader("📝 AI 대화 요약")
                
                if summary_result.get('success', False):
                    # 전체 요약
                    st.markdown("### 전체 대화 요약")
                    summary_text = summary_result.get('summary', '요약을 생성할 수 없습니다.')
                    st.info(summary_text)
                    
                    # 화자별 요약
                    speaker_summaries = summary_result.get('speaker_summaries', {})
                    if speaker_summaries:
                        st.markdown("### 화자별 요약")
                        for speaker, summary in speaker_summaries.items():
                            st.markdown(f"**{speaker}**: {summary}")
                    
                    # 키워드
                    keywords = summary_result.get('keywords', [])
                    if keywords:
                        st.markdown("### 주요 키워드")
                        st.write(", ".join(keywords))
                else:
                    st.error(f"요약 생성 실패: {summary_result.get('summary', '알 수 없는 오류')}")
                    
        except Exception as e:
            st.error(f"요약 생성 중 오류 발생: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    def _cleanup_temp_files(self):
        """임시 파일 정리"""
        try:
            # temp 디렉토리 정리
            temp_path = Path(self.config.TEMP_DIR)
            if temp_path.exists():
                for file in temp_path.glob("*"):
                    file.unlink()
            
            # processed 디렉토리 정리
            processed_path = Path(self.config.PROCESSED_DIR)
            if processed_path.exists():
                for file in processed_path.glob("*"):
                    file.unlink()
            
            st.success("✅ 임시 파일이 정리되었습니다.")
        except Exception as e:
            st.error(f"파일 정리 실패: {str(e)}")


# 앱 실행
if __name__ == "__main__":
    app = CutStudioApp()
    app.run()