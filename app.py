import streamlit as st
import tempfile
import os
from pathlib import Path
import shutil
import time
from video_editor import VideoEditor
from utils import get_video_info, format_time
from youtube_downloader import YouTubeDownloader
from speech_transcriber import SpeechRecognizer, AdvancedSpeechAnalyzer
try:
    from gemini_summarizer import GeminiSummarizer
    GEMINI_AVAILABLE = True
except Exception as e:
    print(f"Gemini 사용 불가: {e}")
    GEMINI_AVAILABLE = False

try:
    from claude_summarizer import ClaudeSummarizer
    CLAUDE_AVAILABLE = True
except Exception as e:
    print(f"Claude 사용 불가: {e}")
    CLAUDE_AVAILABLE = False

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="CutStudio - 동영상 편집기",
    page_icon="🎬",
    layout="wide"
)

if 'video_editor' not in st.session_state:
    st.session_state.video_editor = VideoEditor()

if 'youtube_downloader' not in st.session_state:
    st.session_state.youtube_downloader = YouTubeDownloader()

if 'speech_recognizer' not in st.session_state:
    st.session_state.speech_recognizer = None

if 'gemini_summarizer' not in st.session_state:
    if GEMINI_AVAILABLE:
        try:
            st.session_state.gemini_summarizer = GeminiSummarizer()
        except Exception as e:
            print(f"Gemini 초기화 실패: {e}")
            st.session_state.gemini_summarizer = None
    else:
        st.session_state.gemini_summarizer = None

if 'claude_summarizer' not in st.session_state:
    if CLAUDE_AVAILABLE:
        try:
            st.session_state.claude_summarizer = ClaudeSummarizer()
        except Exception as e:
            print(f"Claude 초기화 실패: {e}")
            st.session_state.claude_summarizer = None
    else:
        st.session_state.claude_summarizer = None

def get_summarizer():
    """사용 가능한 요약기 반환 (Gemini 우선, 실패 시 Claude)"""
    if st.session_state.gemini_summarizer is not None:
        return st.session_state.gemini_summarizer, "Gemini"
    elif st.session_state.claude_summarizer is not None:
        return st.session_state.claude_summarizer, "Claude"
    else:
        return None, None

def smart_summarize_text(text: str, max_length: int = 150) -> tuple:
    """스마트 텍스트 요약 (Gemini 실패 시 Claude 자동 전환)"""
    # 먼저 Gemini 시도
    if st.session_state.gemini_summarizer is not None:
        try:
            summary = st.session_state.gemini_summarizer.summarize_text(text, max_length)
            # API 할당량 초과 표시가 있으면 실패로 간주
            if "[API 할당량 초과]" not in summary:
                return summary, "Gemini"
        except:
            pass
    
    # Gemini 실패 시 Claude 시도
    if st.session_state.claude_summarizer is not None:
        try:
            summary = st.session_state.claude_summarizer.summarize_text(text, max_length)
            return summary, "Claude"
        except:
            pass
    
    # 둘 다 실패 시 기본 요약
    return text[:max_length] + "..." if len(text) > max_length else text, "기본"

st.title("🎬 CutStudio - 동영상 편집기")
st.markdown("---")

# 탭 생성: 파일 업로드와 YouTube 다운로드
upload_tab, youtube_tab = st.tabs(["📁 파일 업로드", "📺 YouTube 다운로드"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "동영상 파일을 업로드하세요",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="지원 형식: MP4, AVI, MOV, MKV"
    )

with youtube_tab:
    st.write("### YouTube 동영상 다운로드")
    
    youtube_url = st.text_input(
        "YouTube URL을 입력하세요",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    if youtube_url:
        if st.button("동영상 정보 가져오기", type="primary"):
            with st.spinner("동영상 정보를 가져오는 중..."):
                video_info = st.session_state.youtube_downloader.get_video_info(youtube_url)
                
                if video_info:
                    st.session_state.youtube_info = video_info
                    st.success("동영상 정보를 가져왔습니다!")
                else:
                    st.error("동영상 정보를 가져올 수 없습니다. URL을 확인해주세요.")
        
        # 동영상 정보 표시
        if 'youtube_info' in st.session_state and st.session_state.youtube_info:
            info = st.session_state.youtube_info
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if info['thumbnail']:
                    st.image(info['thumbnail'])
            
            with col2:
                st.write(f"**제목:** {info['title']}")
                st.write(f"**업로더:** {info['uploader']}")
                st.write(f"**길이:** {format_time(info['duration'])}")
                st.write(f"**조회수:** {info['view_count']:,}")
                if info['like_count']:
                    st.write(f"**좋아요:** {info['like_count']:,}")
            
            st.markdown("---")
            
            # 다운로드 옵션
            download_type = st.radio(
                "다운로드 형식 선택",
                ["동영상 (MP4)", "오디오만 (MP3)"],
                horizontal=True
            )
            
            if download_type == "동영상 (MP4)":
                # 해상도 선택
                if info['formats']:
                    format_options = [f"{fmt['resolution']} ({fmt['ext']})" for fmt in info['formats']]
                    format_ids = [fmt['format_id'] for fmt in info['formats']]
                    
                    selected_index = st.selectbox(
                        "해상도 선택",
                        range(len(format_options)),
                        format_func=lambda x: format_options[x]
                    )
                    selected_format_id = format_ids[selected_index]
                else:
                    selected_format_id = None
            
            # 다운로드 버튼
            if st.button("다운로드 시작", type="primary", key="download_youtube"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(percent, downloaded, total):
                    progress_bar.progress(int(percent))
                    if total > 0:
                        status_text.text(f"다운로드 중... {percent:.1f}% ({downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB)")
                
                with st.spinner("다운로드 중..."):
                    if download_type == "동영상 (MP4)":
                        file_path = st.session_state.youtube_downloader.download_video(
                            youtube_url, 
                            selected_format_id if 'selected_format_id' in locals() else None,
                            update_progress
                        )
                    else:
                        file_path = st.session_state.youtube_downloader.download_audio_only(
                            youtube_url,
                            update_progress
                        )
                
                if file_path:
                    st.success("다운로드 완료!")
                    
                    # 동영상인 경우 편집을 위해 로드
                    if download_type == "동영상 (MP4)":
                        st.session_state.video_editor.load_video(file_path)
                        st.session_state.youtube_video_path = file_path
                        st.info("다운로드된 동영상이 편집을 위해 로드되었습니다. 아래에서 편집할 수 있습니다.")
                    
                    # 다운로드 버튼
                    with open(file_path, "rb") as file:
                        st.download_button(
                            label=f"파일 다운로드 ({os.path.basename(file_path)})",
                            data=file,
                            file_name=os.path.basename(file_path),
                            mime="video/mp4" if download_type == "동영상 (MP4)" else "audio/mp3"
                        )
                else:
                    st.error("다운로드 실패. 다시 시도해주세요.")

# 비디오 처리 (업로드 또는 YouTube 다운로드)
video_loaded = False

if uploaded_file is not None:
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    temp_file_path = temp_dir / uploaded_file.name
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.video_editor.load_video(str(temp_file_path))
    video_loaded = True
elif 'youtube_video_path' in st.session_state:
    temp_file_path = st.session_state.youtube_video_path
    video_loaded = True

if video_loaded and temp_file_path:
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("원본 동영상")
        st.video(str(temp_file_path))
    
    with col2:
        st.subheader("동영상 정보")
        video_info = get_video_info(str(temp_file_path))
        st.write(f"**시간:** {format_time(video_info['duration'])}")
        st.write(f"**해상도:** {video_info['width']}x{video_info['height']}")
        st.write(f"**FPS:** {video_info['fps']:.2f}")
    
    st.markdown("---")
    st.subheader("편집 도구")
    
    tab1, tab2, tab3, tab4 = st.tabs(["✂️ 자르기", "🎞️ 트림", "🎨 효과", "👥 화자 구분"])
    
    with tab1:
        st.write("**구간 자르기**")
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.number_input(
                "시작 시간 (초)",
                min_value=0.0,
                max_value=video_info['duration'],
                value=0.0,
                step=0.1
            )
        with col2:
            end_time = st.number_input(
                "종료 시간 (초)",
                min_value=0.0,
                max_value=video_info['duration'],
                value=video_info['duration'],
                step=0.1
            )
        
        if st.button("자르기", type="primary"):
            with st.spinner("동영상을 자르는 중..."):
                output_path = st.session_state.video_editor.cut_video(start_time, end_time)
                if output_path:
                    st.success("동영상 자르기 완료!")
                    st.video(output_path)
                    
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="편집된 동영상 다운로드",
                            data=file,
                            file_name=f"cut_{os.path.basename(temp_file_path)}",
                            mime="video/mp4"
                        )
    
    with tab2:
        st.write("**동영상 트림 (앞뒤 제거)**")
        col1, col2 = st.columns(2)
        with col1:
            trim_start = st.number_input(
                "앞부분 제거 (초)",
                min_value=0.0,
                max_value=video_info['duration']/2,
                value=0.0,
                step=0.1
            )
        with col2:
            trim_end = st.number_input(
                "뒷부분 제거 (초)",
                min_value=0.0,
                max_value=video_info['duration']/2,
                value=0.0,
                step=0.1
            )
        
        if st.button("트림하기", type="primary", key="trim"):
            with st.spinner("동영상을 트림하는 중..."):
                output_path = st.session_state.video_editor.trim_video(trim_start, trim_end)
                if output_path:
                    st.success("동영상 트림 완료!")
                    st.video(output_path)
                    
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="트림된 동영상 다운로드",
                            data=file,
                            file_name=f"trim_{os.path.basename(temp_file_path)}",
                            mime="video/mp4",
                            key="download_trim"
                        )
    
    with tab3:
        st.write("**동영상 효과**")
        
        effect_type = st.selectbox(
            "효과 선택",
            ["없음", "흑백", "페이드 인", "페이드 아웃", "속도 변경"]
        )
        
        if effect_type == "속도 변경":
            speed = st.slider(
                "재생 속도",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
        
        if st.button("효과 적용", type="primary", key="effect"):
            with st.spinner("효과를 적용하는 중..."):
                if effect_type == "흑백":
                    output_path = st.session_state.video_editor.apply_grayscale()
                elif effect_type == "페이드 인":
                    output_path = st.session_state.video_editor.apply_fade_in()
                elif effect_type == "페이드 아웃":
                    output_path = st.session_state.video_editor.apply_fade_out()
                elif effect_type == "속도 변경":
                    output_path = st.session_state.video_editor.change_speed(speed)
                else:
                    output_path = None
                
                if output_path:
                    st.success("효과 적용 완료!")
                    st.video(output_path)
                    
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="효과 적용된 동영상 다운로드",
                            data=file,
                            file_name=f"effect_{os.path.basename(temp_file_path)}",
                            mime="video/mp4",
                            key="download_effect"
                        )
    
    with tab4:
        st.write("**화자별 구간 감지 및 자르기**")
        st.info("동영상에서 화자를 구분하여 각 화자의 발화 구간을 자동으로 감지합니다.")
        
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            min_duration = st.slider(
                "최소 발화 시간 (초)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="이 시간보다 짧은 발화는 무시됩니다"
            )
        
        with col2:
            speaker_option = st.selectbox(
                "화자 수 설정",
                ["자동 감지", "2명", "3명", "4명", "5명", "6명"],
                help="자동 감지를 선택하면 AI가 화자 수를 추정합니다"
            )
            
            if speaker_option == "자동 감지":
                num_speakers = None
            else:
                num_speakers = int(speaker_option[0])
        
        with col3:
            detection_method = st.selectbox(
                "감지 방법",
                ["허깅페이스 AI (최신)", "실용적 (권장)", "고급 (향상된 특징 + 스펙트럴)", "자동 (MFCC + 클러스터링)", "간단 (에너지 기반)"],
                help="허깅페이스: 최신 AI 모델 사용 (정확도 최고), 실용적: 속도와 정확도의 균형 (1-2분), 고급: 높은 정확도 (5-10분), 자동: 기본 성능, 간단: 빠르지만 덜 정확함"
            )
        
        if st.button("화자 구간 감지", type="primary", key="detect_speakers"):
            if detection_method.startswith("허깅페이스"):
                st.success("🤗 허깅페이스 AI 모델을 사용합니다 (pyannote/speaker-diarization-3.1)")
                st.info("""
                🚀 **허깅페이스 AI 감지 진행 단계:**
                1. 오디오 추출
                2. Pyannote 3.1 모델로 화자 분리
                3. 자동 화자 수 감지 및 세그먼트 추출
                4. 높은 정확도의 화자 구분
                
                **참고:** 처음 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다.
                """)
                
                # 허깅페이스 토큰 확인
                if not os.getenv("HUGGINGFACE_TOKEN"):
                    st.error("⚠️ 허깅페이스 토큰이 필요합니다!")
                    st.markdown("""
                    **토큰 설정 방법:**
                    1. https://huggingface.co/settings/tokens 에서 토큰 생성
                    2. 환경변수 설정: `export HUGGINGFACE_TOKEN=your_token_here`
                    3. 또는 `.env` 파일에 추가: `HUGGINGFACE_TOKEN=your_token_here`
                    """)
                    st.stop()
                    
            elif detection_method.startswith("실용적"):
                st.success("✅ 실용적 감지는 속도와 정확도의 균형을 제공합니다 (1-2분)")
                st.info("""
                ⚡ **실용적 감지 진행 단계:**
                1. 오디오 추출
                2. 빠른 음성 구간 검출 (Silero VAD)
                3. 핵심 특징만 추출 (MFCC 13개, 기본 피치, 스펙트럴)
                4. 적응형 K-means 클러스터링
                5. 빠른 후처리
                """)
            elif detection_method.startswith("고급"):
                st.warning("⚠️ 고급 감지는 정확하지만 시간이 오래 걸립니다 (1-3분)")
                st.info("""
                🔍 **진행 단계:**
                1. 오디오 추출
                2. 음성 구간 검출 (Silero VAD)
                3. 각 구간에서 특징 추출 (MFCC, 피치, 포먼트 등)
                4. 화자 클러스터링
                5. 후처리 및 병합
                """)
            
            # 진행 상황 표시를 위한 컨테이너
            progress_container = st.empty()
            
            with progress_container.container():
                st.info(f"🎯 화자 구간을 감지하는 중... ({detection_method})")
                
                # 동영상 길이 확인
                video_info = get_video_info(st.session_state.video_editor.video_path)
                if video_info and 'duration' in video_info:
                    duration = video_info['duration']
                    st.write(f"📹 동영상 길이: {format_time(duration)}")
                    
                    if detection_method.startswith("허깅페이스"):
                        estimated_time = duration * 0.3
                        st.warning(f"⏱️ 예상 처리 시간: {int(estimated_time)}초 ~ {int(estimated_time*2)}초")
                        st.info("💡 팁: 처음 실행 시 모델 다운로드로 추가 시간이 걸릴 수 있습니다 (약 1-2GB)")
                        
                        # 대안 제시
                        with st.expander("🚀 더 빠른 대안"):
                            st.write("""
                            **시간이 너무 오래 걸린다면:**
                            1. **"실용적 (권장)"** 방법을 사용해보세요 - 1-2분 내 처리
                            2. **"간단 (에너지 기반)"** 방법은 가장 빠르지만 정확도가 낮습니다
                            3. 긴 동영상은 먼저 잘라서 처리하는 것을 권장합니다
                            """)
                
                use_simple = detection_method.startswith("간단")
                use_advanced = detection_method.startswith("고급")
                use_enhanced = detection_method.startswith("향상된")
                use_practical = detection_method.startswith("실용적")
                use_huggingface = detection_method.startswith("허깅페이스")
                
                # 시작 시간 기록
                start_time = time.time()
                
                # 예상 시간 계산
                estimated_total = 60  # 기본값
                if detection_method.startswith("허깅페이스") and duration:
                    estimated_total = max(duration * 0.4, 30)
                elif detection_method.startswith("실용적"):
                    estimated_total = 120
                elif detection_method.startswith("고급"):
                    estimated_total = max(duration * 0.8, 180)
                elif detection_method.startswith("간단"):
                    estimated_total = 30
                
                # 시작 시간과 예상 종료 시간 계산
                start_time_str = time.strftime('%H:%M:%S')
                estimated_end_time = start_time + estimated_total
                estimated_end_str = time.strftime('%H:%M:%S', time.localtime(estimated_end_time))
                
                # Streamlit 네이티브 컴포넌트로 시작 정보 표시
                st.success("🎯 **화자 구간 감지 시작**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="🚀 시작 시간",
                        value=start_time_str
                    )
                
                with col2:
                    st.metric(
                        label="🏁 예상 종료",
                        value=estimated_end_str
                    )
                
                with col3:
                    st.metric(
                        label="⏱️ 예상 소요",
                        value=f"{int(estimated_total//60)}분 {int(estimated_total%60)}초"
                    )
                
                st.info(f"🤖 **사용 방법**: {detection_method}")
                st.info(f"📺 **동영상 길이**: {format_time(duration) if duration else '알 수 없음'}")
                
                # 단계별 진행 상황 표시
                with st.expander("📋 처리 단계", expanded=True):
                    st.write("🔄 **1단계**: 오디오 추출 및 전처리")
                    st.write("🔄 **2단계**: 음성 특징 분석") 
                    st.write("🔄 **3단계**: 화자 클러스터링")
                    st.write("🔄 **4단계**: 결과 후처리")
                    st.info(f"🤖 현재 사용 중인 방법: **{detection_method}**")
                
                # 화자 감지 실행 (st.spinner 사용)
                with st.spinner("🎯 화자 구간을 분석하는 중... 처리 시간이 오래 걸릴 수 있습니다."):
                    segments = st.session_state.video_editor.detect_speakers(
                        min_duration, 
                        num_speakers=num_speakers,
                        use_simple=use_simple,
                        use_advanced=use_advanced,
                        use_enhanced=use_enhanced,
                        use_practical=use_practical,
                        use_huggingface=use_huggingface
                    )
                
                # 최종 완료 시간
                elapsed_time = time.time() - start_time
                end_time = time.strftime('%H:%M:%S')
                
                # 실제 완료 시간과 비교 분석
                actual_end_time = time.time()
                actual_end_str = time.strftime('%H:%M:%S', time.localtime(actual_end_time))
                time_diff = elapsed_time - estimated_total
                estimated_end_str = time.strftime('%H:%M:%S', time.localtime(start_time + estimated_total))
                
                # Streamlit 네이티브 컴포넌트로 완료 정보 표시
                st.success("🎉 **처리 완료!**")
                
                # 시간 정보 표시
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="🚀 시작 시간",
                        value=time.strftime('%H:%M:%S', time.localtime(start_time))
                    )
                
                with col2:
                    st.metric(
                        label="🏁 완료 시간",
                        value=actual_end_str
                    )
                
                with col3:
                    st.metric(
                        label="⏱️ 총 소요시간",
                        value=f"{int(elapsed_time//60)}분 {elapsed_time%60:.1f}초"
                    )
                
                # 예상 vs 실제 비교
                st.info("📊 **예상 vs 실제 비교**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**예상 종료**: {estimated_end_str}")
                with col_b:
                    st.write(f"**실제 종료**: {actual_end_str}")
                
                # 성능 분석
                time_diff_msg = ""
                if time_diff < -10:
                    time_diff_msg = f"🚀 **{abs(int(time_diff))}초 빨랐습니다!**"
                    st.success(time_diff_msg)
                elif time_diff > 30:
                    time_diff_msg = f"⏰ **{int(time_diff)}초 더 걸렸습니다.**"
                    st.warning(time_diff_msg)
                else:
                    time_diff_msg = "✨ **예상 시간과 거의 일치합니다!**"
                    st.info(time_diff_msg)
                
                # 성능 평가 메시지
                if elapsed_time < estimated_total * 0.7:
                    st.success(f"🚀 예상보다 **{estimated_total - elapsed_time:.1f}초** 빠르게 완료되었습니다!")
                elif elapsed_time > estimated_total * 1.5:
                    st.warning(f"⏰ 예상보다 **{elapsed_time - estimated_total:.1f}초** 더 걸렸습니다. 복잡한 오디오이거나 시스템 부하가 있을 수 있습니다.")
                else:
                    st.info("✨ 예상 시간 범위 내에 완료되었습니다.")
                
                # 결과 요약
                st.success(f"🎉 처리 완료! 총 소요 시간: **{elapsed_time:.1f}초**")
                
                # 성능 평가
                if elapsed_time < estimated_total * 0.7:
                    st.info("🚀 예상보다 **빠르게** 완료되었습니다!")
                elif elapsed_time > estimated_total * 1.3:
                    st.warning("⏰ 예상보다 시간이 **오래** 걸렸습니다. 긴 동영상이거나 복잡한 오디오일 수 있습니다.")
                else:
                    st.info("✨ 예상 시간 내에 완료되었습니다.")
                
                if segments:
                    st.session_state.speaker_segments = segments
                    st.success(f"{len(segments)}개의 화자 구간을 감지했습니다!")
                    
                    # 화자별 통계 표시
                    speakers = {}
                    for seg in segments:
                        speaker = seg['speaker']
                        if speaker not in speakers:
                            speakers[speaker] = {'count': 0, 'total_duration': 0}
                        speakers[speaker]['count'] += 1
                        speakers[speaker]['total_duration'] += seg['duration']
                    
                    st.write("**화자별 통계:**")
                    for speaker, stats in speakers.items():
                        st.write(f"- {speaker}: {stats['count']}개 구간, 총 {format_time(stats['total_duration'])}")
                    
                    # 자동으로 음성 인식 실행
                    st.info("🎤 자동으로 음성 인식을 시작합니다...")
                    
                    # Whisper 모델 초기화 (tiny 모델 사용)
                    if st.session_state.speech_recognizer is None:
                        with st.spinner("Whisper 모델 로딩 중..."):
                            st.session_state.speech_recognizer = SpeechRecognizer("tiny")
                    
                    if st.session_state.speech_recognizer.model is not None:
                        with st.spinner("🗣️ 음성 인식 중... (시간이 걸릴 수 있습니다)"):
                            # 전체 비디오 음성 인식
                            result = st.session_state.speech_recognizer.transcribe_video(
                                st.session_state.video_editor.video_path,
                                language="ko"
                            )
                            
                            if result:
                                st.session_state.full_transcription = result
                                
                                if 'segments' in result:
                                    # 인식된 텍스트를 화자 세그먼트에 매핑
                                    recognized_segments = []
                                    whisper_segments = result['segments']
                                    
                                    for seg in segments:
                                        seg_copy = seg.copy()
                                        seg_copy['text'] = ""
                                        matched_texts = []
                                        
                                        seg_start = seg['start']
                                        seg_end = seg['end']
                                        
                                        for whisper_seg in whisper_segments:
                                            w_start = whisper_seg['start']
                                            w_end = whisper_seg['end']
                                            
                                            if (w_start < seg_end and w_end > seg_start):
                                                overlap_start = max(w_start, seg_start)
                                                overlap_end = min(w_end, seg_end)
                                                overlap_duration = overlap_end - overlap_start
                                                
                                                if overlap_duration > 0.2 * (w_end - w_start):
                                                    matched_texts.append(whisper_seg['text'].strip())
                                        
                                        seg_copy['text'] = " ".join(matched_texts)
                                        seg_copy['has_text'] = bool(seg_copy['text'])
                                        recognized_segments.append(seg_copy)
                                    
                                    st.session_state.recognized_segments = recognized_segments
                                    
                                    # 인식된 텍스트 개수 확인
                                    text_count = sum(1 for seg in recognized_segments if seg.get('text', '').strip())
                                    st.success(f"✅ 음성 인식 완료! {text_count}/{len(recognized_segments)}개 구간에서 음성 감지")
                                else:
                                    st.warning("음성 인식 결과가 없습니다.")
                            else:
                                st.error("음성 인식에 실패했습니다.")
                    else:
                        st.error("Whisper 모델 로딩에 실패했습니다.")
                else:
                    st.warning("화자 구간을 감지하지 못했습니다.")
        
        # 화자 구간이 감지된 경우
        if 'speaker_segments' in st.session_state and st.session_state.speaker_segments:
            st.markdown("---")
            
            # 화자별 프로필 생성 및 표시
            st.subheader("👥 화자별 프로필")
            
            # 화자별 프로필 정보 생성
            speaker_profiles = st.session_state.video_editor.generate_speaker_profile(st.session_state.speaker_segments)
            
            if speaker_profiles:
                # 탭으로 프로필과 상세 정보 분리
                profile_tab, detail_tab, transcript_tab = st.tabs(["👤 프로필", "📊 상세 통계", "📝 음성 인식"])
                
                with profile_tab:
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
                                if profile['has_thumbnail']:
                                    thumbnail = profile['thumbnail']
                                    st.image(
                                        f"data:image/jpeg;base64,{thumbnail['image_base64']}",
                                        caption=f"타임스탬프: {format_time(thumbnail['timestamp'])}",
                                        width=150
                                    )
                                else:
                                    st.info("썸네일 생성 실패")
                                
                                # 요약 정보 표시
                                if profile['has_summary']:
                                    summary = profile['summary']
                                    st.metric("총 발화 시간", f"{format_time(summary['total_duration'])}")
                                    st.metric("발화 횟수", f"{summary['segment_count']}회")
                                    st.metric("참여율", f"{summary['participation_rate']}%")
                                    
                                    # 첫 등장 시간
                                    st.write(f"**첫 등장:** {format_time(summary['first_appearance'])}")
                                    st.write(f"**마지막 등장:** {format_time(summary['last_appearance'])}")
                                
                                st.markdown("---")
                
                with detail_tab:
                    st.write("**화자별 상세 통계:**")
                    
                    # 통계 테이블 생성
                    stats_data = []
                    for speaker_id, profile in speaker_profiles.items():
                        if profile['has_summary']:
                            summary = profile['summary']
                            stats_data.append({
                                '화자': speaker_id,
                                '총 발화 시간': format_time(summary['total_duration']),
                                '발화 횟수': summary['segment_count'],
                                '평균 발화 길이': format_time(summary['avg_duration']),
                                '참여율 (%)': summary['participation_rate'],
                                '첫 등장': format_time(summary['first_appearance']),
                                '마지막 등장': format_time(summary['last_appearance'])
                            })
                    
                    if stats_data:
                        st.dataframe(stats_data, use_container_width=True)
                
                with transcript_tab:
                    st.write("**음성 인식 및 내용 요약**")
                    
                    # 음성 인식 초기화 확인
                    if st.session_state.speech_recognizer is None:
                        st.info("음성 인식 기능을 초기화하려면 아래 버튼을 클릭하세요.")
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            model_size = st.selectbox(
                                "Whisper 모델 크기",
                                ["tiny", "base", "small", "medium"],
                                index=1,
                                help="tiny: 빠름/낮은 정확도, base: 균형, small: 높은 정확도, medium: 최고 정확도/느림"
                            )
                        
                        if st.button("음성 인식 초기화", type="primary"):
                            with st.spinner("Whisper 모델을 로딩하는 중..."):
                                st.session_state.speech_recognizer = SpeechRecognizer(model_size)
                                if st.session_state.speech_recognizer.model is not None:
                                    st.success("음성 인식기가 초기화되었습니다!")
                                    st.rerun()
                                else:
                                    st.error("음성 인식기 초기화에 실패했습니다.")
                    
                    else:
                        # 음성 인식 실행 옵션
                        col1, col2 = st.columns(2)
                        with col1:
                            language = st.selectbox(
                                "언어 선택",
                                ["ko", "en", "auto"],
                                index=0,
                                help="ko: 한국어, en: 영어, auto: 자동 감지"
                            )
                        
                        with col2:
                            include_summary = st.checkbox("대화 내용 요약 포함", value=True)
                        
                        if st.button("음성 인식 실행", type="primary"):
                            with st.spinner("음성 인식을 실행하는 중..."):
                                # 화자별 세그먼트에 음성 인식 적용
                                recognized_segments = st.session_state.speech_recognizer.transcribe_segments(
                                    st.session_state.video_editor.video_path,
                                    st.session_state.speaker_segments,
                                    language=language
                                )
                                
                                if recognized_segments:
                                    st.session_state.recognized_segments = recognized_segments
                                    st.success("음성 인식이 완료되었습니다!")
                                else:
                                    st.error("음성 인식에 실패했습니다.")
                        
                        # 음성 인식 결과 표시
                        if 'recognized_segments' in st.session_state:
                            st.markdown("### 📝 음성 인식 결과")
                            
                            # 화자별로 그룹화하여 표시
                            recognized_by_speaker = {}
                            for segment in st.session_state.recognized_segments:
                                speaker = segment['speaker']
                                if speaker not in recognized_by_speaker:
                                    recognized_by_speaker[speaker] = []
                                recognized_by_speaker[speaker].append(segment)
                            
                            for speaker_id, segments in recognized_by_speaker.items():
                                with st.expander(f"{speaker_id} 발화 내용", expanded=True):
                                    for i, segment in enumerate(segments):
                                        if segment.get('has_text', False):
                                            st.write(f"**{format_time(segment['start'])} - {format_time(segment['end'])}**")
                                            st.write(f"🗣️ {segment['text']}")
                                            st.markdown("---")
                                        else:
                                            st.write(f"**{format_time(segment['start'])} - {format_time(segment['end'])}**: *음성 인식 결과 없음*")
                            
                            # 대화 요약 생성
                            if include_summary:
                                st.markdown("### 📋 대화 요약")
                                
                                analyzer = AdvancedSpeechAnalyzer(st.session_state.speech_recognizer)
                                meeting_summary = analyzer.generate_meeting_summary(st.session_state.recognized_segments)
                                
                                if meeting_summary:
                                    st.text_area(
                                        "종합 요약",
                                        meeting_summary,
                                        height=300,
                                        disabled=True
                                    )
                                
                                # 대화 흐름 분석
                                conversation_analysis = analyzer.analyze_conversation_flow(st.session_state.recognized_segments)
                                
                                if conversation_analysis.get('timeline'):
                                    st.markdown("### 🕒 대화 타임라인")
                                    
                                    timeline_data = []
                                    for item in conversation_analysis['timeline']:
                                        timeline_data.append({
                                            '시간': format_time(item['time']),
                                            '화자': item['speaker'],
                                            '내용': item['text'],
                                            '길이': format_time(item['duration'])
                                        })
                                    
                                    st.dataframe(timeline_data, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🎬 전체 대화 타임라인")
            
            # 시간순으로 정렬
            sorted_segments = sorted(st.session_state.speaker_segments, key=lambda x: x['start'])
            
            # 화자별 프로필 정보 생성 (썸네일 포함)
            all_profiles = st.session_state.video_editor.generate_speaker_profile(sorted_segments)
            
            # 음성 인식이 있는 경우 텍스트 매핑
            segment_texts = {}
            if 'recognized_segments' in st.session_state:
                for rec_seg in st.session_state.recognized_segments:
                    key = f"{rec_seg['start']:.1f}_{rec_seg['end']:.1f}"
                    segment_texts[key] = rec_seg.get('text', '')
            
            # 전체 타임라인 표시
            st.info(f"📊 총 {len(sorted_segments)}개의 발화 구간")
            
            # 세그먼트를 행당 4개씩 표시
            cols_per_row = 4
            for i in range(0, len(sorted_segments), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, seg in enumerate(sorted_segments[i:i+cols_per_row]):
                    if j < len(cols):
                        with cols[j]:
                            # 썸네일 생성
                            try:
                                if st.session_state.video_editor.video_clip:
                                    mid_time = (seg['start'] + seg['end']) / 2
                                    frame = st.session_state.video_editor.video_clip.get_frame(mid_time)
                                    
                                    from PIL import Image
                                    import base64
                                    import io
                                    
                                    pil_image = Image.fromarray(frame)
                                    pil_image.thumbnail((200, 150), Image.Resampling.LANCZOS)
                                    
                                    # Base64 인코딩
                                    buffer = io.BytesIO()
                                    pil_image.save(buffer, format='JPEG', quality=85)
                                    img_str = base64.b64encode(buffer.getvalue()).decode()
                                    
                                    # 썸네일 표시
                                    st.image(f"data:image/jpeg;base64,{img_str}", use_container_width=True)
                            except:
                                st.image("https://via.placeholder.com/200x150?text=No+Thumbnail", use_container_width=True)
                            
                            # 화자 정보
                            speaker_color = {
                                'SPEAKER_0': '🔴',
                                'SPEAKER_1': '🔵', 
                                'SPEAKER_2': '🟢',
                                'SPEAKER_3': '🟡',
                                'SPEAKER_4': '🟣',
                                'SPEAKER_5': '🟤'
                            }
                            
                            speaker_emoji = speaker_color.get(seg['speaker'], '⚪')
                            st.markdown(f"### {speaker_emoji} {seg['speaker']}")
                            
                            # 시간 정보
                            st.caption(f"⏱️ {format_time(seg['start'])} - {format_time(seg['end'])}")
                            st.caption(f"📏 길이: {format_time(seg['duration'])}")
                            
                            # 텍스트/요약
                            seg_key = f"{seg['start']:.1f}_{seg['end']:.1f}"
                            text = segment_texts.get(seg_key, '')
                            
                            # recognized_segments에서 직접 찾기
                            if not text and 'recognized_segments' in st.session_state:
                                for rec_seg in st.session_state.recognized_segments:
                                    if (abs(rec_seg['start'] - seg['start']) < 1.0 and 
                                        abs(rec_seg['end'] - seg['end']) < 1.0):
                                        text = rec_seg.get('text', '').strip()
                                        break
                            
                            if text:
                                # 텍스트 길이 확인
                                text_length = len(text)
                                
                                # Gemini로 요약 시도 (임계값 50자로 낮춤)
                                if GEMINI_AVAILABLE and st.session_state.gemini_summarizer is not None:
                                    try:
                                        if text_length > 50:
                                            # Gemini 요약 시도 (길이 늘림)
                                            summary = st.session_state.gemini_summarizer.summarize_text(text, 150)
                                            st.info(f"💬 {summary}")
                                            st.caption(f"✅ Gemini 요약 완료 | 원본: {text_length}자")
                                        else:
                                            # 짧은 텍스트는 그대로 표시
                                            st.info(f"💬 {text}")
                                            st.caption(f"원본 텍스트 | {text_length}자")
                                    except Exception as e:
                                        # 에러 발생 시 더 나은 기본 요약
                                        st.warning(f"⚠️ Gemini 요약 실패: {str(e)}")
                                        # 문장 단위로 더 나은 기본 요약
                                        sentences = text.replace('?', '.').replace('!', '.').split('.')
                                        sentences = [s.strip() for s in sentences if s.strip()]
                                        if sentences:
                                            summary = sentences[0]
                                            if len(summary) > 80:
                                                summary = summary[:80] + "..."
                                        else:
                                            summary = text[:80] + "..." if text_length > 80 else text
                                        st.info(f"💬 {summary}")
                                        st.caption(f"기본 요약 (Gemini 오류) | 원본: {text_length}자")
                                else:
                                    # Gemini 사용 불가 시 더 나은 기본 요약
                                    sentences = text.replace('?', '.').replace('!', '.').split('.')
                                    sentences = [s.strip() for s in sentences if s.strip()]
                                    if sentences and len(sentences) > 1:
                                        # 첫 문장 + 키워드
                                        summary = sentences[0]
                                        if len(summary) > 60:
                                            summary = summary[:60] + "..."
                                        st.info(f"💬 {summary}")
                                        st.caption(f"기본 요약 | 원본: {text_length}자")
                                    else:
                                        # 텍스트가 짧거나 문장 분리 실패
                                        summary = text[:80] + "..." if text_length > 80 else text
                                        st.info(f"💬 {summary}")
                                        st.caption(f"원본 텍스트 | {text_length}자")
                            else:
                                st.caption("🔇 음성 인식 필요")
                            
                            # 구분선
                            st.markdown("---")
                    
            # 음성 인식 버튼 추가
            st.markdown("---")
            
            # 전체 음성 인식 결과 표시 (화자 구분 없이)
            if 'full_transcription' in st.session_state and st.session_state.full_transcription:
                with st.expander("📝 전체 음성 인식 결과", expanded=True):
                    full_text = st.session_state.full_transcription.get('text', '')
                    
                    # 전체 텍스트 요약 생성
                    if full_text and len(full_text.strip()) > 100:
                        # 요약 헤더와 버튼
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader("📋 전체 대화 요약")
                        with col2:
                            if st.button("🔄 요약 새로고침", key="refresh_summary"):
                                # 요약 캐시 삭제하고 새로고침
                                if 'summary_cache' in st.session_state:
                                    del st.session_state.summary_cache
                                st.rerun()
                        
                        # Gemini로 전체 요약 시도
                        if GEMINI_AVAILABLE and st.session_state.gemini_summarizer is not None:
                            try:
                                # 전체 대화 요약 (더 긴 요약)
                                full_summary = st.session_state.gemini_summarizer.summarize_text(full_text, 200)
                                st.success("✅ **AI 요약 완료:**")
                                st.info(full_summary)
                                
                                # 키워드 추출
                                keywords = st.session_state.gemini_summarizer.extract_keywords(full_text, 8)
                                if keywords:
                                    st.write("✅ **키워드 추출 완료:**")
                                    keyword_tags = " ".join([f"`{kw}`" for kw in keywords])
                                    st.markdown(keyword_tags)
                                
                            except Exception as e:
                                st.warning(f"⚠️ AI 요약 실패: {str(e)}")
                                # 기본 요약
                                sentences = full_text.replace('?', '.').replace('!', '.').split('.')
                                sentences = [s.strip() for s in sentences if s.strip()]
                                if len(sentences) > 3:
                                    summary = '. '.join(sentences[:3]) + '.'
                                else:
                                    summary = full_text[:200] + "..."
                                st.info(f"📝 **기본 요약:** {summary}")
                        else:
                            # Gemini 사용 불가 시 기본 요약
                            sentences = full_text.replace('?', '.').replace('!', '.').split('.')
                            sentences = [s.strip() for s in sentences if s.strip()]
                            if len(sentences) > 3:
                                summary = '. '.join(sentences[:3]) + '.'
                            else:
                                summary = full_text[:200] + "..."
                            st.info(f"📝 **기본 요약:** {summary}")
                        
                        # 통계 정보
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("총 글자 수", f"{len(full_text):,}자")
                        with col2:
                            word_count = len(full_text.split())
                            st.metric("총 단어 수", f"{word_count:,}개")
                        with col3:
                            if 'segments' in st.session_state.full_transcription:
                                seg_count = len(st.session_state.full_transcription['segments'])
                                st.metric("음성 세그먼트", f"{seg_count}개")
                        
                        st.markdown("---")
                    
                    # 화자별 종합 분석 (음성 인식된 세그먼트가 있는 경우)
                    if 'recognized_segments' in st.session_state and st.session_state.recognized_segments:
                        st.subheader("👥 화자별 대화 분석")
                        
                        # Gemini로 회의/대화 종합 요약
                        if GEMINI_AVAILABLE and st.session_state.gemini_summarizer is not None:
                            try:
                                # 화자별 세그먼트 준비
                                segments_for_analysis = []
                                for seg in st.session_state.recognized_segments:
                                    if seg.get('text', '').strip():
                                        segments_for_analysis.append({
                                            'speaker': seg['speaker'],
                                            'text': seg['text'],
                                            'start': seg['start'],
                                            'end': seg['end']
                                        })
                                
                                if segments_for_analysis:
                                    # 화자별 요약 (회의 분석 대신 화자별 분석으로 변경)
                                    speaker_summaries = st.session_state.gemini_summarizer.summarize_conversation(segments_for_analysis)
                                    if speaker_summaries:
                                        st.write("**✅ 화자별 발언 요약 완료:**")
                                        for speaker, summary in speaker_summaries.items():
                                            if summary and summary != "발화 내용 없음":
                                                st.write(f"**🎤 {speaker}:**")
                                                st.info(summary)
                                
                            except Exception as e:
                                st.warning(f"⚠️ 화자별 분석 실패: {str(e)}")
                        
                        st.markdown("---")
                    
            
            # 전체 텍스트 원본 (별도 expander)
            if 'full_transcription' in st.session_state and st.session_state.full_transcription:
                full_text = st.session_state.full_transcription.get('text', '')
                if full_text:
                    with st.expander("📄 전체 텍스트 원본", expanded=False):
                        st.write("**전체 텍스트:**")
                        st.text_area("", full_text, height=200, disabled=True)
                    
                    # 세그먼트별 상세 (별도 expander)
                    if 'segments' in st.session_state.full_transcription:
                        with st.expander(f"🔍 세그먼트별 상세 ({len(st.session_state.full_transcription['segments'])}개)", expanded=False):
                            for i, seg in enumerate(st.session_state.full_transcription['segments']):
                                st.write(f"{i+1}. [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}")
            
            # Gemini API 상태 확인 (별도 expander)
            with st.expander("🤖 Gemini API 상태", expanded=False):
                st.write(f"**Gemini 사용 가능:** {'✅ Yes' if GEMINI_AVAILABLE else '❌ No'}")
                if GEMINI_AVAILABLE:
                    st.write(f"**Gemini 초기화:** {'✅ Yes' if st.session_state.gemini_summarizer is not None else '❌ No'}")
                    
                    if st.session_state.gemini_summarizer is not None:
                        # Gemini 테스트
                        if st.button("🧪 Gemini 테스트", key="test_gemini"):
                            try:
                                test_text = "이것은 Gemini API 테스트를 위한 긴 텍스트입니다. 여러 문장을 포함하고 있으며, API가 제대로 작동하는지 확인하기 위한 목적으로 작성되었습니다. 이 텍스트가 제대로 요약되면 Gemini API가 정상적으로 작동하는 것입니다."
                                result = st.session_state.gemini_summarizer.summarize_text(test_text, 50)
                                st.success("✅ Gemini API 테스트 완료!")
                                st.write(f"**원본:** {test_text}")
                                st.write(f"**요약:** {result}")
                            except Exception as e:
                                st.error(f"❌ Gemini API 오류: {str(e)}")
                else:
                    st.info("Gemini API를 사용하려면 google-generativeai 패키지가 필요합니다.")
                    st.code("pip install google-generativeai")
            
            # 음성 인식 디버그 옵션
            with st.expander("🔧 음성 인식 디버그", expanded=False):
                if st.button("🧪 음성 인식 테스트 (첫 30초)", key="test_whisper"):
                    with st.spinner("테스트 중..."):
                        if st.session_state.speech_recognizer is None:
                            st.session_state.speech_recognizer = SpeechRecognizer("tiny")
                        
                        # 비디오 경로 확인
                        video_path = st.session_state.video_editor.video_path
                        st.write(f"비디오 경로: {video_path}")
                        
                        # 간단한 테스트 - 첫 30초만
                        try:
                            import tempfile
                            from moviepy.editor import VideoFileClip
                            
                            # 첫 30초만 추출
                            with VideoFileClip(video_path) as video:
                                test_duration = min(30, video.duration)
                                subclip = video.subclip(0, test_duration)
                                
                                # 임시 파일로 저장
                                temp_path = tempfile.mktemp(suffix='.mp4')
                                subclip.write_videofile(temp_path, verbose=False, logger=None)
                                
                                # Whisper 테스트
                                result = st.session_state.speech_recognizer.transcribe_video(temp_path, language="ko")
                                
                                if result and 'text' in result:
                                    st.success("✅ 음성 인식 성공!")
                                    st.write("전체 텍스트:", result['text'])
                                    
                                    if 'segments' in result:
                                        st.write(f"세그먼트 수: {len(result['segments'])}")
                                        for i, seg in enumerate(result['segments'][:5]):  # 첫 5개만
                                            st.write(f"{i+1}. [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}")
                                else:
                                    st.error("❌ 음성 인식 실패: 텍스트 없음")
                                
                                # 임시 파일 삭제
                                import os
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                    
                        except Exception as e:
                            st.error(f"테스트 실패: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            
            # 수동 음성 인식 옵션 (이미 자동 실행된 경우 표시)
            if 'recognized_segments' in st.session_state and st.session_state.recognized_segments:
                st.success("✅ 음성 인식이 이미 완료되었습니다!")
                
                # 재실행 옵션
                with st.expander("🔄 다른 모델로 재실행", expanded=False):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        whisper_model = st.selectbox(
                            "🎙️ Whisper 모델",
                            ["tiny", "base", "small"],
                            index=0,
                            help="tiny: 빠름(39MB), base: 균형(74MB), small: 정확(244MB)"
                        )
                    
                    if st.button("🔄 음성 인식 재실행", type="secondary", key="rerun_whisper"):
                        # 기존 코드와 동일한 음성 인식 로직
                        try:
                            with st.spinner(f"Whisper {whisper_model} 모델 로딩 중..."):
                                if st.session_state.speech_recognizer is None or st.session_state.speech_recognizer.model_size != whisper_model:
                                    st.session_state.speech_recognizer = SpeechRecognizer(whisper_model)
                                
                                if st.session_state.speech_recognizer.model is not None:
                                    # 진행 상황 표시
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    # 전체 비디오 음성 인식
                                    status_text.text("🎵 오디오 추출 중...")
                                    progress_bar.progress(10)
                                    
                                    # 직접 전체 비디오 음성 인식 실행
                                    status_text.text("🗣️ 음성 인식 중... (시간이 걸릴 수 있습니다)")
                                    progress_bar.progress(30)
                                    
                                    result = st.session_state.speech_recognizer.transcribe_video(
                                        st.session_state.video_editor.video_path,
                                        language="ko"
                                    )
                                    
                                    progress_bar.progress(80)
                                    
                                    # 전체 음성 인식 결과 저장
                                    if result:
                                        st.session_state.full_transcription = result
                                    
                                    if result and 'segments' in result:
                                        # 인식된 텍스트를 화자 세그먼트에 매핑
                                        recognized_segments = []
                                        whisper_segments = result['segments']
                                        
                                        # 디버그 정보
                                        st.write(f"🎯 Whisper 세그먼트 수: {len(whisper_segments)}")
                                        st.write(f"🎯 화자 세그먼트 수: {len(sorted_segments)}")
                                        
                                        for seg in sorted_segments:
                                            seg_copy = seg.copy()
                                            seg_copy['text'] = ""
                                            matched_texts = []
                                            
                                            # 해당 시간대의 텍스트 찾기 (더 유연한 매칭)
                                            seg_start = seg['start']
                                            seg_end = seg['end']
                                            
                                            for whisper_seg in whisper_segments:
                                                w_start = whisper_seg['start']
                                                w_end = whisper_seg['end']
                                                
                                                # 시간 범위가 겹치는지 확인 (더 관대한 조건)
                                                if (w_start < seg_end and w_end > seg_start):
                                                    # 겹치는 비율 계산
                                                    overlap_start = max(w_start, seg_start)
                                                    overlap_end = min(w_end, seg_end)
                                                    overlap_duration = overlap_end - overlap_start
                                                    
                                                    # 20% 이상 겹치면 포함
                                                    if overlap_duration > 0.2 * (w_end - w_start):
                                                        matched_texts.append(whisper_seg['text'].strip())
                                            
                                            seg_copy['text'] = " ".join(matched_texts)
                                            seg_copy['has_text'] = bool(seg_copy['text'])
                                            recognized_segments.append(seg_copy)
                                        
                                        # 매칭 결과 확인
                                        matched_count = sum(1 for s in recognized_segments if s['has_text'])
                                        st.write(f"🎯 매칭된 세그먼트: {matched_count}/{len(recognized_segments)}")
                                        
                                        st.session_state.recognized_segments = recognized_segments
                                        
                                        # 인식된 텍스트 개수 확인
                                        text_count = sum(1 for seg in recognized_segments if seg.get('text', '').strip())
                                        progress_bar.progress(100)
                                        st.success(f"✅ 음성 인식 완료! {text_count}/{len(recognized_segments)}개 구간에서 음성 감지")
                                        status_text.text("")
                                        st.rerun()
                                    else:
                                        st.error("❌ 음성 인식 실패: 텍스트를 추출할 수 없습니다")
                                else:
                                    st.error("❌ Whisper 모델 로딩 실패")
                        except Exception as e:
                            st.error(f"❌ 오류 발생: {str(e)}")
                            st.info("💡 팁: 더 작은 모델(tiny)을 시도해보세요")
            else:
                # 음성 인식이 아직 안 된 경우 (자동 실행이 실패했거나 아직 화자 감지를 안 한 경우)
                st.warning("⚠️ 음성 인식이 아직 실행되지 않았습니다.")
                
                # Whisper 모델 선택
                col1, col2 = st.columns([1, 2])
                with col1:
                    whisper_model = st.selectbox(
                        "🎙️ Whisper 모델",
                        ["tiny", "base", "small"],
                        index=0,
                        help="tiny: 빠름(39MB), base: 균형(74MB), small: 정확(244MB)"
                    )
                
                if st.button("🎤 음성 인식 실행", type="primary", key="run_whisper"):
                    try:
                        with st.spinner(f"Whisper {whisper_model} 모델 로딩 중..."):
                            if st.session_state.speech_recognizer is None or st.session_state.speech_recognizer.model_size != whisper_model:
                                st.session_state.speech_recognizer = SpeechRecognizer(whisper_model)
                            
                            if st.session_state.speech_recognizer.model is not None:
                                # 진행 상황 표시
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # 전체 비디오 음성 인식
                                status_text.text("🎵 오디오 추출 중...")
                                progress_bar.progress(10)
                                
                                # 직접 전체 비디오 음성 인식 실행
                                status_text.text("🗣️ 음성 인식 중... (시간이 걸릴 수 있습니다)")
                                progress_bar.progress(30)
                                
                                result = st.session_state.speech_recognizer.transcribe_video(
                                    st.session_state.video_editor.video_path,
                                    language="ko"
                                )
                                
                                progress_bar.progress(80)
                                
                                # 전체 음성 인식 결과 저장
                                if result:
                                    st.session_state.full_transcription = result
                                
                                if result and 'segments' in result:
                                    # 인식된 텍스트를 화자 세그먼트에 매핑
                                    recognized_segments = []
                                    whisper_segments = result['segments']
                                    
                                    # 디버그 정보
                                    st.write(f"🎯 Whisper 세그먼트 수: {len(whisper_segments)}")
                                    st.write(f"🎯 화자 세그먼트 수: {len(sorted_segments)}")
                                    
                                    for seg in sorted_segments:
                                        seg_copy = seg.copy()
                                        seg_copy['text'] = ""
                                        matched_texts = []
                                        
                                        # 해당 시간대의 텍스트 찾기 (더 유연한 매칭)
                                        seg_start = seg['start']
                                        seg_end = seg['end']
                                        
                                        for whisper_seg in whisper_segments:
                                            w_start = whisper_seg['start']
                                            w_end = whisper_seg['end']
                                            
                                            # 시간 범위가 겹치는지 확인 (더 관대한 조건)
                                            if (w_start < seg_end and w_end > seg_start):
                                                # 겹치는 비율 계산
                                                overlap_start = max(w_start, seg_start)
                                                overlap_end = min(w_end, seg_end)
                                                overlap_duration = overlap_end - overlap_start
                                                
                                                # 20% 이상 겹치면 포함
                                                if overlap_duration > 0.2 * (w_end - w_start):
                                                    matched_texts.append(whisper_seg['text'].strip())
                                        
                                        seg_copy['text'] = " ".join(matched_texts)
                                        seg_copy['has_text'] = bool(seg_copy['text'])
                                        recognized_segments.append(seg_copy)
                                    
                                    # 매칭 결과 확인
                                    matched_count = sum(1 for s in recognized_segments if s['has_text'])
                                    st.write(f"🎯 매칭된 세그먼트: {matched_count}/{len(recognized_segments)}")
                                    
                                    st.session_state.recognized_segments = recognized_segments
                                    
                                    # 인식된 텍스트 개수 확인
                                    text_count = sum(1 for seg in recognized_segments if seg.get('text', '').strip())
                                    progress_bar.progress(100)
                                    st.success(f"✅ 음성 인식 완료! {text_count}/{len(recognized_segments)}개 구간에서 음성 감지")
                                    status_text.text("")
                                    st.rerun()
                                else:
                                    st.error("❌ 음성 인식 실패: 텍스트를 추출할 수 없습니다")
                            else:
                                st.error("❌ Whisper 모델 로딩 실패")
                    except Exception as e:
                        st.error(f"❌ 오류 발생: {str(e)}")
                        st.info("💡 팁: 더 작은 모델(tiny)을 시도해보세요")
            
            # 화자별 요약 통계 표시
            st.markdown("---")
            st.subheader("📊 화자별 요약 통계")
            
            # 화자별 통계 수집
            speaker_stats = {}
            for seg in sorted_segments:
                speaker = seg['speaker']
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        'count': 0,
                        'total_duration': 0,
                        'segments': []
                    }
                speaker_stats[speaker]['count'] += 1
                speaker_stats[speaker]['total_duration'] += seg['duration']
                speaker_stats[speaker]['segments'].append(seg)
            
            # 화자별 통계 카드 표시
            stats_cols = st.columns(len(speaker_stats))
            for idx, (speaker, stats) in enumerate(speaker_stats.items()):
                with stats_cols[idx]:
                    speaker_emoji = {
                        'SPEAKER_0': '🔴',
                        'SPEAKER_1': '🔵', 
                        'SPEAKER_2': '🟢',
                        'SPEAKER_3': '🟡',
                        'SPEAKER_4': '🟣',
                        'SPEAKER_5': '🟤'
                    }.get(speaker, '⚪')
                    
                    st.metric(
                        label=f"{speaker_emoji} {speaker}",
                        value=f"{stats['count']}회",
                        delta=f"{format_time(stats['total_duration'])}"
                    )
                    
                    # 참여율 계산
                    if st.session_state.video_editor.video_clip:
                        participation = (stats['total_duration'] / st.session_state.video_editor.video_clip.duration) * 100
                        st.caption(f"참여율: {participation:.1f}%")
            
            # 전체 변경사항 저장 버튼
            if st.button("🔄 전체 페이지 새로고침", type="primary"):
                st.rerun()
            
            st.markdown("---")
            st.write("**화자별 동영상 자르기 옵션:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("모든 화자 구간을 개별 파일로 저장", type="secondary"):
                    with st.spinner("화자별로 동영상을 자르는 중..."):
                        output_files = st.session_state.video_editor.cut_by_speaker(st.session_state.speaker_segments)
                        
                        if output_files:
                            st.success(f"{len(output_files)}개의 동영상 파일이 생성되었습니다!")
                            
                            for file_info in output_files:
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.write(f"**{file_info['speaker']}** - {format_time(file_info['start'])} ~ {format_time(file_info['end'])}")
                                with col_b:
                                    with open(file_info['path'], "rb") as file:
                                        st.download_button(
                                            label="다운로드",
                                            data=file,
                                            file_name=f"{file_info['speaker']}_{format_time(file_info['start']).replace(':', '_')}.mp4",
                                            mime="video/mp4",
                                            key=f"download_{file_info['path']}"
                                        )
            
            with col2:
                # 특정 화자만 선택하여 하나의 파일로 만들기
                speakers_list = list(set(seg['speaker'] for seg in st.session_state.speaker_segments))
                selected_speaker = st.selectbox("특정 화자 선택", speakers_list)
                
                if selected_speaker and st.button(f"{selected_speaker}의 모든 구간 합치기", type="secondary"):
                    with st.spinner(f"{selected_speaker}의 구간을 합치는 중..."):
                        output_path = st.session_state.video_editor.cut_single_speaker(selected_speaker)
                        
                        if output_path:
                            st.success(f"{selected_speaker}의 모든 구간을 하나로 합쳤습니다!")
                            st.video(output_path)
                            
                            with open(output_path, "rb") as file:
                                st.download_button(
                                    label=f"{selected_speaker} 전체 동영상 다운로드",
                                    data=file,
                                    file_name=f"{selected_speaker}_combined.mp4",
                                    mime="video/mp4",
                                    key=f"download_combined_{selected_speaker}"
                                )

else:
    st.info("동영상 파일을 업로드하여 편집을 시작하세요.")

if st.button("임시 파일 정리"):
    temp_dir = Path("temp")
    processed_dir = Path("processed")
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    
    st.success("임시 파일이 정리되었습니다.")
    st.rerun()