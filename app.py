import streamlit as st
import tempfile
import os
from pathlib import Path
import shutil
import time
from video_editor import VideoEditor
from utils import get_video_info, format_time
from youtube_downloader import YouTubeDownloader

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
                
                # 감지 시작
                start_time = time.time()
                
                # 진행률 표시
                if detection_method.startswith("허깅페이스"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 주기적으로 진행 상황 업데이트 (실제로는 추정치)
                    status_text.text("모델 초기화 중...")
                    progress_bar.progress(10)
                
                segments = st.session_state.video_editor.detect_speakers(
                    min_duration, 
                    num_speakers=num_speakers,
                    use_simple=use_simple,
                    use_advanced=use_advanced,
                    use_enhanced=use_enhanced,
                    use_practical=use_practical,
                    use_huggingface=use_huggingface
                )
                
                # 소요 시간 표시
                elapsed_time = time.time() - start_time
                
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
                else:
                    st.warning("화자 구간을 감지하지 못했습니다.")
        
        # 화자 구간이 감지된 경우
        if 'speaker_segments' in st.session_state and st.session_state.speaker_segments:
            st.markdown("---")
            st.write("**화자별 구간 정보:**")
            
            # 구간 정보를 표로 표시 (편집 가능)
            st.write("**💡 팁:** 화자 열을 클릭하여 수정할 수 있습니다.")
            
            segment_data = []
            for i, seg in enumerate(st.session_state.speaker_segments):
                segment_data.append({
                    '번호': i + 1,
                    '화자': seg['speaker'],
                    '시작': format_time(seg['start']),
                    '종료': format_time(seg['end']),
                    '길이': format_time(seg['duration']),
                    '신뢰도': f"{seg.get('confidence', 0.8):.1%}" if 'confidence' in seg else "N/A"
                })
            
            # 편집 가능한 데이터프레임
            edited_df = st.data_editor(
                segment_data, 
                column_config={
                    "번호": st.column_config.NumberColumn("번호", disabled=True),
                    "화자": st.column_config.SelectboxColumn(
                        "화자",
                        help="클릭하여 화자를 변경할 수 있습니다",
                        options=[f"SPEAKER_{i}" for i in range(6)],
                        required=True,
                    ),
                    "시작": st.column_config.TextColumn("시작", disabled=True),
                    "종료": st.column_config.TextColumn("종료", disabled=True),
                    "길이": st.column_config.TextColumn("길이", disabled=True),
                    "신뢰도": st.column_config.TextColumn("신뢰도", disabled=True)
                },
                use_container_width=True,
                hide_index=True,
                key="speaker_segments_editor"
            )
            
            # 변경사항 적용 버튼
            if st.button("변경사항 적용", type="secondary"):
                # 편집된 데이터로 session_state 업데이트
                for i, row in edited_df.iterrows():
                    st.session_state.speaker_segments[i]['speaker'] = row['화자']
                st.success("화자 정보가 업데이트되었습니다!")
            
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
                
                if st.button(f"{selected_speaker}의 모든 구간 합치기", type="secondary"):
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