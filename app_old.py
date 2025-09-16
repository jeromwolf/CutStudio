"""
CutStudio v5.0 - MVP with Streamlit
교육 영상 편집을 위한 3가지 핵심 기능
1. 화자 구분
2. 스마트 컷 (무음 제거)
3. 음성 인식 + 요약
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import time
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="CutStudio v5.0",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .feature-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF5252;
    }
    </style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'video_uploaded' not in st.session_state:
    st.session_state.video_uploaded = False
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'upload'
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = None

# 헤더
st.markdown('<h1 class="main-header">🎬 CutStudio v5.0</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">교육 영상 편집 시간을 10시간에서 15분으로!</p>', unsafe_allow_html=True)

# 3개 핵심 기능 소개
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
        <h3>👥 화자 구분</h3>
        <p>최대 10명까지 자동으로 화자를 구분하고 각자의 발언을 분리합니다.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
        <h3>✂️ 스마트 컷</h3>
        <p>침묵, 간투사를 자동으로 제거하여 영상 길이를 50% 단축합니다.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
        <h3>📝 음성→텍스트</h3>
        <p>자동 자막 생성과 AI 요약으로 핵심 내용을 텍스트로 추출합니다.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# 메인 워크플로우
# 처리 완료 후에는 결과 화면으로 전환
if st.session_state.processing_complete:
    # 🎉 결과 화면 전체 표시
    st.header("🎉 처리 완료!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("👥 화자 분석")
        if st.session_state.results.get('speakers'):
            for speaker in st.session_state.results['speakers']:
                with st.container():
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #FF6B6B;">
                    <h4 style="margin: 0; color: #FF6B6B;">{speaker['name']}</h4>
                    <p style="margin: 0.5rem 0;"><strong>발화 시간:</strong> {speaker['duration']//60:02d}분 {int(speaker['duration']%60):02d}초</p>
                    <p style="margin: 0;"><strong>비율:</strong> {speaker['percentage']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("화자 분석을 선택하지 않았습니다.")

    with col2:
        st.subheader("✂️ 편집 결과")
        if st.session_state.results.get('smart_cut'):
            cut_data = st.session_state.results['smart_cut']
            original_duration = cut_data['original_duration']
            cut_duration = cut_data['cut_duration']
            reduction = ((original_duration - cut_duration) / original_duration) * 100

            # 큰 메트릭으로 표시
            col2_1, col2_2, col2_3 = st.columns(3)

            with col2_1:
                st.metric(
                    "원본 길이",
                    f"{int(original_duration//60):02d}분 {int(original_duration%60):02d}초"
                )

            with col2_2:
                st.metric(
                    "편집 후",
                    f"{int(cut_duration//60):02d}분 {int(cut_duration%60):02d}초",
                    f"-{reduction:.0f}%"
                )

            with col2_3:
                st.metric("제거된 침묵", f"{cut_data.get('removed_silence', 0)}개")
        else:
            st.info("스마트 컷을 선택하지 않았습니다.")

    # 다운로드 섹션
    st.markdown("---")
    st.subheader("💾 결과 다운로드")

    download_col1, download_col2, download_col3, download_col4 = st.columns(4)

    with download_col1:
        if st.button("📹 편집된 영상", type="primary", use_container_width=True):
            st.success("영상 다운로드 시작!")

    with download_col2:
        if st.button("📝 자막 파일", use_container_width=True):
            st.success("자막 다운로드 시작!")

    with download_col3:
        if st.button("📄 요약 문서", use_container_width=True):
            st.success("요약 다운로드 시작!")

    with download_col4:
        if st.button("🔄 새로 시작", use_container_width=True):
            # 상태 초기화
            for key in list(st.session_state.keys()):
                if key not in ['gemini_api_key']:  # API 키는 유지
                    del st.session_state[key]
            st.rerun()

else:
    # 🔧 작업 화면
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Step 1: 동영상 선택
        st.header("📥 Step 1: 동영상 선택")

        # 탭으로 업로드 방식 구분
        tab1, tab2 = st.tabs(["📁 파일 업로드", "🔗 YouTube URL"])

        with tab1:
        st.markdown("**컴퓨터에서 파일을 직접 업로드하세요**")
        uploaded_file = st.file_uploader(
            "교육 영상을 업로드하세요 (최대 2GB)",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="MP4, AVI, MOV, MKV 형식을 지원합니다"
        )

        if uploaded_file is not None:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
                st.session_state.video_uploaded = True

            # 비디오 정보 표시
            st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("파일 크기", f"{uploaded_file.size / (1024*1024):.1f} MB")
            with col2:
                st.metric("파일 형식", uploaded_file.type.split('/')[-1].upper())
            with col3:
                st.metric("업로드 시간", datetime.now().strftime("%H:%M:%S"))

            # 비디오 미리보기
            st.video(uploaded_file)

    with tab2:
        st.markdown("**YouTube 영상을 URL로 다운로드하세요**")

        # YouTube URL 입력
        youtube_url = st.text_input(
            "YouTube URL을 입력하세요",
            placeholder="https://www.youtube.com/watch?v=...",
            help="YouTube, YouTube Shorts URL을 지원합니다"
        )

        if youtube_url:
            if st.button("📥 YouTube 영상 다운로드", type="primary"):
                try:
                    # YouTube 다운로더 import
                    from youtube_downloader import YouTubeDownloader

                    downloader = YouTubeDownloader()

                    # 다운로드 진행 상황 표시
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    with st.spinner("YouTube 영상 정보를 가져오는 중..."):
                        status_text.text("📡 영상 정보 확인 중...")
                        progress_bar.progress(20)

                        # 영상 정보 가져오기
                        video_info = downloader.get_video_info(youtube_url)

                        if video_info:
                            st.success(f"✅ 영상 발견: {video_info['title'][:50]}...")

                            # 다운로드 품질 선택 (형식 기반)
                            quality_options = {
                                "최고 품질": None,  # 기본값 사용
                                "720p": "best[height<=720]/best",
                                "480p": "best[height<=480]/best",
                                "360p": "best[height<=360]/best"
                            }
                            selected_quality_label = st.selectbox("다운로드 품질 선택", list(quality_options.keys()), index=1)
                            selected_format = quality_options[selected_quality_label]

                            status_text.text("⬇️ 다운로드 중...")
                            progress_bar.progress(60)

                            # 실제 다운로드
                            video_path = downloader.download_video(youtube_url, format_id=selected_format)
                            progress_bar.progress(100)

                            if video_path and os.path.exists(video_path):
                                st.session_state.video_path = video_path
                                st.session_state.video_uploaded = True

                                # 다운로드 완료 정보 표시
                                st.success(f"🎉 다운로드 완료!")

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    file_size = os.path.getsize(video_path) / (1024*1024)
                                    st.metric("파일 크기", f"{file_size:.1f} MB")
                                with col2:
                                    st.metric("품질", selected_quality_label)
                                with col3:
                                    st.metric("다운로드 시간", datetime.now().strftime("%H:%M:%S"))

                                # 비디오 미리보기
                                st.video(video_path)
                                status_text.empty()
                            else:
                                st.error("❌ 다운로드 실패. URL을 확인해주세요.")
                        else:
                            st.error("❌ 영상 정보를 가져올 수 없습니다. URL을 확인해주세요.")

                except Exception as e:
                    st.error(f"❌ 오류 발생: {str(e)}")
                    st.error("YouTube URL이 올바른지, 네트워크 연결을 확인해주세요.")
                    # progress_bar와 status_text가 정의되어 있을 때만 정리
                    try:
                        progress_bar.empty()
                        status_text.empty()
                    except:
                        pass

    # Step 2: 처리 옵션 선택
    if st.session_state.video_uploaded:
        st.markdown("---")
        st.header("⚙️ Step 2: 처리 옵션 선택")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("기본 설정")

            # 화자 감지
            enable_speaker = st.checkbox("👥 화자 구분", value=True,
                help="여러 명이 말하는 영상에서 화자를 자동으로 구분합니다")
            if enable_speaker:
                max_speakers = st.slider("최대 화자 수", 2, 10, 3)

            # 스마트 컷
            enable_smart_cut = st.checkbox("✂️ 스마트 컷", value=True,
                help="침묵과 불필요한 부분을 자동으로 제거합니다")
            if enable_smart_cut:
                silence_threshold = st.slider("침묵 감지 민감도", 1, 5, 3)
                cut_mode = st.radio("제거 강도",
                    ["보수적 (조금만 제거)", "균형 (적당히 제거)", "공격적 (많이 제거)"],
                    index=1)

        with col2:
            st.subheader("AI 기능")

            # 음성 인식
            enable_transcription = st.checkbox("🎤 음성 인식", value=True,
                help="음성을 텍스트로 변환합니다")
            if enable_transcription:
                language = st.selectbox("언어 선택",
                    ["자동 감지", "한국어", "영어", "일본어", "중국어"])

            # AI 요약
            enable_summary = st.checkbox("🤖 AI 요약", value=False,
                help="Google Gemini를 사용해 내용을 자동으로 요약합니다")

            gemini_api_key = None
            if enable_summary:
                st.info("💡 AI 요약을 사용하려면 Google Gemini API 키가 필요합니다.")
                gemini_api_key = st.text_input(
                    "🔑 Gemini API 키를 입력하세요",
                    type="password",
                    help="https://aistudio.google.com/app/apikey 에서 무료로 발급받을 수 있습니다"
                )

                if gemini_api_key:
                    st.session_state.gemini_api_key = gemini_api_key
                    st.success("✅ API 키가 입력되었습니다")
                    summary_length = st.select_slider("요약 길이",
                        options=["매우 짧게", "짧게", "보통", "길게", "매우 길게"],
                        value="보통")
                else:
                    st.warning("⚠️ API 키를 입력해야 AI 요약 기능을 사용할 수 있습니다")

        st.markdown("---")

        # Step 3: 처리 시작
        st.header("🚀 Step 3: 처리 시작")

        # 예상 시간 계산
        estimated_time = 5  # 기본 5분
        if enable_speaker:
            estimated_time += 2
        if enable_smart_cut:
            estimated_time += 1
        if enable_transcription:
            estimated_time += 3
        if enable_summary:
            estimated_time += 1

        st.info(f"⏱️ 예상 처리 시간: 약 {estimated_time}분")

        if st.button("🎬 처리 시작", type="primary", use_container_width=True):
            st.session_state.current_step = 'processing'

            # 프로그레스 바
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 처리 시뮬레이션 (실제 구현 시 실제 처리 함수 호출)
            steps = []
            if enable_speaker:
                steps.append(("화자 감지 중...", 30))
            if enable_smart_cut:
                steps.append(("스마트 컷 처리 중...", 20))
            if enable_transcription:
                steps.append(("음성 인식 중...", 30))
            if enable_summary:
                steps.append(("AI 요약 생성 중...", 20))

            current_progress = 0
            for step_name, step_progress in steps:
                status_text.text(f"🔄 {step_name}")
                for i in range(step_progress):
                    current_progress += 1
                    progress_bar.progress(current_progress / 100)
                    time.sleep(0.05)  # 시뮬레이션

            progress_bar.progress(100)
            status_text.text("✅ 처리 완료!")
            st.session_state.processing_complete = True

            # 더미 결과 생성
            st.session_state.results = {
                'speakers': [
                    {'id': 1, 'name': '화자 1', 'duration': 1200, 'percentage': 60},
                    {'id': 2, 'name': '화자 2', 'duration': 600, 'percentage': 30},
                    {'id': 3, 'name': '화자 3', 'duration': 200, 'percentage': 10},
                ] if enable_speaker else [],
                'smart_cut': {
                    'original_duration': 2000,
                    'cut_duration': 1200,
                    'removed_silence': 500,
                    'removed_filler': 300
                } if enable_smart_cut else None,
                'transcript': "안녕하세요. 오늘은 CutStudio에 대해 설명드리겠습니다..." if enable_transcription else None,
                'summary': "CutStudio는 교육 영상 편집을 자동화하는 도구입니다. 화자 구분, 스마트 컷, 음성 인식 기능을 제공합니다." if enable_summary else None
            }

            st.balloons()

    # 오른쪽 사이드바: 도움말
    with col_right:
        st.header("💡 도움말")

        st.info("""
        **처리 순서:**
        1. 영상 업로드 또는 YouTube URL
        2. 원하는 기능 선택
        3. 처리 시작 버튼 클릭
        4. 결과 확인 및 다운로드
        """)

        st.success("""
        **추천 설정:**
        - 화자가 많으면 'Balanced' 모드
        - 빠른 처리가 필요하면 'Fast' 모드
        - 최고 품질은 'Accurate' 모드
        """)

        if st.session_state.video_uploaded:
            st.warning("""
            **주의사항:**
            - 처리 중에는 페이지를 새로고침하지 마세요
            - 큰 파일은 시간이 오래 걸릴 수 있습니다
            """)

        st.markdown("---")
        st.markdown("**🔗 관련 링크**")
        st.markdown("- [Gemini API 키 발급](https://aistudio.google.com/app/apikey)")
        st.markdown("- [Hugging Face 토큰](https://huggingface.co/settings/tokens)")

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; padding: 2rem 0;">
    <p>CutStudio v5.0 MVP | Made with ❤️ for Educators</p>
    <p>문의: contact@cutstudio.ai</p>
</div>
""", unsafe_allow_html=True)