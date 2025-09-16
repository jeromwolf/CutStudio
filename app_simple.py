"""
CutStudio v5.0 - 간단한 MVP 버전
천천히 단계별로 구현
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from datetime import datetime

# 필요한 경우에만 라이브러리 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv가 없어도 계속 진행

# 페이지 설정
st.set_page_config(
    page_title="CutStudio v5.0",
    page_icon="🎬",
    layout="wide"
)

# 세션 상태 초기화
if 'video_uploaded' not in st.session_state:
    st.session_state.video_uploaded = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

# 헤더
st.title("🎬 CutStudio v5.0")
st.markdown("**교육 영상 편집 시간을 10시간에서 15분으로!**")
st.markdown("---")

# Step 1: 동영상 선택
st.header("📥 Step 1: 동영상 선택")

tab1, tab2 = st.tabs(["📁 파일 업로드", "🔗 YouTube URL"])

with tab1:
    st.markdown("**컴퓨터에서 파일을 직접 업로드하세요**")

    uploaded_file = st.file_uploader(
        "교육 영상을 업로드하세요",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="MP4, AVI, MOV, MKV 형식을 지원합니다"
    )

    if uploaded_file is not None:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state.video_path = tmp_file.name
            st.session_state.video_uploaded = True

        st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")

        # 파일 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("파일 크기", f"{uploaded_file.size / (1024*1024):.1f} MB")
        with col2:
            st.metric("파일 형식", uploaded_file.type.split('/')[-1].upper())
        with col3:
            st.metric("업로드 시간", datetime.now().strftime("%H:%M:%S"))

with tab2:
    st.markdown("**YouTube 영상을 URL로 다운로드하세요**")

    youtube_url = st.text_input(
        "YouTube URL을 입력하세요",
        placeholder="https://www.youtube.com/watch?v=...",
        help="YouTube URL을 지원합니다"
    )

    if youtube_url:
        if st.button("📥 YouTube 영상 다운로드", type="primary"):
            try:
                import time
                start_time = time.time()

                # YouTube 다운로더 import
                from youtube_downloader import YouTubeDownloader

                downloader = YouTubeDownloader()

                # 진행 상황 표시
                progress_bar = st.progress(0)
                status_text = st.empty()

                with st.spinner("YouTube 영상 정보를 가져오는 중..."):
                    status_text.text("📡 영상 정보 확인 중...")
                    progress_bar.progress(20)

                    # 영상 정보 가져오기
                    video_info = downloader.get_video_info(youtube_url)

                    if video_info:
                        st.success(f"✅ 영상 발견: {video_info['title'][:50]}...")

                        status_text.text("⬇️ 다운로드 중...")
                        progress_bar.progress(60)

                        # 실제 다운로드 (기본 품질)
                        video_path = downloader.download_video(youtube_url)
                        progress_bar.progress(100)

                        if video_path and os.path.exists(video_path):
                            st.session_state.video_path = video_path
                            st.session_state.video_uploaded = True

                            # 다운로드 소요 시간 계산
                            end_time = time.time()
                            download_time = end_time - start_time

                            # 다운로드 완료 정보 표시
                            st.success("🎉 다운로드 완료!")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                file_size = os.path.getsize(video_path) / (1024*1024)
                                st.metric("파일 크기", f"{file_size:.1f} MB")
                            with col2:
                                # 비디오 길이 표시
                                try:
                                    import subprocess
                                    import json
                                    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_path]
                                    result = subprocess.run(cmd, capture_output=True, text=True)
                                    if result.returncode == 0:
                                        data = json.loads(result.stdout)
                                        duration = float(data['format']['duration'])
                                        duration_min = int(duration // 60)
                                        duration_sec = int(duration % 60)
                                        st.metric("영상 길이", f"{duration_min}분 {duration_sec}초")
                                    else:
                                        st.metric("영상 제목", video_info['title'][:20] + "...")
                                except:
                                    st.metric("영상 제목", video_info['title'][:20] + "...")
                            with col3:
                                st.metric("다운로드 시간", f"{download_time:.1f}초")

                            status_text.empty()
                            progress_bar.empty()
                        else:
                            st.error("❌ 다운로드 실패. URL을 확인해주세요.")
                    else:
                        st.error("❌ 영상 정보를 가져올 수 없습니다. URL을 확인해주세요.")

            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                st.error("YouTube URL이 올바른지, yt-dlp가 설치되어 있는지 확인해주세요.")

# Step 2: 옵션 선택 (비디오가 업로드된 경우에만)
if st.session_state.video_uploaded:
    st.markdown("---")
    st.header("⚙️ Step 2: 처리 옵션 선택")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("기본 기능")

        # 화자 구분
        enable_speaker = st.checkbox("👥 화자 구분", value=True)
        if enable_speaker:
            max_speakers = st.slider("최대 화자 수", 2, 10, 3)
            speaker_mode = st.selectbox("처리 모드", ["Fast", "Balanced", "Accurate"])

        # 스마트 컷
        enable_cut = st.checkbox("✂️ 스마트 컷", value=True)
        if enable_cut:
            cut_mode = st.selectbox("컷 모드", ["Conservative", "Balanced", "Aggressive"])

    with col2:
        st.subheader("고급 기능")

        # 음성 인식
        enable_transcription = st.checkbox("📝 음성 인식", value=True)
        if enable_transcription:
            language = st.selectbox("언어", ["자동 감지", "한국어", "영어"])

        # AI 요약
        enable_summary = st.checkbox("🤖 AI 요약", value=False)

        if enable_summary:
            st.info("💡 AI 요약을 사용하려면 Gemini API 키가 필요합니다.")
            gemini_api_key = st.text_input(
                "🔑 Gemini API 키",
                type="password",
                help="https://aistudio.google.com/app/apikey"
            )

            if gemini_api_key:
                st.success("✅ API 키가 입력되었습니다")

    # Step 3: 처리 시작
    st.markdown("---")
    st.header("🚀 Step 3: 처리 시작")

    if st.button("🎬 처리 시작", type="primary", use_container_width=True):
        st.success("처리를 시작합니다!")

        results = {}

        # 화자 감지 실행
        if enable_speaker:
            with st.spinner("👥 화자 구분 중..."):
                try:
                    from modules.speaker_detector import RealSpeakerDetector

                    # Hugging Face 토큰 가져오기 (.env에서)
                    hf_token = os.getenv('HUGGINGFACE_TOKEN')

                    detector = RealSpeakerDetector(
                        max_speakers=max_speakers,
                        huggingface_token=hf_token
                    )

                    # 실제 화자 감지 실행
                    speaker_result = detector.detect_speakers(
                        st.session_state.video_path,
                        mode=speaker_mode.lower()
                    )

                    if speaker_result['success']:
                        results['speaker_detection'] = speaker_result
                        st.success(f"✅ 화자 감지 완료 ({speaker_result['method']})")
                    else:
                        st.warning("⚠️ 화자 감지에서 문제가 발생했습니다.")

                except Exception as e:
                    st.error(f"❌ 화자 감지 오류: {str(e)}")

        # 스마트 컷 실행
        if enable_cut:
            with st.spinner("✂️ 스마트 컷 분석 중..."):
                try:
                    from modules.real_smart_cutter import RealSmartCutter

                    cutter = RealSmartCutter(mode=cut_mode.lower())

                    # 실제 침묵 구간 분석
                    cut_result = cutter.detect_silence(st.session_state.video_path)

                    if cut_result['success']:
                        results['smart_cut'] = cut_result
                        st.success(f"✅ 스마트 컷 분석 완료 ({cut_result['method']})")
                    else:
                        st.warning("⚠️ 스마트 컷 분석에서 문제가 발생했습니다.")

                except Exception as e:
                    st.error(f"❌ 스마트 컷 오류: {str(e)}")

        # 음성 인식 실행
        if enable_transcription:
            with st.spinner("📝 음성 인식 중..."):
                try:
                    # 간단한 Whisper 기반 음성 인식
                    import subprocess
                    import json

                    # 오디오 추출
                    audio_path = tempfile.mktemp(suffix='.wav')
                    cmd = [
                        'ffmpeg', '-i', st.session_state.video_path,
                        '-acodec', 'pcm_s16le', '-ar', '16000',
                        '-ac', '1', '-y', audio_path
                    ]
                    subprocess.run(cmd, capture_output=True)

                    if os.path.exists(audio_path):
                        try:
                            import whisper
                            model = whisper.load_model("tiny")
                            result = model.transcribe(audio_path, language=language.lower() if language != "자동 감지" else None)

                            # SRT 형식으로 변환
                            srt_content = ""
                            for i, segment in enumerate(result['segments'], 1):
                                start = segment['start']
                                end = segment['end']
                                text = segment['text'].strip()

                                # SRT 시간 형식
                                start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
                                end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"

                                srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"

                            results['transcription'] = {
                                'text': result['text'],
                                'srt': srt_content,
                                'segments': result['segments']
                            }
                            st.success("✅ 음성 인식 완료")

                            # 오디오 파일 정리
                            os.unlink(audio_path)

                        except ImportError:
                            st.warning("⚠️ Whisper가 설치되지 않았습니다. pip install openai-whisper")
                except Exception as e:
                    st.error(f"❌ 음성 인식 오류: {str(e)}")

        st.balloons()
        st.success("🎉 처리가 완료되었습니다!")

        # 실제 결과 표시
        st.subheader("📊 처리 결과")

        col1, col2 = st.columns(2)

        with col1:
            # 화자 분석 결과
            if results.get('speaker_detection'):
                st.subheader("👥 화자 분석")
                speaker_data = results['speaker_detection']

                st.write(f"**🎯 감지 방법:** {speaker_data['method']}")
                st.write(f"**⏱️ 영상 길이:** {speaker_data['duration']:.1f}초")
                st.write(f"**👥 화자 수:** {speaker_data['speaker_count']}명")

                # 화자별 상세 정보
                if speaker_data['stats']:
                    st.write("**화자별 분석:**")
                    for speaker_id, stats in speaker_data['stats'].items():
                        duration_min = int(stats['total_duration'] // 60)
                        duration_sec = int(stats['total_duration'] % 60)
                        st.write(f"- **{speaker_id}**: {duration_min}분 {duration_sec}초 ({stats['percentage']:.1f}%)")

        with col2:
            # 스마트 컷 결과
            if results.get('smart_cut'):
                st.subheader("✂️ 스마트 컷 분석")
                cut_data = results['smart_cut']

                st.write(f"**🎯 분석 방법:** {cut_data['method']}")

                # 주요 메트릭
                original_min = int(cut_data['original_duration'] // 60)
                original_sec = int(cut_data['original_duration'] % 60)
                speech_min = int(cut_data['speech_duration'] // 60)
                speech_sec = int(cut_data['speech_duration'] % 60)
                silence_min = int(cut_data['silence_duration'] // 60)
                silence_sec = int(cut_data['silence_duration'] % 60)

                st.metric("원본 길이", f"{original_min}분 {original_sec}초")
                st.metric("음성 구간", f"{speech_min}분 {speech_sec}초")
                st.metric("침묵 구간", f"{silence_min}분 {silence_sec}초",
                         f"-{cut_data['reduction_rate']*100:.1f}%")

                # 간투사 정보 표시
                if cut_data.get('filler_count'):
                    filler_duration = cut_data.get('filler_duration', 0)
                    filler_min = int(filler_duration // 60)
                    filler_sec = int(filler_duration % 60)
                    st.metric("간투사 제거", f"{filler_min}분 {filler_sec}초",
                             f"{cut_data.get('filler_count', 0)}개")

                    # 전체 제거 가능 시간
                    if cut_data.get('total_removable'):
                        total_removable = cut_data['total_removable']
                        removable_min = int(total_removable // 60)
                        removable_sec = int(total_removable % 60)
                        new_reduction = cut_data.get('new_reduction_rate', 0) * 100
                        st.info(f"💡 총 제거 가능: {removable_min}분 {removable_sec}초 (-{new_reduction:.1f}%)")
                else:
                    st.metric("침묵 개수", f"{cut_data['silence_count']}개")

        if not results:
            st.info("처리할 기능을 선택해주세요.")

        # 처리 결과가 있을 때 추가 작업 섹션
        if results:
            st.markdown("---")
            st.subheader("📥 다운로드 & 추가 작업")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("💾 편집된 영상", type="primary", use_container_width=True,
                           help="침묵과 간투사가 제거된 영상을 다운로드합니다",
                           key="download_edited_video"):
                    if results.get('smart_cut'):
                        with st.spinner("🎬 영상 편집 중... (시간이 걸릴 수 있습니다)"):
                            try:
                                from modules.real_smart_cutter import RealSmartCutter

                                cutter = RealSmartCutter()
                                cut_data = results['smart_cut']

                                # 침묵 구간으로 비디오 편집
                                edited_path = cutter.cut_video(
                                    st.session_state.video_path,
                                    cut_data.get('silence_segments', [])
                                )

                                if edited_path and os.path.exists(edited_path):
                                    st.success("✅ 영상 편집 완료!")

                                    # 다운로드 버튼 제공
                                    with open(edited_path, 'rb') as f:
                                        video_bytes = f.read()

                                    st.download_button(
                                        label="📥 편집된 영상 다운로드",
                                        data=video_bytes,
                                        file_name=f"edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                        mime="video/mp4"
                                    )
                                else:
                                    st.error("❌ 영상 편집 실패")
                            except Exception as e:
                                st.error(f"❌ 편집 오류: {str(e)}")
                    else:
                        st.warning("⚠️ 먼저 스마트 컷 분석을 실행해주세요")

            with col2:
                if st.button("📝 자막 파일", use_container_width=True,
                           help="SRT 형식의 자막 파일을 다운로드합니다",
                           key="download_subtitle"):
                    if results.get('transcription'):
                        srt_content = results['transcription']['srt']
                        st.download_button(
                            label="📥 SRT 자막 다운로드",
                            data=srt_content,
                            file_name=f"subtitle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt",
                            mime="text/plain"
                        )
                    else:
                        st.warning("⚠️ 먼저 음성 인식을 실행해주세요")

            with col3:
                if st.button("📊 분석 리포트", use_container_width=True,
                           help="화자 분석과 편집 결과를 PDF로 다운로드합니다",
                           key="download_report"):
                    # 간단한 텍스트 리포트 생성
                    report = "=== CutStudio 분석 리포트 ===\n\n"
                    report += f"📅 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

                    if results.get('speaker_detection'):
                        data = results['speaker_detection']
                        report += "👥 화자 분석 결과\n"
                        report += f"- 감지 방법: {data['method']}\n"
                        report += f"- 영상 길이: {data['duration']:.1f}초\n"
                        report += f"- 화자 수: {data['speaker_count']}명\n\n"

                    if results.get('smart_cut'):
                        data = results['smart_cut']
                        report += "✂️ 스마트 컷 분석 결과\n"
                        report += f"- 원본 길이: {data['original_duration']:.1f}초\n"
                        report += f"- 침묵 구간: {data['silence_duration']:.1f}초\n"
                        report += f"- 단축률: {data['reduction_rate']*100:.1f}%\n"

                    st.download_button(
                        label="📥 리포트 다운로드",
                        data=report,
                        file_name=f"cutstudio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

            with col4:
                if st.button("🔄 새 영상 분석", use_container_width=True,
                           help="새로운 영상을 분석합니다",
                           key="new_analysis"):
                    # 세션 상태 초기화
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

            # 추가 옵션들
            with st.expander("🎯 고급 옵션", expanded=False):
                st.subheader("📤 내보내기 설정")

                export_col1, export_col2 = st.columns(2)

                with export_col1:
                    export_format = st.selectbox(
                        "출력 형식",
                        ["MP4 (H.264)", "MOV", "AVI", "MKV"],
                        help="편집된 영상의 출력 형식을 선택합니다"
                    )

                    export_quality = st.select_slider(
                        "출력 품질",
                        options=["낮음", "보통", "높음", "최고"],
                        value="높음",
                        help="출력 영상의 품질을 설정합니다"
                    )

                with export_col2:
                    export_resolution = st.selectbox(
                        "해상도",
                        ["원본 유지", "1080p", "720p", "480p"],
                        help="출력 영상의 해상도를 선택합니다"
                    )

                    export_fps = st.selectbox(
                        "프레임레이트",
                        ["원본 유지", "60fps", "30fps", "24fps"],
                        help="출력 영상의 프레임레이트를 선택합니다"
                    )

                st.info(f"""
                📌 **현재 설정:**
                - 형식: {export_format}
                - 품질: {export_quality}
                - 해상도: {export_resolution}
                - FPS: {export_fps}
                """)

            # 타임라인 미리보기 (선택사항)
            with st.expander("📽️ 타임라인 미리보기", expanded=False):
                st.info("🚧 타임라인 시각화 기능은 구현 중입니다")

                # 임시 타임라인 표시
                if results.get('speaker_detection'):
                    st.write("**화자별 타임라인:**")
                    speaker_data = results['speaker_detection']
                    for speaker_id, stats in speaker_data.get('stats', {}).items():
                        progress = stats['percentage'] / 100
                        st.progress(progress)
                        st.caption(f"{speaker_id}: {stats['percentage']:.1f}%")

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999;">
Made with ❤️ by Kelly | CutStudio v5.0 MVP
</div>
""", unsafe_allow_html=True)