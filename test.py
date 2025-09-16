import streamlit as st

st.title("🎬 CutStudio 테스트")
st.write("Streamlit이 작동합니다!")

uploaded_file = st.file_uploader("파일 선택", type=['mp4'])
if uploaded_file:
    st.success("파일 업로드 성공!")

if st.button("테스트 버튼"):
    st.balloons()
    st.write("버튼이 작동합니다!")