import streamlit as st
import cv2
import numpy as np
from scipy.io import wavfile
import io
import plotly.graph_objects as go
from pydub import AudioSegment # MP3 변환용

# 페이지 설정
st.set_page_config(layout="wide", page_title="GarageLight DAW")
st.title("🎹 GarageLight: Optical Digital Audio Workstation")

# --- 1. 사이드바 정보창 ---
with st.sidebar:
    st.header("🎛 Control Panel")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.info("개별 광원을 트래킹하여 독립적인 신시사이저 트랙을 생성합니다.")

if uploaded_file:
    try:
        # 영상 처리 설정 (Full Frame 모드)
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        sample_rate = 44100
        # 최대 10개의 독립 광원 트랙 생성 (서버 부하 방지)
        max_tracks = 10
        tracks_audio = [[] for _ in range(max_tracks)]
        tracks_visual = [[] for _ in range(max_tracks)]
        
        st.write(f"🚀 {total_frames}프레임 전체 분석 중... (전자음악 모드)")
        prog = st.progress(0)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            # 광원 분석
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 한 프레임의 지속 시간
            duration = 1.0 / fps
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # 상위 10개 광원만 트래킹
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_tracks]
            
            for idx, cnt in enumerate(sorted_contours):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                # 색상 추출 (B, G, R)
                b, g, r = frame[cy, cx]
                
                # 매핑 공식 (위치->음정, 색상->음색, 밝기->볼륨)
                base_freq = 200 + ( (height - cy) * 2 ) # 높을수록 고음
                # 색상에 따른 배음 추가 (전자음 효과)
                freq = base_freq + (r * 0.5)
                vol = min((area / 1000) * (np.mean([r, g, b]) / 255), 1.0)
                
                # 사각파(Square Wave) 생성 - 더 전자음악스러운 소리
                tone = vol * np.sign(np.sin(2 * np.pi * freq * t))
                
                tracks_audio[idx].append(tone)
                tracks_visual[idx].append(vol)
            
            # 광원이 없는 트랙은 침묵 처리
            for j in range(len(sorted_contours), max_tracks):
                tracks_audio[j].append(np.zeros_like(t))
                tracks_visual[j].append(0)
                
            if i % 10 == 0: prog.progress(i / total_frames)

        # UI 배치 (영상 창 / DAW 타임라인 창)
        col_vid, col_daw = st.columns([1, 2])
        
        with col_vid:
            st.header("📽 Input Source")
            st.video(uploaded_file)
            st.metric("Resolution", f"{width}x{height}")
            st.metric("Frame Count", total_frames)

        with col_daw:
            st.header("🎹 GarageLight DAW Timeline")
            
            for idx in range(max_tracks):
                if any(tracks_visual[idx]):
                    with st.container():
                        st.markdown(f"**Track {idx+1}: Optical Oscillator**")
                        # 타임라인 파형 시각화
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=tracks_visual[idx], fill='tozeroy', line_color='#007AFF')) # 개러지밴드 블루
                        fig.update_layout(height=80, margin=dict(l=0, r=0, t=0, b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        
                        # 오디오 및 MP3 다운로드
                        full_audio = np.concatenate(tracks_audio[idx])
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.audio(full_audio, sample_rate=sample_rate)
                        with c2:
                            # MP3 변환
                            buf = io.BytesIO()
                            wavfile.write(buf, sample_rate, (full_audio * 32767).astype(np.int16))
                            audio_seg = AudioSegment.from_wav(io.BytesIO(buf.getvalue()))
                            mp3_buf = io.BytesIO()
                            audio_seg.export(mp3_buf, format="mp3")
                            st.download_button("MP3", mp3_buf.getvalue(), f"track_{idx+1}.mp3")

    except Exception as e:
        st.error(f"오류 발생: {e}. 'pydub' 라이브러리와 'ffmpeg'가 필요할 수 있습니다.")
