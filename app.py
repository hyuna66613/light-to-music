import streamlit as st
import cv2
import numpy as np
from scipy.io import wavfile
import io
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(layout="wide", page_title="GarageLight DAW")
st.title("🎹 GarageLight: Optical DAW (Multi-Track Mode)")

# --- 1. 사이드바 컨트롤 ---
with st.sidebar:
    st.header("🎛 Control Panel")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.info("광원별로 생성된 트랙을 선택하여 조합하고 다운로드하세요.")

if uploaded_file:
    try:
        # 영상 처리 설정 (Full Frame)
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        sample_rate = 22050 # 처리 속도와 안정성을 위해 조정
        max_tracks = 8 # 시각적 편의를 위해 8개 트랙으로 설정
        tracks_audio = [[] for _ in range(max_tracks)]
        tracks_visual = [[] for _ in range(max_tracks)]
        
        status_text = st.empty()
        status_text.write(f"🚀 {total_frames}프레임 분석 및 사운드 합성 중...")
        prog = st.progress(0)

        # 분석 루프
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            duration = 1.0 / fps
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_tracks]
            
            for idx, cnt in enumerate(sorted_contours):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                # 높이에 따른 음정 + 전자음(Square wave)
                freq = 150 + ((frame.shape[0] - cy) * 1.5)
                vol = min(area / 1500, 0.7)
                tone = vol * np.sign(np.sin(2 * np.pi * freq * t)) # Square Wave
                
                tracks_audio[idx].append(tone)
                tracks_visual[idx].append(vol)
            
            # 빈 트랙 채우기
            for j in range(len(sorted_contours), max_tracks):
                tracks_audio[j].append(np.zeros_like(t))
                tracks_visual[j].append(0)
            
            if i % 20 == 0: prog.progress(i / total_frames)
        
        prog.empty()
        status_text.success("✅ 오케스트라 레이어 생성 완료!")

        # UI 레이아웃
        col_vid, col_daw = st.columns([1, 2])
        
        with col_vid:
            st.header("📽 Input Video")
            st.video(uploaded_file)

        with col_daw:
            st.header("🎹 Timeline & Mixer")
            
            # --- 트랙 선택 기능 ---
            available_tracks = [f"Track {i+1}" for i in range(max_tracks) if any(tracks_visual[i])]
            
            col_sel1, col_sel2 = st.columns([3, 1])
            with col_sel1:
                selected_tracks = st.multiselect("조합할 악기(트랙)를 선택하세요:", available_tracks, default=available_tracks)
            with col_sel2:
                if st.button("전체 선택/해제"):
                    selected_tracks = available_tracks

            # 선택된 트랙들 합치기
            mixed_audio = None
            if selected_tracks:
                for t_name in selected_tracks:
                    idx = int(t_name.split()[1]) - 1
                    track_data = np.concatenate(tracks_audio[idx])
                    if mixed_audio is None:
                        mixed_audio = track_data
                    else:
                        # 길이 맞추기 및 믹싱
                        min_len = min(len(mixed_audio), len(track_data))
                        mixed_audio = mixed_audio[:min_len] + track_data[:min_len]

            # --- 마스터 출력부 ---
            if mixed_audio is not None:
                st.subheader("🎚 Master Output (Selected Tracks Mixed)")
                # 피크 방지 (노멀라이징)
                mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.8
                st.audio(mixed_audio, sample_rate=sample_rate)
                
                buf = io.BytesIO()
                wavfile.write(buf, sample_rate, (mixed_audio * 32767).astype(np.int16))
                st.download_button(f"⬇️ 선택한 {len(selected_tracks)}개 악기 조합 다운로드 (WAV)", buf.getvalue(), "mixed_lights.wav")

            st.divider()

            # --- 개별 트랙 타임라인 ---
            for i, name in enumerate(available_tracks):
                idx = int(name.split()[1]) - 1
                with st.expander(f"🎵 {name} Details", expanded=True):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=tracks_visual[idx], fill='tozeroy', line_color='#00d1ff'))
                    fig.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), xaxis_visible=False, yaxis_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
