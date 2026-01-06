import streamlit as st
import cv2
import numpy as np
from scipy.io import wavfile
import io
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Musical Light DAW")
st.title("🎹 Harmonic Synth DAW (Error Fixed)")

# --- 음악적 설정 ---
NOTES = [130.81, 155.56, 174.61, 196.00, 233.08, 
         261.63, 311.13, 349.23, 392.00, 466.16, 
         523.25, 622.25, 698.46, 783.99, 932.33]

def get_nearest_note(freq):
    return min(NOTES, key=lambda x: abs(x - freq))

def apply_envelope(tone, sample_rate):
    n = len(tone)
    if n < 100: return tone
    attack = int(min(sample_rate * 0.01, n * 0.1))
    release = int(min(sample_rate * 0.05, n * 0.2))
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return tone * env

with st.sidebar:
    st.header("🎛 Control Panel")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1 or np.isnan(fps): fps = 30 # FPS 예외 처리
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = 22050 
        max_tracks = 6
        
        # 오디오 데이터를 프레임 단위로 저장하지 않고 전체 스트림으로 관리
        master_l = np.zeros(int(sample_rate * (total_frames / fps)) + 100)
        master_r = np.zeros(int(sample_rate * (total_frames / fps)) + 100)
        
        tracks_visual = [[] for _ in range(max_tracks)]
        
        prog = st.progress(0)
        status_text = st.empty()

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 현재 프레임의 오디오가 시작될 위치 계산
            start_idx = int(i * (sample_rate / fps))
            duration = 1.0 / fps
            num_samples = int(sample_rate * duration)
            t = np.linspace(0, duration, num_samples, False)
            
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_tracks]
            
            for idx, cnt in enumerate(sorted_contours):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                note_freq = get_nearest_note(150 + ((frame.shape[0] - cy) * 1.5))
                vol = min(area / 1500, 0.5)
                
                # 기본음 + 배음 합성
                tone = vol * np.sin(2 * np.pi * note_freq * t)
                tone += (vol * 0.2) * np.sin(2 * np.pi * (note_freq * 2) * t)
                tone = apply_envelope(tone, sample_rate)
                
                # 스테레오 팬닝
                pan_r = cx / frame.shape[1]
                pan_l = 1 - pan_r
                
                # --- 핵심 해결책: 배열 크기를 맞춰서 가산 ---
                end_idx = start_idx + len(tone)
                if end_idx < len(master_l):
                    master_l[start_idx:end_idx] += tone * pan_l
                    master_r[start_idx:end_idx] += tone * pan_r
                
                tracks_visual[idx].append({'time': i/fps, 'freq': note_freq, 'vol': vol})
            
            if i % 30 == 0:
                prog.progress(i / total_frames)
                status_text.text(f"분석 중: {i}/{total_frames} 프레임")

        # 결과물 정리 (노멀라이징)
        master_stereo = np.vstack((master_l, master_r)).T
        max_val = np.max(np.abs(master_stereo))
        if max_val > 0:
            master_stereo = (master_stereo / max_val * 32767).astype(np.int16)

        # UI 출력
        col1, col2 = st.columns([1, 1])
        with col1:
            st.header("🎞 View & Play")
            st.video(uploaded_file)
            st.audio(master_stereo, sample_rate=sample_rate)
            
            buf = io.BytesIO()
            wavfile.write(buf, sample_rate, master_stereo)
            st.download_button("💾 Download Master (WAV)", buf.getvalue(), "musical_bus.wav")

        with col2:
            st.header("📊 Harmonic Timeline")
            for idx in range(max_tracks):
                if tracks_visual[idx]:
                    times = [v['time'] for v in tracks_visual[idx]]
                    freqs = [v['freq'] for v in tracks_visual[idx]]
                    fig = go.Figure(go.Scatter(x=times, y=freqs, mode='lines', line=dict(color='#00d1ff')))
                    fig.update_layout(height=100, margin=dict(l=0,r=0,t=10,b=10), xaxis_title="Time(s)", yaxis_title="Hz")
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
