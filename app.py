import streamlit as st
import cv2
import numpy as np
from scipy.io import wavfile
import io
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Musical Light DAW")
st.title("🎹 Harmonic Synth DAW (Audio Clipping Fixed)")

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
        if fps < 1 or np.isnan(fps): fps = 30
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = 22050 
        max_tracks = 6
        
        # 넉넉한 길이의 마스터 배열 생성
        audio_len = int(sample_rate * (total_frames / fps)) + sample_rate
        master_l = np.zeros(audio_len)
        master_r = np.zeros(audio_len)
        
        tracks_visual = [[] for _ in range(max_tracks)]
        
        prog = st.progress(0)
        status_text = st.empty()

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_idx = int(i * (sample_rate / fps))
            duration = 1.0 / fps
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_tracks]
            
            for idx, cnt in enumerate(sorted_contours):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                note_freq = get_nearest_note(150 + ((frame.shape[0] - cy) * 1.5))
                vol = min(area / 1500, 0.5)
                
                tone = vol * np.sin(2 * np.pi * note_freq * t)
                tone += (vol * 0.2) * np.sin(2 * np.pi * (note_freq * 2) * t)
                tone = apply_envelope(tone, sample_rate)
                
                pan_r = cx / frame.shape[1]
                pan_l = 1 - pan_r
                
                end_idx = start_idx + len(tone)
                if end_idx < audio_len:
                    master_l[start_idx:end_idx] += tone * pan_l
                    master_r[start_idx:end_idx] += tone * pan_r
                
                tracks_visual[idx].append({'time': i/fps, 'freq': note_freq})
            
            if i % 30 == 0:
                prog.progress(min(i / total_frames, 1.0))

        # --- 핵심 해결책: 안전한 노멀라이징 ---
        master_stereo = np.vstack((master_l, master_r)).T
        
        # 1. 절대값 기준 가장 큰 소리를 찾습니다.
        max_val = np.max(np.abs(master_stereo))
        
        if max_val > 0:
            # 2. 모든 소리를 -1.0 ~ 1.0 사이로 압축합니다. (Clipping 방지)
            master_normalized = master_stereo / max_val
            # 3. 16비트 오디오 범위(-32768 ~ 32767)로 안전하게 변환합니다.
            master_final = (master_normalized * 32767).astype(np.int16)
        else:
            master_final = master_stereo.astype(np.int16)

        # UI 출력
        col1, col2 = st.columns([1, 1])
        with col1:
            st.header("🎞 View & Play")
            st.video(uploaded_file)
            st.audio(master_final, sample_rate=sample_rate)
            
            buf = io.BytesIO()
            wavfile.write(buf, sample_rate, master_final)
            st.download_button("💾 Download Master (WAV)", buf.getvalue(), "musical_bus.wav")

        with col2:
            st.header("📊 Harmonic Timeline")
            # 시각화 로직 (동일)
            for idx in range(max_tracks):
                if tracks_visual[idx]:
                    times = [v['time'] for v in tracks_visual[idx]]
                    freqs = [v['freq'] for v in tracks_visual[idx]]
                    fig = go.Figure(go.Scatter(x=times, y=freqs, mode='lines', line=dict(color='#00d1ff')))
                    fig.update_layout(height=100, margin=dict(l=0,r=0,t=10,b=10), xaxis_title="Time(s)", yaxis_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
