import streamlit as st
import cv2
import numpy as np
from scipy.io import wavfile
import io
import plotly.graph_objects as go

# --- 설정 ---
st.set_page_config(layout="wide", page_title="Musical Light DAW")
st.title("🎼 GarageLight: Optical Synth DAW (Bug Fixed)")

# 음계 설정
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
    env = np.ones(n, dtype=np.float32)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return tone * env

with st.sidebar:
    st.header("🎛 컨트롤 패널")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.divider()
    vol_boost = st.slider("볼륨 증폭도 (Gain)", 0.1, 2.0, 1.0, 0.1)

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
        
        # 오디오 데이터를 실수형(float32)으로 관리 (에러 방지 핵심)
        audio_len = int(sample_rate * (total_frames / fps)) + (sample_rate * 2)
        master_l = np.zeros(audio_len, dtype=np.float32)
        master_r = np.zeros(audio_len, dtype=np.float32)
        
        tracks_visual = [[] for _ in range(max_tracks)]
        
        prog_bar = st.progress(0)
        status = st.empty()

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_idx = int(i * (sample_rate / fps))
            duration = 1.0 / fps
            t = np.linspace(0, duration, int(sample_rate * duration), False).astype(np.float32)
            
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_tracks]
            
            for idx, cnt in enumerate(sorted_contours):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                note_freq = get_nearest_note(150 + ((frame.shape[0] - cy) * 1.5))
                vol = (min(area / 1500, 0.5) * vol_boost)
                
                # 사운드 합성
                tone = (vol * np.sin(2 * np.pi * note_freq * t)).astype(np.float32)
                tone = apply_envelope(tone, sample_rate)
                
                pan_r = cx / frame.shape[1]
                pan_l = 1.0 - pan_r
                
                end_idx = start_idx + len(tone)
                if end_idx < audio_len:
                    master_l[start_idx:end_idx] += (tone * pan_l)
                    master_r[start_idx:end_idx] += (tone * pan_r)
                
                tracks_visual[idx].append({'time': i/fps, 'freq': note_freq})
            
            if i % 30 == 0:
                prog_bar.progress(min(i / total_frames, 1.0))

        # --- [에러 해결의 핵심: 정수 변환을 하지 않고 float32로 직접 저장] ---
        master_stereo = np.vstack((master_l, master_r)).T
        
        # 볼륨 평준화 (Peak Normalization)
        max_peak = np.max(np.abs(master_stereo))
        if max_peak > 0:
            master_final = (master_stereo / max_peak) * 0.8 # 리미터 적용
        else:
            master_final = master_stereo

        # UI 출력
        col1, col2 = st.columns([1, 1])
        with col1:
            st.header("🎞 View & Play")
            st.video(uploaded_file)
            
            # float32 형식으로 오디오 재생 (가장 안전함)
            st.audio(master_final, sample_rate=sample_rate, format="audio/wav")
            
            # 다운로드 버튼용 바이너리 생성
            buf = io.BytesIO()
            # float32 데이터를 그대로 저장하여 'H' format 에러 원천 차단
            wavfile.write(buf, sample_rate, master_final.astype(np.float32))
            st.download_button("💾 Download Master (WAV)", buf.getvalue(), "musical_bus.wav")

        with col2:
            st.header("📊 Harmonic Timeline")
            for idx in range(max_tracks):
                if tracks_visual[idx]:
                    v_times = [v['time'] for v in tracks_visual[idx]]
                    v_freqs = [v['freq'] for v in tracks_visual[idx]]
                    fig = go.Figure(go.Scatter(x=v_times, y=v_freqs, mode='lines', line=dict(color='#00d1ff', width=1)))
                    fig.update_layout(height=100, margin=dict(l=0,r=0,t=10,b=10), xaxis_visible=False, yaxis_title="Hz", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
