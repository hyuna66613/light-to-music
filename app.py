import streamlit as st
import cv2
import numpy as np
from scipy.io import wavfile
import io
import plotly.graph_objects as go

# 설정
st.set_page_config(layout="wide", page_title="GarageLight DAW")
st.title("🎼 GarageLight: Optical Synth DAW (Final Fixed)")

# 음계 설정 (밤의 펜타토닉)
NOTES = [130.81, 155.56, 174.61, 196.00, 233.08, 261.63, 311.13, 349.23, 392.00, 466.16, 523.25, 622.25, 698.46, 783.99, 932.33]

def get_nearest_note(freq):
    return NOTES[np.abs(np.array(NOTES) - freq).argmin()]

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
    st.header("🎛 Control Panel")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    vol_boost = st.slider("볼륨 증폭도", 0.1, 2.0, 1.0)

if uploaded_file:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = 22050 
        
        # 오디오 캔버스 (float32로 시작)
        audio_len = int(sample_rate * (total_frames / fps)) + sample_rate
        master_l = np.zeros(audio_len, dtype=np.float32)
        master_r = np.zeros(audio_len, dtype=np.float32)
        
        tracks_visual = [[] for _ in range(6)]
        prog = st.progress(0)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_idx = int(i * (sample_rate / fps))
            t = np.linspace(0, 1/fps, int(sample_rate/fps), False).astype(np.float32)
            
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
            for idx, cnt in enumerate(sorted_cnts):
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                
                note_f = get_nearest_note(150 + (frame.shape[0]-cy)*1.5)
                vol = min(cv2.contourArea(cnt)/1500, 0.5) * vol_boost
                
                tone = apply_envelope(vol * np.sin(2 * np.pi * note_f * t), sample_rate)
                
                pan_r = cx / frame.shape[1]
                pan_l = 1.0 - pan_r
                
                end_idx = start_idx + len(tone)
                if end_idx < audio_len:
                    master_l[start_idx:end_idx] += tone * pan_l
                    master_r[start_idx:end_idx] += tone * pan_r
                
                tracks_visual[idx].append({'t': i/fps, 'f': note_f})
            
            if i % 30 == 0: prog.progress(min(i / total_frames, 1.0))

        # --- [에러 해결의 핵심 로직] ---
        master_stereo = np.vstack((master_l, master_r)).T
        
        # 1. 최고점 찾기
        max_val = np.max(np.abs(master_stereo))
        if max_val > 0:
            # 2. 모든 데이터를 -1.0 ~ 1.0 범위로 강제 고정
            master_stereo = (master_stereo / max_peak) if 'max_peak' in locals() else master_stereo / max_val
            
        # 3. float32 데이터를 16비트 정수로 직접 변환 (가장 안전한 방식)
        # 32767을 곱하고 정수로 반올림하여 'H' format 에러를 원천 차단
        audio_final = np.clip(master_stereo * 32767, -32768, 32767).astype(np.int16)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.header("🎞 Playback")
            st.video(uploaded_file)
            
            # 오디오 출력
            st.audio(audio_final, sample_rate=sample_rate)
            
            # 다운로드 버튼
            buf = io.BytesIO()
            wavfile.write(buf, sample_rate, audio_final)
            st.download_button("💾 Download WAV", buf.getvalue(), "bus_music.wav")

        with col2:
            st.header("📊 Timeline")
            for idx in range(6):
                if tracks_visual[idx]:
                    v_t = [v['t'] for v in tracks_visual[idx]]
                    v_f = [v['f'] for v in tracks_visual[idx]]
                    fig = go.Figure(go.Scatter(x=v_t, y=v_f, mode='lines', line_color='#00d1ff'))
                    fig.update_layout(height=100, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"오류 발생: {e}")
