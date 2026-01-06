import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="GarageLight DAW v2")
st.title("🎹 GarageLight: Optical DAW (Pro Mode)")

# --- 사운드 디자인: 몽환적인 신시사이저 ---
NOTES = [130.81, 146.83, 164.81, 196.00, 220.00, 261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25, 783.99, 880.00]

def get_nearest_note(freq):
    return NOTES[np.abs(np.array(NOTES) - freq).argmin()]

def apply_synth_effects(tone, sample_rate):
    # 1. ADSR Envelope
    n = len(tone)
    env = np.ones(n, dtype=np.float32)
    attack = int(n * 0.2)
    release = int(n * 0.4)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    tone = tone * env
    
    # 2. 미세한 에코(Echo) 추가 (음악적 재미)
    delay_samples = int(sample_rate * 0.15)
    if n > delay_samples:
        echo = np.zeros_like(tone)
        echo[delay_samples:] = tone[:-delay_samples] * 0.3
        tone = tone + echo
    return tone

# --- 사이드바: 믹서 컨트롤 ---
with st.sidebar:
    st.header("🎛 Mixer & Tracks")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.divider()
    selected_tracks = st.multiselect(
        "플레이/저장할 트랙 선택", 
        [f"Track {i+1}" for i in range(8)], 
        default=[f"Track {i+1}" for i in range(8)]
    )
    st.info("선택한 트랙들만 합쳐져서 플레이어와 다운로드에 반영됩니다.")

if uploaded_file:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = 22050 
        
        # 8개 독립 트랙 데이터 보관
        tracks_data_l = [np.zeros(int(sample_rate * (total_frames / fps)) + sample_rate) for _ in range(8)]
        tracks_data_r = [np.zeros(int(sample_rate * (total_frames / fps)) + sample_rate) for _ in range(8)]
        visual_data = [[] for _ in range(8)]
        
        prog_bar = st.progress(0)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_idx = int(i * (sample_rate / fps))
            t = np.linspace(0, 1/fps, int(sample_rate/fps), False).astype(np.float32)
            
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
            for idx, cnt in enumerate(sorted_cnts):
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                
                note_f = get_nearest_note(150 + (frame.shape[0]-cy)*1.8)
                vol = min(cv2.contourArea(cnt)/1200, 0.5)
                
                # 사운드 생성
                tone = vol * (np.sin(2 * np.pi * note_f * t) + 0.2 * np.sin(2 * np.pi * note_f * 1.5 * t))
                tone = apply_synth_effects(tone, sample_rate)
                
                pan_r = cx / frame.shape[1]
                pan_l = 1.0 - pan_r
                
                end_idx = start_idx + len(tone)
                if end_idx < len(tracks_data_l[0]):
                    tracks_data_l[idx][start_idx:end_idx] += tone * pan_l
                    tracks_data_r[idx][start_idx:end_idx] += tone * pan_r
                
                visual_data[idx].append(note_f)
            
            # 빈 데이터 채우기 (동기화용)
            for j in range(len(sorted_cnts), 8):
                visual_data[j].append(None)
            
            if i % 30 == 0: prog_bar.progress(i / total_frames)

        # --- 믹싱 프로세스 (선택된 트랙만) ---
        master_l = np.zeros_like(tracks_data_l[0])
        master_r = np.zeros_like(tracks_data_r[0])
        
        for t_name in selected_tracks:
            t_idx = int(t_name.split()[-1]) - 1
            master_l += tracks_data_l[t_idx]
            master_r += tracks_data_r[t_idx]
            
        master_stereo = np.vstack((master_l, master_r)).T
        if np.max(np.abs(master_stereo)) > 0:
            master_stereo = (master_stereo / np.max(np.abs(master_stereo))) * 0.8
        
        audio_int16 = np.clip(master_stereo * 32767, -32768, 32767).astype(np.int16)
        
        # WAV 바이너리 생성
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        # --- UI: 비디오 & 플레이어 ---
        col_play, col_graph = st.columns([1, 1])
        with col_play:
            st.header("📽 Master Player")
            st.video(uploaded_file)
            st.audio(wav_buf.getvalue())
            st.download_button("💾 선택한 조합 다운로드", wav_buf.getvalue(), "my_remix.wav")

        with col_graph:
            st.header("📊 Interactive Timeline")
            time_axis = np.linspace(0, total_frames/fps, total_frames)
            
            fig = go.Figure()
            for i, t_name in enumerate(selected_tracks):
                t_idx = int(t_name.split()[-1]) - 1
                fig.add_trace(go.Scatter(
                    x=time_axis, y=visual_data[t_idx], 
                    name=t_name, mode='lines', line=dict(width=1.5)
                ))
            
            # 진행 표시선(Current Time Marker) 설정 가능하게 구성
            fig.update_layout(
                template="plotly_dark", height=400,
                xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            # Plotly의 모드바에 'spikeline' 활성화하여 현재 마우스 위치 동기화 보조
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

    except Exception as e:
        st.error(f"연산 오류: {e}")
