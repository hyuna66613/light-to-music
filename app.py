import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Refined Optical DAW")
st.title("🎹 Refined Sound: Optical Electronic DAW")

# --- 정밀 사운드 엔진 ---
def apply_soft_envelope(tone, layer_idx):
    n = len(tone)
    t_env = np.linspace(0, 1, n)
    
    if layer_idx == 0:  # Deep Bass: 아주 부드럽고 묵직하게
        env = np.sin(t_env * np.pi * 0.5 + np.pi * 0.5) # 서서히 줄어드는 감쇠
        return tone * env * 1.2
    elif layer_idx == 1 or layer_idx == 2:  # Warm Pluck/Lead: 고음의 날카로움을 깎음
        # 소리가 툭 치고 부드럽게 사라짐
        env = np.exp(-t_env * 5)
        return tone * env * 0.7
    else:  # Track 4: Snap/Chirp (기존의 좋은 느낌 유지)
        env = np.exp(-t_env * 25) # 아주 짧은 타격감
        return tone * env * 0.9

def generate_tuned_wave(freq, duration, layer_idx):
    t = np.linspace(0, duration, int(22050 * duration), False)
    
    if layer_idx == 0: # Bass: 순수한 저음
        wave_data = np.sin(2 * np.pi * freq * t)
    elif layer_idx == 1: # Track 2: 부드러운 오르간 느낌
        wave_data = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    elif layer_idx == 2: # Track 3: 따뜻한 패드 느낌
        # 배음을 섞되 위상을 조절해 날카로움을 중화
        wave_data = np.sin(2 * np.pi * freq * t) * 0.7 + np.sin(2 * np.pi * (freq + 2) * t) * 0.3
    else: # Track 4: 기존의 핑거스냅/클릭 질감
        wave_data = np.sign(np.sin(2 * np.pi * freq * t)) * (np.random.rand(len(t)) * 0.1 + 0.9)
        
    return apply_soft_envelope(wave_data, layer_idx)

# --- 메인 로직 ---
uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    
    BPM = 120
    UNIT_SEC = (60/BPM) / 2 # 8분 음표 기준
    num_units = int(video_len / UNIT_SEC)
    
    tracks_l = [np.zeros(int(22050 * video_len) + 1000) for _ in range(4)]
    tracks_r = [np.zeros(int(22050 * video_len) + 1000) for _ in range(4)]
    vis_data = [[] for _ in range(4)]

    prog = st.progress(0)
    for u in range(num_units):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(u * UNIT_SEC * fps))
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
        
        start_s = int(u * UNIT_SEC * 22050)
        for idx, cnt in enumerate(sorted_cnts):
            area = cv2.contourArea(cnt)
            cx = int(cv2.moments(cnt)["m10"]/cv2.moments(cnt)["m00"]) if cv2.moments(cnt)["m00"] != 0 else 0
            
            freq = [65.4, 130.8, 261.6, 880.0][idx] + (area % 20)
            tone = generate_tuned_wave(freq, UNIT_SEC, idx)
            
            pan_r = np.clip(cx / frame.shape[1], 0.1, 0.9)
            pan_l = 1.0 - pan_r
            
            end_s = start_s + len(tone)
            if end_s < len(tracks_l[idx]):
                tracks_l[idx][start_s:end_s] += tone * pan_l
                tracks_r[idx][start_s:end_s] += tone * pan_r
            vis_data[idx].append(freq)
        for j in range(len(sorted_cnts), 4): vis_data[j].append(None)
        if u % 10 == 0: prog.progress(u / num_units)
    cap.release()

    # --- UI ---
    st.header("🎞 Master Mix & Analysis")
    col_v, col_g = st.columns(2)
    with col_v:
        st.video(temp_path)
        m_l = np.clip(np.sum(tracks_l, axis=0), -1, 1)
        m_r = np.clip(np.sum(tracks_r, axis=0), -1, 1)
        master = np.vstack((m_l, m_r)).T
        m_io = io.BytesIO()
        with wave.open(m_io, 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(22050); wf.writeframes((master * 32767).astype(np.int16).tobytes())
        st.audio(m_io.getvalue())
        st.download_button("💾 전체 Mix 저장", m_io.getvalue(), "final_mix.wav")

    with col_g:
        fig = go.Figure()
        t_axis = np.linspace(0, video_len, len(vis_data[0]))
        for i, color in enumerate(['#00E5FF', '#FF3D00', '#D500F9', '#FFEA00']):
            fig.add_trace(go.Scatter(x=t_axis, y=vis_data[i], name=f"Layer {i+1}", line=dict(color=color)))
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📁 Layer Stems (Individual Play & Save)")
    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            l_data = np.vstack((tracks_l[i], tracks_r[i])).T
            p = np.max(np.abs(l_data))
            if p > 0: l_data = (l_data / p) * 0.7
            l_io = io.BytesIO()
            with wave.open(l_io, 'wb') as wf:
                wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(22050); wf.writeframes((l_data * 32767).astype(np.int16).tobytes())
            st.write(f"Track {i+1}")
            st.audio(l_io.getvalue())
            st.download_button(f"📥 Layer {i+1} 저장", l_io.getvalue(), f"track_{i+1}.wav")
