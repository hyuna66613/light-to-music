import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Professional Optical DAW")
st.title("🎹 Studio Grade: Optical Electronic DAW")

# --- 고급 사운드 합성 엔진 ---
def generate_pro_sound(t, freq, layer_idx, sample_rate):
    """
    Layer 0: Deep Sub Bass - 묵직한 저음 (Sine + Harmonic)
    Layer 1: Warm Pluck - 따뜻하게 끊기는 리듬 (Filtered Square)
    Layer 2: Dreamy Lead - 부드러운 멜로디 (Filtered Saw)
    Layer 3: Top Chirp - 섬세한 고음 질감 (Pure Sine High)
    """
    if layer_idx == 0:  # 🎸 Deep Sub Bass
        # 주파수를 낮추고(Base 40-80Hz), 배음을 섞어 묵직하게
        base_freq = freq * 0.5 
        wave = np.sin(2 * np.pi * base_freq * t) + 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
        env = np.ones(len(t)) # 베이스는 지속성 있게
        return (wave * env * 0.8).astype(np.float32)

    elif layer_idx == 1:  # 🎹 Warm Pluck
        # 사각파를 쓰되 고음의 날카로움을 억제하기 위해 사인파와 혼합
        wave = 0.7 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sign(np.sin(2 * np.pi * freq * t))
        # 지수적 감쇠 (Pluck)
        env = np.exp(-np.linspace(0, 8, len(t))) 
        return (wave * env * 0.6).astype(np.float32)

    elif layer_idx == 2:  # 🎤 Dreamy Lead
        # 톱니파를 쓰되, 고역대를 부드럽게 처리
        wave = 0.5 * (2 * (t * freq - np.floor(0.5 + t * freq))) + 0.5 * np.sin(2 * np.pi * freq * t)
        # 소리가 서서히 커졌다가 작아짐 (Soft Attack)
        env = np.sin(np.linspace(0, np.pi, len(t))) 
        return (wave * env * 0.4).astype(np.float32)

    else:  # ✨ Top Chirp
        # 매우 높은 주파수에서 찰나의 소리
        wave = np.sin(2 * np.pi * freq * 3 * t)
        env = np.zeros(len(t))
        env[:int(len(t)*0.2)] = np.linspace(1, 0, int(len(t)*0.2))
        return (wave * env * 0.3).astype(np.float32)

with st.sidebar:
    st.header("🎛 Studio Mixer")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.divider()
    active_layers = st.multiselect(
        "🔊 레이어 활성화",
        ["Layer 1 (Sub Bass)", "Layer 2 (Warm Pluck)", "Layer 3 (Soft Lead)", "Layer 4 (High Texture)"],
        default=["Layer 1 (Sub Bass)", "Layer 2 (Warm Pluck)", "Layer 3 (Soft Lead)", "Layer 4 (High Texture)"]
    )
    intensity_val = st.slider("광원 감도", 30, 255, 180)
    master_gain = st.slider("Master Output", 0.5, 5.0, 2.0)

if uploaded_file:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = 22050 
        duration = total_frames / fps
        
        tracks_l = [np.zeros(int(sample_rate * duration) + sample_rate) for _ in range(4)]
        tracks_r = [np.zeros(int(sample_rate * duration) + sample_rate) for _ in range(4)]
        
        prog = st.progress(0)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, intensity_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_idx = int(i * (sample_rate / fps))
            t = np.linspace(0, 1/fps, int(sample_rate/fps), False).astype(np.float32)
            
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
            
            for idx, cnt in enumerate(sorted_cnts):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx = int(M["m10"]/M["m00"])
                
                # 주파수 매핑 최적화 (베이스 레이어는 낮게, 리드는 높게)
                base_f = [60, 150, 400, 1200][idx]
                freq = base_f + (area % 200)
                
                tone = generate_pro_sound(t, freq, idx, sample_rate)
                
                # 스테레오 팬닝 최적화
                pan_r = np.clip(cx / frame.shape[1], 0.1, 0.9)
                pan_l = 1.0 - pan_r
                
                end_idx = start_idx + len(tone)
                if end_idx < len(tracks_l[0]):
                    tracks_l[idx][start_idx:end_idx] += tone * pan_l * master_gain
                    tracks_r[idx][start_idx:end_idx] += tone * pan_r * master_gain

            if i % 30 == 0: prog.progress(i / total_frames)

        # 믹싱
        final_l, final_r = np.zeros_like(tracks_l[0]), np.zeros_like(tracks_r[0])
        for idx, name in enumerate(["Layer 1 (Sub Bass)", "Layer 2 (Warm Pluck)", "Layer 3 (Soft Lead)", "Layer 4 (High Texture)"]):
            if name in active_layers:
                final_l += tracks_l[idx]
                final_r += tracks_r[idx]

        # 마스터링 (Soft Clipping)
        master_stereo = np.vstack((final_l, final_r)).T
        peak = np.max(np.abs(master_stereo))
        if peak > 0: master_stereo = (master_stereo / peak) * 0.85
        audio_int16 = (master_stereo * 32767).astype(np.int16)

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(audio_int16.tobytes())

        # UI
        st.header("🎧 Master Mix Playback")
        st.video(uploaded_file)
        st.audio(wav_buf.getvalue())
        st.download_button("💾 Studio Mix 다운로드", wav_buf.getvalue(), "studio_mix.wav")

    except Exception as e:
        st.error(f"오류: {e}")
