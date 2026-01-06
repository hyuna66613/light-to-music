import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Luxury Optical Synth")
st.title("🎹 Luxury Electronica: Optical Sound Design")

# --- 고급 사운드 및 리버브 엔진 ---
def apply_reverb_style(tone, sample_rate, decay=0.4):
    """소리를 감싸주는 듯한 잔향(Reverb) 효과 시뮬레이션"""
    n = len(tone)
    env = np.ones(n)
    # 끝부분을 아주 부드럽게 감쇠시켜 공간감 형성
    env = np.exp(-np.linspace(0, 1/decay, n))
    return tone * env

def generate_luxury_sound(t, freq, layer_idx, sample_rate):
    """
    Layer 0: Deep & Warm Bass (감싸주는 저음)
    Layer 1: Analog Pluck (따뜻한 리듬)
    Layer 2: Soft Poly Lead (부드러운 멜로디)
    Layer 3: Crystal Bell (고급스러운 고음 울림)
    """
    if layer_idx == 0:  # 🌊 Deep & Warm Bass
        # 저음 주파수 고정 및 사인파 위주로 구성 (감싸주는 느낌)
        base_f = 50 + (freq % 40)
        wave_data = np.sin(2 * np.pi * base_f * t) + 0.2 * np.sin(2 * np.pi * base_f * 2 * t)
        return apply_reverb_style(wave_data, sample_rate, decay=2.0)

    elif layer_idx == 1:  # 🎹 Warm Pluck
        wave_data = 0.8 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sign(np.sin(2 * np.pi * freq * t))
        return apply_reverb_style(wave_data, sample_rate, decay=0.2)

    elif layer_idx == 2:  # 🎤 Soft Lead
        wave_data = np.sin(2 * np.pi * freq * t) * (1 + 0.2 * np.sin(2 * np.pi * 5 * t)) # 비브라토 추가
        return apply_reverb_style(wave_data, sample_rate, decay=0.8)

    else:  # 🔔 Crystal Bell (고급스러운 고음)
        # 날카로운 소리를 없애기 위해 여러 사인파를 중첩 (FM 합성 느낌)
        wave_data = (np.sin(2 * np.pi * freq * t) * 0.6 + 
                     np.sin(2 * np.pi * freq * 2.01 * t) * 0.3 + 
                     np.sin(2 * np.pi * freq * 3.02 * t) * 0.1)
        # 짧지만 끝이 부드러운 엔벨로프
        env = np.exp(-np.linspace(0, 12, len(t)))
        return (wave_data * env).astype(np.float32)

with st.sidebar:
    st.header("🎛 Studio Mixer")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.divider()
    active_layers = st.multiselect(
        "🔊 레이어 믹싱 선택",
        ["Layer 1 (Deep Bass)", "Layer 2 (Warm Pluck)", "Layer 3 (Soft Lead)", "Layer 4 (Crystal Bell)"],
        default=["Layer 1 (Deep Bass)", "Layer 2 (Warm Pluck)", "Layer 3 (Soft Lead)", "Layer 4 (Crystal Bell)"]
    )
    intensity_threshold = st.slider("광원 감도", 30, 255, 180)
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
        vis_data = [[] for _ in range(4)]
        
        prog = st.progress(0)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, intensity_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_idx = int(i * (sample_rate / fps))
            t = np.linspace(0, 1/fps, int(sample_rate/fps), False).astype(np.float32)
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
            
            for idx, cnt in enumerate(sorted_cnts):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx = int(M["m10"]/M["m00"])
                
                # 주파수 매핑
                base_f = [55, 220, 440, 880][idx]
                freq = base_f + (area % 100)
                
                tone = generate_luxury_sound(t, freq, idx, sample_rate)
                
                pan_r = np.clip(cx / frame.shape[1], 0.1, 0.9)
                pan_l = 1.0 - pan_r
                
                end_idx = start_idx + len(tone)
                if end_idx < len(tracks_l[0]):
                    tracks_l[idx][start_idx:end_idx] += tone * pan_l * master_gain
                    tracks_r[idx][start_idx:end_idx] += tone * pan_r * master_gain
                vis_data[idx].append(freq)

            for j in range(len(sorted_cnts), 4): vis_data[j].append(None)
            if i % 30 == 0: prog.progress(i / total_frames)

        # 믹싱 처리
        final_l, final_r = np.zeros_like(tracks_l[0]), np.zeros_like(tracks_r[0])
        for idx, name in enumerate(["Layer 1 (Deep Bass)", "Layer 2 (Warm Pluck)", "Layer 3 (Soft Lead)", "Layer 4 (Crystal Bell)"]):
            if name in active_layers:
                final_l += tracks_l[idx]
                final_r += tracks_r[idx]

        master_stereo = np.vstack((final_l, final_r)).T
        peak = np.max(np.abs(master_stereo))
        if peak > 0: master_stereo = (master_stereo / peak) * 0.85
        audio_int16 = (master_stereo * 32767).astype(np.int16)

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(audio_int16.tobytes())

        # --- UI 레이아웃 ---
        st.header("🎞 Sync Performance & Visualizer")
        col_vid, col_graph = st.columns([1, 1])
        
        with col_vid:
            st.video(uploaded_file)
            st.write("🔊 **Master Mix**")
            st.audio(wav_buf.getvalue())
            st.download_button("💾 전체 믹스 다운로드", wav_buf.getvalue(), "full_master_mix.wav")

        with col_graph:
            # 그래프 복구
            time_axis = np.linspace(0, duration, total_frames)
            fig = go.Figure()
            colors = ['#00d1ff', '#ff4b4b', '#7752fe', '#ffd700']
            for i in range(4):
                if any(f"Layer {i+1}" in n for n in active_layers):
                    fig.add_trace(go.Scatter(x=time_axis, y=vis_data[i], name=f"Layer {i+1}", line=dict(color=colors[i])))
            fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(rangeslider=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)

        # --- 부분 재생 및 레이어별 다운로드 기능 ---
        st.divider()
        st.subheader("📁 Individual Layer Tracks (Stem Export)")
        export_cols = st.columns(4)
        for i in range(4):
            with export_cols[i]:
                layer_name = ["Deep Bass", "Warm Pluck", "Soft Lead", "Crystal Bell"][i]
                t_buf = io.BytesIO()
                t_data = np.vstack((tracks_l[i], tracks_r[i])).T
                t_peak = np.max(np.abs(t_data))
                if t_peak > 0: t_data = (t_data / t_peak) * 0.8
                t_int16 = (t_data * 32767).astype(np.int16)
                with wave.open(t_buf, 'wb') as wf:
                    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(t_int16.tobytes())
                
                st.write(f"**Track {i+1}**")
                st.caption(layer_name)
                st.audio(t_buf.getvalue()) # 부분 재생 기능
                st.download_button(f"📥 {layer_name} 저장", t_buf.getvalue(), f"track_{i+1}_{layer_name}.wav")

    except Exception as e:
        st.error(f"오류: {e}")
