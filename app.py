import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Optical Layer DAW")
st.title("🎛 Layer-Specific Optical DAW: Contrast Mode")

# --- 레이어별 고유 사운드 엔진 ---
def generate_layer_sound(t, freq, area, intensity, layer_idx, sample_rate):
    """
    Layer 1 (가장 큰 빛): 웅장한 울림 (Ambient Pad) - 지속성 길고 부드러움
    Layer 2 (두 번째): 딱딱 끊기는 비트 (Percussive) - 매우 짧고 타격감 있음
    Layer 3 (세 번째): 일렉트로 리드 (Acid Lead) - 날카롭고 변조가 심함
    Layer 4 (네 번째): 하이파이 신스 (Chirp) - 매우 높고 톡톡 튀는 소리
    """
    if layer_idx == 0:  # 🌊 Layer 1: 웅장한 울림
        wave = np.sin(2 * np.pi * freq * t)
        # 매우 긴 페이드 아웃
        env = np.linspace(1, 0.3, len(t))
        return (wave * env).astype(np.float32)

    elif layer_idx == 1:  # 🥁 Layer 2: 딱딱 끊기는 비트
        # 사각파를 사용하여 타격감 부여
        wave = np.sign(np.sin(2 * np.pi * freq * t))
        # 아주 짧은 엔벨로프 (Pluck 소리)
        env = np.exp(-np.linspace(0, 10, len(t))) 
        return (wave * env * 0.6).astype(np.float32)

    elif layer_idx == 2:  # 🎸 Layer 3: 날카로운 리드
        # 톱니파 + 필터 변조
        wave = 2 * (t * freq - np.floor(0.5 + t * freq))
        env = np.ones(len(t))
        env[-int(len(t)*0.5):] = np.linspace(1, 0, int(len(t)*0.5))
        return (wave * env * 0.5).astype(np.float32)

    else:  # ✨ Layer 4: 고음 Chirp
        wave = np.sin(2 * np.pi * freq * 2 * t) # 주파수 2배
        # 0.05초만 소리 나고 끊김
        env = np.zeros(len(t))
        env[:int(len(t)*0.3)] = 1
        return (wave * env * 0.4).astype(np.float32)

with st.sidebar:
    st.header("🎛 Layer Mixer")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.divider()
    # 개별 레이어 활성화/비활성화
    active_layers = st.multiselect(
        "🔊 플레이할 레이어 선택",
        ["Layer 1 (웅장한 울림)", "Layer 2 (딱딱한 비트)", "Layer 3 (날카로운 리드)", "Layer 4 (고음 Chirp)"],
        default=["Layer 1 (웅장한 울림)", "Layer 2 (딱딱한 비트)", "Layer 3 (날카로운 리드)", "Layer 4 (고음 Chirp)"]
    )
    intensity_threshold = st.slider("빛 감지 문턱값", 50, 255, 200)
    master_gain = st.slider("Master Gain", 0.1, 3.0, 1.5)

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
        vis_pitch = [[] for _ in range(4)]
        
        prog = st.progress(0)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, intensity_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_idx = int(i * (sample_rate / fps))
            t = np.linspace(0, 1/fps, int(sample_rate/fps), False).astype(np.float32)
            
            # 면적 순으로 정렬하여 각 레이어에 배분
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
            
            for idx, cnt in enumerate(sorted_cnts):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx = int(M["m10"]/M["m00"])
                
                # 빛의 특성에 따른 주파수 (면적 -> 저음, 소형 -> 고음)
                freq = 80 + (idx * 150) + (1000 / (np.sqrt(area) + 1))
                
                # 레이어별 특화 사운드 생성
                tone = generate_layer_sound(t, freq, area, 255, idx, sample_rate)
                
                pan_r = cx / frame.shape[1]
                pan_l = 1.0 - pan_r
                
                end_idx = start_idx + len(tone)
                if end_idx < len(tracks_l[0]):
                    tracks_l[idx][start_idx:end_idx] += tone * pan_l * master_gain
                    tracks_r[idx][start_idx:end_idx] += tone * pan_r * master_gain
                vis_pitch[idx].append(freq)

            for j in range(len(sorted_cnts), 4): vis_pitch[j].append(None)
            if i % 30 == 0: prog.progress(i / total_frames)

        # --- [실시간 믹싱] ---
        master_l = np.zeros_like(tracks_l[0])
        master_r = np.zeros_like(tracks_r[0])
        for idx, name in enumerate(["Layer 1 (웅장한 울림)", "Layer 2 (딱딱한 비트)", "Layer 3 (날카로운 리드)", "Layer 4 (고음 Chirp)"]):
            if name in active_layers:
                master_l += tracks_l[idx]
                master_r += tracks_r[idx]

        master_stereo = np.vstack((master_l, master_r)).T
        peak = np.max(np.abs(master_stereo))
        if peak > 0: master_stereo = (master_stereo / peak) * 0.9
        audio_int16 = (master_stereo * 32767).astype(np.int16)

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(audio_int16.tobytes())

        # UI 출력
        col_main, col_sub = st.columns([1.5, 1])
        with col_main:
            st.header("🎞 Sync Performance")
            st.video(uploaded_file)
            st.audio(wav_buf.getvalue())
            st.download_button("💾 전체 믹스 다운로드", wav_buf.getvalue(), "layer_contrast_mix.wav")

        with col_sub:
            st.header("📊 MIDI-Style Timeline")
            time_axis = np.linspace(0, duration, total_frames)
            fig = go.Figure()
            colors = ['#00d1ff', '#ff4b4b', '#7752fe', '#00ff88']
            for i in range(4):
                if any(f"Layer {i+1}" in n for n in active_layers):
                    fig.add_trace(go.Scatter(x=time_axis, y=vis_pitch[i], name=f"L{i+1}", line=dict(color=colors[i])))
            fig.update_layout(template="plotly_dark", height=400, xaxis=dict(rangeslider=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"오류: {e}")
