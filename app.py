import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="120BPM Optical DAW")
st.title("🎧 120 BPM Sync: Professional Optical DAW")

# --- 음악적 설정 ---
BPM = 120
BEAT_DURATION = 60 / BPM  # 1박자 길이 (0.5초)
SUBDIVISION = 4           # 1박자를 4개로 쪼개기 (16분음표 단위 분석)
SAMPLE_RATE = 22050

def apply_eq_and_envelope(tone, layer_idx):
    """레이어별 특화 EQ 및 엔벨로프 적용"""
    n = len(tone)
    t_env = np.linspace(0, 1, n)
    
    if layer_idx == 0:  # 🎸 Deep Bass: 저음역대 강조, 고음 커트 (Low Pass)
        env = np.ones(n)
        # 끝부분만 살짝 페이드아웃하여 웅장함 유지
        env[-int(n*0.2):] = np.linspace(1, 0, int(n*0.2))
        return tone * env * 0.9
    
    elif layer_idx == 1:  # 🎹 Warm Pluck: 중음역대 강조, 짧은 타격감
        env = np.exp(-t_env * 10)  # 아주 빠르게 사라지는 소리
        return tone * env * 0.7
    
    elif layer_idx == 2:  # 🎤 Airy Lead: 중고음역대, 부드러운 연결
        env = np.sin(t_env * np.pi) # 부드럽게 시작해서 부드럽게 끝남
        return tone * env * 0.5
    
    else:  # ✨ Shimmer Bell: 고음역대 전용, 잔향 강조
        env = np.exp(-t_env * 5)
        return tone * env * 0.4

def generate_musical_wave(freq, duration, layer_idx):
    """레이어 특성에 맞는 파형 생성"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    if layer_idx == 0: # Bass: Sine + 1옥타브 위 배음 살짝
        wave = np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
    elif layer_idx == 1: # Pluck: Square(Filtered 느낌) + Sine
        wave = 0.5 * np.sign(np.sin(2 * np.pi * freq * t)) + 0.5 * np.sin(2 * np.pi * freq * t)
    elif layer_idx == 2: # Lead: Sawtooth(부드럽게)
        wave = 2 * (t * freq - np.floor(0.5 + t * freq))
    else: # Bell: Sine FM 합성 느낌
        wave = np.sin(2 * np.pi * freq * t + 0.5 * np.sin(2 * np.pi * freq * 2.01 * t))
        
    return apply_eq_and_envelope(wave, layer_idx)

with st.sidebar:
    st.header("🎛 Global Settings")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.divider()
    sensitivity = st.slider("광원 감도", 50, 255, 180)
    master_gain = st.slider("Master Output", 0.5, 5.0, 1.8)
    st.info(f"현재 템포: {BPM} BPM (16분음표 단위 분석)")

if uploaded_file:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        # 분석 단위 설정 (16분음표 길이만큼 프레임을 묶어서 분석)
        unit_duration = BEAT_DURATION / SUBDIVISION
        frames_per_unit = int(fps * unit_duration)
        num_units = int(video_duration / unit_duration)
        
        tracks_l = [np.zeros(int(SAMPLE_RATE * video_duration) + SAMPLE_RATE) for _ in range(4)]
        tracks_r = [np.zeros(int(SAMPLE_RATE * video_duration) + SAMPLE_RATE) for _ in range(4)]
        vis_pitches = [[] for _ in range(4)]
        
        prog = st.progress(0)
        
        for u in range(num_units):
            # 해당 박자 단위의 중간 프레임 추출 (분절 감소를 위해 대표값 사용)
            frame_idx = u * frames_per_unit
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, sensitivity, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
            start_sample = int(u * unit_duration * SAMPLE_RATE)
            
            for idx, cnt in enumerate(sorted_cnts):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx = int(M["m10"]/M["m00"])
                
                # 주파수를 음악적 스케일에 맞춰 조정 (C major scale 느낌)
                scale = [65.41, 130.81, 261.63, 523.25] # C2, C3, C4, C5
                freq = scale[idx] + (area % 50)
                
                # 음악적 단위(16분음표) 길이의 소리 생성
                tone = generate_musical_wave(freq, unit_duration, idx)
                
                pan_r = np.clip(cx / frame.shape[1], 0.1, 0.9)
                pan_l = 1.0 - pan_r
                
                end_sample = start_sample + len(tone)
                if end_sample < len(tracks_l[0]):
                    tracks_l[idx][start_sample:end_sample] += tone * pan_l * master_gain
                    tracks_r[idx][start_sample:end_sample] += tone * pan_r * master_gain
                vis_pitches[idx].append(freq)

            for j in range(len(sorted_cnts), 4): vis_pitches[j].append(None)
            if u % 10 == 0: prog.progress(min(u / num_units, 1.0))

        # 믹싱 및 노멀라이징
        final_l = np.sum(tracks_l, axis=0)
        final_r = np.sum(tracks_r, axis=0)
        master_stereo = np.vstack((final_l, final_r)).T
        peak = np.max(np.abs(master_stereo))
        if peak > 0: master_stereo = (master_stereo / peak) * 0.85
        audio_int16 = (master_stereo * 32767).astype(np.int16)

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE); wf.writeframes(audio_int16.tobytes())

        # --- UI 레이아웃 ---
        st.header("🎵 120 BPM Quantized Mix")
        col_v, col_g = st.columns([1, 1])
        
        with col_v:
            st.video(uploaded_file)
            st.audio(wav_buf.getvalue())
            st.download_button("💾 전체 믹스 저장", wav_buf.getvalue(), "quantized_mix.wav")

        with col_g:
            time_axis = np.linspace(0, video_duration, len(vis_pitches[0]))
            fig = go.Figure()
            colors = ['#00E5FF', '#FF3D00', '#D500F9', '#FFEA00']
            for i in range(4):
                fig.add_trace(go.Scatter(x=time_axis, y=vis_pitches[i], name=f"Layer {i+1}", line=dict(color=colors[i], width=2)))
            fig.update_layout(template="plotly_dark", height=420, xaxis=dict(rangeslider=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("📁 개별 레이어 스테레오 파일 (EQ 적용됨)")
        cols = st.columns(4)
        layer_names = ["Deep Bass", "Warm Pluck", "Airy Lead", "Shimmer Bell"]
        for i in range(4):
            with cols[i]:
                t_buf = io.BytesIO()
                t_data = np.vstack((tracks_l[i], tracks_r[i])).T
                t_peak = np.max(np.abs(t_data))
                if t_peak > 0: t_data = (t_data / t_peak) * 0.8
                with wave.open(t_buf, 'wb') as wf:
                    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE); wf.writeframes((t_data * 32767).astype(np.int16).tobytes())
                st.write(f"**{layer_names[i]}**")
                st.audio(t_buf.getvalue())
                st.download_button(f"📥 {i+1}번 저장", t_buf.getvalue(), f"track_{i+1}.wav")

    except Exception as e:
        st.error(f"오류: {e}")
