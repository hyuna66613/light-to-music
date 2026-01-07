import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

# --- 설정 ---
st.set_page_config(layout="wide", page_title="Professional Optical DAW")
st.title("🎹 Luxury Optical DAW: Individual Stem Control")

BPM = 120
SAMPLE_RATE = 22050
BEAT_SEC = 60 / BPM 
UNIT_SEC = BEAT_SEC / 2  # 8분 음표 단위 분석

def apply_luxury_envelope(tone, layer_idx, mood_v):
    """레이어별 고급 엔벨로프 및 EQ 적용"""
    n = len(tone)
    t_env = np.linspace(0, 1, n)
    
    if layer_idx == 0:  # Deep Bass: 묵직하고 긴 잔향
        env = np.ones(n)
        env[-int(n*0.3):] = np.linspace(1, 0, int(n*0.3))
        return tone * env * 1.1
    elif layer_idx == 3:  # Crystal Bell: 영롱하게 사라지는 소리
        env = np.exp(-t_env * (15 - mood_v * 5))
        return tone * env * 0.8
    else:  # Lead & Pluck: 부드러운 곡선형
        env = np.sin(t_env * np.pi)
        return tone * env * 0.6

def generate_pro_wave(freq, duration, layer_idx, mood_v):
    """영상 무드를 반영한 고품질 파형 생성"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    if layer_idx == 0: # Bass
        wave_data = np.sin(2 * np.pi * freq * t)
    elif layer_idx == 3: # Bell (FM 합성 스타일)
        wave_data = np.sin(2 * np.pi * freq * t + 0.5 * np.sin(2 * np.pi * freq * 2.01 * t))
    else: # Pluck & Lead
        wave_data = 0.6 * np.sin(2 * np.pi * freq * t) + 0.4 * np.sign(np.sin(2 * np.pi * freq * t))
        
    return apply_luxury_envelope(wave_data, layer_idx, mood_v)

# --- 파일 업로드 ---
uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # 영상 무드 분석 (밝기 기준)
    cap = cv2.VideoCapture(temp_path)
    ret, frame = cap.read()
    avg_v = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,2]) / 255 if ret else 0.5
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_len = total_frames / fps
    num_units = int(video_len / UNIT_SEC)
    
    tracks_l = [np.zeros(int(SAMPLE_RATE * video_len) + 500) for _ in range(4)]
    tracks_r = [np.zeros(int(SAMPLE_RATE * video_len) + 500) for _ in range(4)]
    vis_data = [[] for _ in range(4)]

    prog = st.progress(0)
    for u in range(num_units):
        target_frame = int(u * UNIT_SEC * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
        start_s = int(u * UNIT_SEC * SAMPLE_RATE)
        
        for idx, cnt in enumerate(sorted_cnts):
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"]/M["m00"])
            
            # 음악적 스케일 기반 주파수
            freq = [65.4, 130.8, 261.6, 523.2][idx] + (area % 30)
            tone = generate_pro_wave(freq, UNIT_SEC, idx, avg_v)
            
            pan_r = np.clip(cx / frame.shape[1], 0.1, 0.9)
            pan_l = 1.0 - pan_r
            
            end_s = start_s + len(tone)
            if end_s < len(tracks_l[0]):
                tracks_l[idx][start_s:end_s] += tone * pan_l
                tracks_r[idx][start_s:end_s] += tone * pan_r
            vis_data[idx].append(freq)
        
        for j in range(len(sorted_cnts), 4): vis_data[j].append(None)
        if u % 5 == 0: prog.progress(u / num_units)
    cap.release()

    # --- 메인 결과 UI ---
    st.header("🎞 Master Performance")
    col_v, col_g = st.columns([1, 1])
    
    with col_v:
        st.video(temp_path)
        # 전체 믹싱
        m_l, m_r = np.sum(tracks_l, axis=0), np.sum(tracks_r, axis=0)
        master = np.vstack((m_l, m_r)).T
        if np.max(np.abs(master)) > 0: master = (master / np.max(np.abs(master))) * 0.85
        
        m_io = io.BytesIO()
        with wave.open(m_io, 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes((master * 32767).astype(np.int16).tobytes())
        
        st.write("🔊 **Total Master Mix**")
        st.audio(m_io.getvalue())
        st.download_button("💾 전체 음원 저장", m_io.getvalue(), "master_mix.wav")

    with col_g:
        st.header("📊 MIDI Timeline")
        fig = go.Figure()
        t_axis = np.linspace(0, video_len, len(vis_data[0]))
        colors = ['#00E5FF', '#FF3D00', '#D500F9', '#FFEA00']
        for i in range(4):
            fig.add_trace(go.Scatter(x=t_axis, y=vis_data[i], name=f"Layer {i+1}", line=dict(color=colors[i])))
        fig.update_layout(template="plotly_dark", height=430)
        st.plotly_chart(fig, use_container_width=True)

    # --- 개별 플레이 및 저장 섹션 (강화된 기능) ---
    st.divider()
    st.subheader("📁 Layer Stems: 개별 플레이 및 저장")
    st.info("각 레이어의 소리를 하나씩 들어보고 개별 파일(WAV)로 저장할 수 있습니다.")
    
    cols = st.columns(4)
    layer_names = ["Deep Bass (저음)", "Warm Pluck (리듬)", "Airy Lead (멜로디)", "Crystal Bell (고음)"]
    
    for i in range(4):
        with cols[i]:
            # 개별 트랙 데이터 추출 및 노멀라이징
            l_data = np.vstack((tracks_l[i], tracks_r[i])).T
            p = np.max(np.abs(l_data))
            if p > 0: l_data = (l_data / p) * 0.75
            
            l_io = io.BytesIO()
            with wave.open(l_io, 'wb') as wf:
                wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                wf.writeframes((l_data * 32767).astype(np.int16).tobytes())
            
            # UI 배치: 제목 -> 개별 플레이어 -> 저장 버튼
            st.markdown(f"### Track {i+1}")
            st.caption(layer_names[i])
            st.audio(l_io.getvalue()) # 개별 플레이 기능
            st.download_button(
                label=f"📥 {i+1}번 트랙 저장",
                data=l_io.getvalue(),
                file_name=f"layer_{i+1}_{layer_names[i].split()[0]}.wav",
                mime="audio/wav",
                key=f"dl_{i}"
            )
