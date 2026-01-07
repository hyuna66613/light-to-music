import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

# 파일 업로드 용량 제한 해제 시뮬레이션 및 설정
st.set_page_config(layout="wide", page_title="Professional Optical DAW")
st.title("🎹 Optical Music Station (High Stability)")

# --- 글로벌 설정 ---
BPM = 120
SAMPLE_RATE = 22050
BEAT_SEC = 60 / BPM 
UNIT_SEC = BEAT_SEC / 2  # 8분 음표 단위 분석 (산만함 감소)

def apply_pro_eq(tone, layer_idx, brightness_factor):
    """레이어별 EQ 및 필터: 빛의 밝기에 따라 소리의 개방감 조절"""
    n = len(tone)
    # 저음 레이어는 고음 커트, 고음 레이어는 저음 커트
    if layer_idx == 0: # Bass: 묵직하게
        env = np.exp(-np.linspace(0, 2, n)) 
        return tone * env * 1.2
    elif layer_idx == 3: # Bell: 빛이 밝을수록 더 맑게
        env = np.exp(-np.linspace(0, 15, n))
        return tone * env * (0.5 + brightness_factor)
    else:
        env = np.sin(np.linspace(0, np.pi, n))
        return tone * env

@st.cache_data
def analyze_video_characteristics(video_path):
    """영상 전체의 무드를 분석하여 사운드 톤 결정"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret: return 0.5, 0.5 # 기본값
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    avg_v = np.mean(hsv[:,:,2]) / 255  # 평균 밝기
    avg_h = np.mean(hsv[:,:,0]) / 180  # 평균 색상 (온도)
    cap.release()
    return avg_v, avg_h

def generate_wave(freq, duration, layer_idx, mood_v):
    """영상 무드(mood_v)가 반영된 파형 생성"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    # 밝은 영상일수록 파형이 날카로워짐 (배음 추가)
    if layer_idx == 0:
        wave_data = np.sin(2 * np.pi * freq * t)
    elif layer_idx == 1:
        wave_data = 0.5 * np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * freq * 1.5 * t)
    else:
        # 무드에 따른 파형 변화
        wave_data = (1-mood_v) * np.sin(2 * np.pi * freq * t) + mood_v * np.sign(np.sin(2 * np.pi * freq * t))
    
    return apply_pro_eq(wave_data, layer_idx, mood_v)

uploaded_file = st.file_uploader("영상을 업로드하세요 (최적화 완료)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    # 임시 파일 저장 (메모리 확보)
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # 1. 영상 특색 분석
    avg_brightness, avg_hue = analyze_video_characteristics(temp_path)
    st.info(f"✨ 영상 분석 완료: {'밝고 차가운' if avg_brightness > 0.5 else '어둡고 따뜻한'} 무드의 사운드를 생성합니다.")

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_len = total_frames / fps
    
    num_units = int(video_len / UNIT_SEC)
    tracks_l = [np.zeros(int(SAMPLE_RATE * video_len) + 100) for _ in range(4)]
    tracks_r = [np.zeros(int(SAMPLE_RATE * video_len) + 100) for _ in range(4)]
    vis_data = [[] for _ in range(4)]

    prog = st.progress(0)
    for u in range(num_units):
        # 정해진 비트 타이밍의 프레임만 정확히 짚어서 분석 (산만함 제거 핵심)
        target_frame = int(u * UNIT_SEC * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 빛의 세기(Intensity) 기반 동적 임계값
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
        start_s = int(u * UNIT_SEC * SAMPLE_RATE)
        
        for idx, cnt in enumerate(sorted_cnts):
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"]/M["m00"])
            
            # 영상 무드와 레이어에 맞춤화된 주파수 (C-Major 기반)
            base_freqs = [65, 130, 261, 523]
            freq = base_freqs[idx] + (avg_hue * 50) + (area % 20)
            
            tone = generate_wave(freq, UNIT_SEC, idx, avg_brightness)
            
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

    # --- 믹싱 & 마스터링 ---
    master_l, master_r = np.sum(tracks_l, axis=0), np.sum(tracks_r, axis=0)
    master_stereo = np.vstack((master_l, master_r)).T
    peak = np.max(np.abs(master_stereo))
    if peak > 0: master_stereo = (master_stereo / peak) * 0.8
    
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes((master_stereo * 32767).astype(np.int16).tobytes())

    # --- 결과 UI ---
    col_v, col_g = st.columns([1, 1])
    with col_v:
        st.header("🎞 Sync View")
        st.video(temp_path)
        st.audio(wav_io.getvalue())
        st.download_button("💾 전체 음원 저장", wav_io.getvalue(), "optical_pro_mix.wav")

    with col_g:
        st.header("📊 MIDI Quantized Graph")
        fig = go.Figure()
        colors = ['#00E5FF', '#FF3D00', '#D500F9', '#FFEA00']
        t_axis = np.linspace(0, video_len, len(vis_data[0]))
        for i in range(4):
            fig.add_trace(go.Scatter(x=t_axis, y=vis_data[i], name=f"Layer {i+1}", line=dict(color=colors[i])))
        fig.update_layout(template="plotly_dark", height=400, xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)

    # 레이어별 개별 청취 및 저장
    st.divider()
    st.subheader("📁 Layer Stems")
    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            l_io = io.BytesIO()
            l_data = np.vstack((tracks_l[i], tracks_r[i])).T
            l_peak = np.max(np.abs(l_data))
            if l_peak > 0: l_data = (l_data / l_peak) * 0.7
            with wave.open(l_io, 'wb') as wf:
                wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                wf.writeframes((l_data * 32767).astype(np.int16).tobytes())
            st.write(f"Track {i+1}")
            st.audio(l_io.getvalue())
            st.download_button(f"📥 Layer {i+1} 저장", l_io.getvalue(), f"layer_{i+1}.wav")
