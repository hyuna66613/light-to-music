import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Optical Physics DAW")
st.title("🔦 Optical Physics Synth: Light-to-Sound Mapping")

# --- 물리 기반 매핑 엔진 ---
def generate_phys_tone(t, freq, area, color_temp, intensity, sample_rate):
    """
    area (면적) -> Bass/Sub 성분 결정
    color_temp (색온도/색상) -> 기본 주파수 및 배음 구조
    intensity (세기/밝기) -> Cutoff Filter (소리의 선명도)
    """
    # 1. 면적에 따른 무게감 (면적이 클수록 서브 하모닉스 추가)
    base_wave = np.sin(2 * np.pi * freq * t)
    if area > 1000:
        base_wave += 0.5 * np.sin(2 * np.pi * (freq/2) * t)
    
    # 2. 색온도 기반 배음 (차가운 색일수록 날카로운 사각파 혼합)
    # color_temp: 0(따뜻함/적색) ~ 180(차가움/청색)
    overtone_ratio = color_temp / 180.0
    wave_shape = (1 - overtone_ratio) * base_wave + overtone_ratio * np.sign(base_wave)
    
    # 3. 세기(Intensity) 기반 필터링 효과
    # 밝기가 낮으면 고주파를 깎고, 밝으면 날카롭게 (Low-pass effect)
    cutoff = max(0.1, intensity / 255.0)
    wave_shape = wave_shape * cutoff
    
    return wave_shape.astype(np.float32)

def apply_sustain(tone, sample_rate, persistence):
    """
    persistence (지속성) -> Reverb/Release 결정
    """
    n = len(tone)
    # 지속성이 높을수록 테일(Tail)이 긴 엔벨로프 적용
    release_time = min(0.1 + (persistence * 0.4), 0.5) 
    release_samples = int(sample_rate * release_time)
    
    if n > release_samples:
        env = np.ones(n)
        env[-release_samples:] = np.linspace(1, 0, release_samples)
        return tone * env
    return tone

with st.sidebar:
    st.header("🔬 Physics Analysis")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.divider()
    threshold_val = st.slider("광원 인식 문턱값 (Intensity)", 50, 255, 200)
    min_area = st.number_input("최소 감지 면적 (Area)", 10, 1000, 100)
    master_vol = st.slider("Master Gain", 0.1, 5.0, 1.5)

if uploaded_file:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = 22050 
        
        # 4채널 레이어 (면적순)
        tracks_l = [np.zeros(int(sample_rate * (total_frames/fps)) + sample_rate) for _ in range(4)]
        tracks_r = [np.zeros(int(sample_rate * (total_frames/fps)) + sample_rate) for _ in range(4)]
        
        # 데이터 시각화용
        vis_intensity = [[] for _ in range(4)]
        
        prog = st.progress(0)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_idx = int(i * (sample_rate / fps))
            t = np.linspace(0, 1/fps, int(sample_rate/fps), False).astype(np.float32)
            
            # 광원을 면적순으로 4개 분석
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
            
            for idx, cnt in enumerate(sorted_cnts):
                area = cv2.contourArea(cnt)
                if area < min_area: continue
                
                # 1. 색상(Hue) -> 색온도 대용
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                avg_hsv = cv2.mean(hsv, mask=mask)
                color_temp = avg_hsv[0] # Hue값
                
                # 2. 세기(Intensity)
                intensity = cv2.mean(gray, mask=mask)[0]
                
                # 3. 지속성 (단순 프레임 분석 대신 면적 가중치 활용)
                persistence = area / 5000.0
                
                # 주파수 매핑 (색온도와 면적 조합)
                freq = 100 + (color_temp * 2) + (10000 / (area + 1))
                
                # 사운드 생성
                tone = generate_phys_tone(t, freq, area, color_temp, intensity, sample_rate)
                tone = apply_sustain(tone, sample_rate, persistence)
                
                # 위치 기반 팬닝
                M = cv2.moments(cnt)
                cx = int(M["m10"]/M["m00"]) if M["m00"] != 0 else frame.shape[1]//2
                pan_r = cx / frame.shape[1]
                pan_l = 1.0 - pan_r
                
                end_idx = start_idx + len(tone)
                if end_idx < len(tracks_l[0]):
                    tracks_l[idx][start_idx:end_idx] += tone * pan_l * master_vol
                    tracks_r[idx][start_idx:end_idx] += tone * pan_r * master_vol
                vis_intensity[idx].append(intensity)

            for j in range(len(sorted_cnts), 4): vis_intensity[j].append(0)
            if i % 30 == 0: prog.progress(i / total_frames)

        # 믹싱 및 마스터링
        master_l = np.clip(np.sum(tracks_l, axis=0), -1, 1)
        master_r = np.clip(np.sum(tracks_r, axis=0), -1, 1)
        master_stereo = np.vstack((master_l, master_r)).T
        audio_int16 = (master_stereo * 32767).astype(np.int16)

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(audio_int16.tobytes())

        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.header("📽 Optical Sync Analysis")
            st.video(uploaded_file)
            st.audio(wav_buf.getvalue())
            
            st.subheader("Layer Mixer (Monitoring)")
            for i in range(4):
                col_btn, col_info = st.columns([1, 2])
                with col_btn:
                    # 개별 트랙 추출 기능 유지
                    t_buf = io.BytesIO()
                    t_data = np.vstack((tracks_l[i], tracks_r[i])).T
                    t_int16 = (np.clip(t_data, -1, 1) * 32767).astype(np.int16)
                    with wave.open(t_buf, 'wb') as wf:
                        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(t_int16.tobytes())
                    st.download_button(f"📥 Layer {i+1} WAV", t_buf.getvalue(), f"layer_{i+1}.wav")
                with col_info:
                    st.caption(f"Track {i+1}: Intensity-driven Resonance")

        with col2:
            st.header("📊 Physical Data")
            time_axis = np.linspace(0, total_frames/fps, total_frames)
            fig = go.Figure()
            for i in range(4):
                fig.add_trace(go.Scatter(x=time_axis, y=vis_intensity[i], name=f"L{i+1} Intensity", fill='tozeroy'))
            fig.update_layout(template="plotly_dark", height=400, xaxis_title="Time", yaxis_title="Intensity (0-255)")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"분석 실패: {e}")
