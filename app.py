import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Color Synth DAW")
st.title("🌈 Color & Light: Multi-Instrument DAW")

# --- 악기 설정 (색상별 특화 사운드) ---
# R: 날카로운 리드, G: 부드러운 패드, B: 깊은 베이스, Y: 퍼커션
INSTRUMENTS = {
    'Red': {'freqs': [440, 554, 659, 880], 'type': 'sawtooth'},   # 날카로운 전자음
    'Green': {'freqs': [196, 246, 293, 392], 'type': 'sine'},      # 부드러운 소리
    'Blue': {'freqs': [82, 110, 123, 146], 'type': 'triangle'},    # 묵직한 베이스
    'Yellow': {'freqs': [329, 392, 493, 587], 'type': 'square'}    # 톡톡 튀는 소리
}

def generate_wave(t, freq, instrument_type, brightness):
    if instrument_type == 'sawtooth': # 빨강: 날카로운 톱니파
        wave_data = 2 * (t * freq - np.floor(0.5 + t * freq))
    elif instrument_type == 'square': # 노랑: 딱딱한 사각파
        wave_data = np.sign(np.sin(2 * np.pi * freq * t))
    elif instrument_type == 'triangle': # 파랑: 웅장한 삼각파
        wave_data = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    else: # 초록: 부드러운 사인파
        wave_data = np.sin(2 * np.pi * freq * t)
    
    # 밝기에 따라 배음(Harmonics) 농도 조절
    harmonics = 0.3 * (brightness / 255) * np.sin(2 * np.pi * freq * 2 * t)
    return wave_data + harmonics

def apply_fade(tone, sample_rate):
    n = len(tone)
    if n < 100: return tone
    fade_len = int(sample_rate * 0.01) # 0.01초 페이드로 클릭 노이즈 제거
    window = np.ones(n)
    window[:fade_len] = np.linspace(0, 1, fade_len)
    window[-fade_len:] = np.linspace(1, 0, fade_len)
    return tone * window

with st.sidebar:
    st.header("🎛 Synth Mixer")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.divider()
    target_tracks = st.multiselect(
        "🔊 플레이할 색상 레이어 선택",
        ["Red (Lead)", "Green (Pad)", "Blue (Bass)", "Yellow (Synth)"],
        default=["Red (Lead)", "Green (Pad)", "Blue (Bass)", "Yellow (Synth)"]
    )
    sensitivity = st.slider("감지 민감도 (낮을수록 민감)", 30, 200, 100)
    st.info("빨강, 초록, 파랑, 노랑 계열의 빛을 분석하여 서로 다른 악기 소리를 냅니다.")

if uploaded_file:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = 22050 
        duration = total_frames / fps
        
        # 4개 색상 트랙 초기화
        color_names = ['Red', 'Green', 'Blue', 'Yellow']
        tracks_l = {name: np.zeros(int(sample_rate * duration) + sample_rate) for name in color_names}
        tracks_r = {name: np.zeros(int(sample_rate * duration) + sample_rate) for name in color_names}
        visual_data = {name: [] for name in color_names}
        
        prog = st.progress(0)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            start_idx = int(i * (sample_rate / fps))
            t = np.linspace(0, 1/fps, int(sample_rate/fps), False).astype(np.float32)

            # 색상 범위 정의 (HSV)
            color_masks = {
                'Red': cv2.inRange(hsv, (0, 100, sensitivity), (10, 255, 255)) + cv2.inRange(hsv, (160, 100, sensitivity), (180, 255, 255)),
                'Green': cv2.inRange(hsv, (40, 100, sensitivity), (80, 255, 255)),
                'Blue': cv2.inRange(hsv, (100, 100, sensitivity), (140, 255, 255)),
                'Yellow': cv2.inRange(hsv, (20, 100, sensitivity), (35, 255, 255))
            }

            for name, mask in color_masks.items():
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    best_cnt = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(best_cnt)
                    if area > 50:
                        M = cv2.moments(best_cnt)
                        cx = int(M["m10"]/M["m00"])
                        brightness = np.mean(frame[mask > 0])
                        
                        freq = INSTRUMENTS[name]['freqs'][int(area % 4)]
                        vol = min(area / 2000, 0.7)
                        
                        tone = vol * generate_wave(t, freq, INSTRUMENTS[name]['type'], brightness)
                        tone = apply_fade(tone, sample_rate)
                        
                        pan_r = cx / frame.shape[1]
                        pan_l = 1.0 - pan_r
                        
                        end_idx = start_idx + len(tone)
                        if end_idx < len(tracks_l[name]):
                            tracks_l[name][start_idx:end_idx] += tone * pan_l
                            tracks_r[name][start_idx:end_idx] += tone * pan_r
                        visual_data[name].append(freq)
                    else: visual_data[name].append(None)
                else: visual_data[name].append(None)
            
            if i % 30 == 0: prog.progress(i / total_frames)

        # 믹싱
        master_l, master_r = np.zeros_like(tracks_l['Red']), np.zeros_like(tracks_r['Red'])
        for layer_opt in target_tracks:
            c_name = layer_opt.split()[0]
            master_l += tracks_l[c_name]
            master_r += tracks_r[c_name]
            
        master_stereo = np.vstack((master_l, master_r)).T
        peak = np.max(np.abs(master_stereo))
        if peak > 0: master_stereo = (master_stereo / peak) * 0.8
        audio_int16 = np.clip(master_stereo * 32767, -32768, 32767).astype(np.int16)

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(audio_int16.tobytes())

        col_left, col_right = st.columns([1.5, 1])
        with col_left:
            st.header("🎞 Performance Display")
            st.video(uploaded_file)
            st.audio(wav_buf.getvalue())
            st.download_button("💾 전체 믹스 다운로드", wav_buf.getvalue(), "color_synth_mix.wav")

        with col_right:
            st.header("📊 Color Timeline")
            time_axis = np.linspace(0, duration, total_frames)
            fig = go.Figure()
            color_map = {'Red': 'red', 'Green': 'green', 'Blue': 'blue', 'Yellow': 'yellow'}
            for name in color_names:
                if any(x in name for x in target_tracks):
                    fig.add_trace(go.Scatter(x=time_axis, y=visual_data[name], name=name, line=dict(color=color_map[name])))
            fig.update_layout(template="plotly_dark", height=400, xaxis_title="Time", yaxis_title="Pitch", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"오류: {e}")
