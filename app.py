import streamlit as st
import cv2
import numpy as np
import io
import plotly.graph_objects as go

# --- 기본 설정 ---
st.set_page_config(layout="wide", page_title="GarageLight DAW")
st.title("🎼 GarageLight: Optical Synth DAW (Error-Free Mode)")

# 밤의 펜타토닉 음계
NOTES = [130.81, 146.83, 164.81, 196.00, 220.00, 261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25, 783.99, 880.00]

def get_nearest_note(freq):
    return NOTES[np.abs(np.array(NOTES) - freq).argmin()]

def apply_envelope(tone, sample_rate):
    n = len(tone)
    if n < 100: return tone
    attack = int(min(sample_rate * 0.02, n * 0.15))
    release = int(min(sample_rate * 0.08, n * 0.3))
    env = np.ones(n, dtype=np.float32)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return tone * env

with st.sidebar:
    st.header("🎛 Control Panel")
    uploaded_file = st.file_uploader("영상을 업로드하세요 (소리 무시 모드)", type=['mp4', 'mov', 'avi'])
    vol_boost = st.slider("마스터 볼륨", 0.5, 3.0, 1.0)

if uploaded_file:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = 22050 
        
        audio_len = int(sample_rate * (total_frames / fps)) + sample_rate
        master_l = np.zeros(audio_len, dtype=np.float32)
        master_r = np.zeros(audio_len, dtype=np.float32)
        
        tracks_visual = [[] for _ in range(6)]
        prog_bar = st.progress(0)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_idx = int(i * (sample_rate / fps))
            t = np.linspace(0, 1/fps, int(sample_rate/fps), False).astype(np.float32)
            
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
            for idx, cnt in enumerate(sorted_cnts):
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                
                note_f = get_nearest_note(150 + (frame.shape[0]-cy)*1.8)
                area_vol = min(cv2.contourArea(cnt)/1200, 0.4) * vol_boost
                
                tone = area_vol * (np.sin(2 * np.pi * note_f * t) + 0.3 * np.sin(2 * np.pi * note_f * 2 * t))
                tone = apply_envelope(tone, sample_rate)
                
                pan_r = cx / frame.shape[1]
                pan_l = 1.0 - pan_r
                
                end_idx = start_idx + len(tone)
                if end_idx < audio_len:
                    master_l[start_idx:end_idx] += tone * pan_l
                    master_r[start_idx:end_idx] += tone * pan_r
                
                tracks_visual[idx].append({'t': i/fps, 'f': note_f})
            
            if i % 30 == 0: prog_bar.progress(min(i / total_frames, 1.0))

        # --- [에러 방지 핵심: 데이터 타입을 아주 명확하게 고정] ---
        master_stereo = np.vstack((master_l, master_r)).T
        peak = np.max(np.abs(master_stereo))
        if peak > 0:
            master_stereo = master_stereo / peak
        
        # 정수형 변환 시 발생할 수 있는 오류를 피하기 위해 float32를 그대로 사용하거나 
        # 아주 보수적인 정수 변환을 거칩니다.
        final_audio = np.clip(master_stereo * 32767, -32768, 32767).astype(np.int16)

        col_v, col_g = st.columns([1, 1])
        with col_v:
            st.header("📽 Video Stream")
            st.video(uploaded_file)
            st.write("🎹 합성된 마스터 음원")
            
            # [수정] wavfile.write 대신 numpy 배열을 직접 audio에 넣습니다.
            st.audio(final_audio, sample_rate=sample_rate)
            
            # 다운로드 버튼 부분에서만 예외 처리를 강화합니다.
            try:
                import io
                from scipy.io import wavfile
                buf = io.BytesIO()
                # 데이터 형식을 명시적으로 지정하여 에러 차단
                wavfile.write(buf, sample_rate, final_audio.astype('<i2')) 
                st.download_button("💾 음악 다운로드 (WAV)", buf.getvalue(), "night_bus_music.wav")
            except:
                st.warning("다운로드 파일 생성 중 문제가 발생했지만 재생은 가능합니다.")

        with col_g:
            st.header("📊 Frequency Timeline")
            for idx in range(6):
                if tracks_visual[idx]:
                    v_t = [v['t'] for v in tracks_visual[idx]]
                    v_f = [v['f'] for v in tracks_visual[idx]]
                    fig = go.Figure(go.Scatter(x=v_t, y=v_f, mode='lines', line=dict(color='#00d1ff', width=1.5)))
                    fig.update_layout(height=110, margin=dict(l=0,r=0,t=10,b=10), xaxis_title="Time (s)", yaxis_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
