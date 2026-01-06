import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

# --- 기본 설정 ---
st.set_page_config(layout="wide", page_title="GarageLight DAW")
st.title("🎼 GarageLight: Precise Optical Synth (Zero Error Mode)")

# 밤의 펜타토닉 음계 (정확한 주파수 연산)
NOTES = [130.81, 146.83, 164.81, 196.00, 220.00, 261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25, 783.99, 880.00]

def get_nearest_note(freq):
    return NOTES[np.abs(np.array(NOTES) - freq).argmin()]

def apply_envelope(tone, sample_rate):
    n = len(tone)
    if n < 100: return tone
    # 부드러운 시작과 끝 (0.02초/0.08초)
    attack = int(min(sample_rate * 0.02, n * 0.15))
    release = int(min(sample_rate * 0.08, n * 0.3))
    env = np.ones(n, dtype=np.float32)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return tone * env

with st.sidebar:
    st.header("🎛 Control Panel")
    uploaded_file = st.file_uploader("영상을 업로드하세요 (소리 무시 모드)", type=['mp4', 'mov', 'avi'])
    vol_boost = st.slider("마스터 볼륨 증폭", 0.1, 1.0, 0.58)

if uploaded_file:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = 22050 
        
        # 오디오 도화지 (float32로 정밀 연산)
        audio_len = int(sample_rate * (total_frames / fps)) + sample_rate
        master_l = np.zeros(audio_len, dtype=np.float32)
        master_r = np.zeros(audio_len, dtype=np.float32)
        
        tracks_visual = [[] for _ in range(6)]
        prog_bar = st.progress(0)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            # 빛 감지 연산 (광원 단위 추출)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_idx = int(i * (sample_rate / fps))
            t = np.linspace(0, 1/fps, int(sample_rate/fps), False).astype(np.float32)
            
            # 가장 밝은 불빛 6개 트래킹
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
            for idx, cnt in enumerate(sorted_cnts):
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                
                # 음높이(y좌표)와 볼륨(크기) 연산
                note_f = get_nearest_note(150 + (frame.shape[0]-cy)*1.8)
                area_vol = min(cv2.contourArea(cnt)/1200, 0.4) * vol_boost
                
                # 전자음 합성 (기본음 + 1옥타브 배음)
                tone = area_vol * (np.sin(2 * np.pi * note_f * t) + 0.3 * np.sin(2 * np.pi * note_f * 2 * t))
                tone = apply_envelope(tone, sample_rate)
                
                # 좌우 공간감 (x좌표)
                pan_r = cx / frame.shape[1]
                pan_l = 1.0 - pan_r
                
                end_idx = start_idx + len(tone)
                if end_idx < audio_len:
                    master_l[start_idx:end_idx] += tone * pan_l
                    master_r[start_idx:end_idx] += tone * pan_r
                
                tracks_visual[idx].append({'t': i/fps, 'f': note_f})
            
            if i % 30 == 0:
                prog_bar.progress(min(i / total_frames, 1.0))

        # --- [에러 방지 핵심: 바이트 스트림 렌더링] ---
        # 1. 스테레오 합치기 및 노멀라이징
        master_stereo = np.vstack((master_l, master_r)).T
        peak = np.max(np.abs(master_stereo))
        if peak > 0:
            master_stereo = (master_stereo / peak) * 0.9  # 안전 헤드룸 확보
        
        # 2. 16비트 부호 있는 정수(signed int16)로 강제 변환
        audio_int16 = np.clip(master_stereo * 32767, -32768, 32767).astype(np.int16)

        # 3. [에러 방지] wave 모듈을 사용해 수동으로 WAV 바이너리 생성
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(2)        # 스테레오
            wf.setsampwidth(2)        # 16비트 (2바이트)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes()) # 바이트 단위로 직접 주입

        col_v, col_g = st.columns([1, 1])
        with col_v:
            st.header("📽 Video Stream")
            st.video(uploaded_file)
            st.write("🎹 합성된 마스터 음원")
            st.audio(wav_buf.getvalue())
            st.download_button("💾 음악 다운로드 (WAV)", wav_buf.getvalue(), "night_bus_music.wav")

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
        st.error(f"정밀 연산 중 오류 발생: {e}")
