import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Electro Light DAW")
st.title("🚀 Electro Light: Synth & Beat DAW")

# --- 일렉트로닉 사운드 디자인 (MIDI 레퍼런스) ---
# 4개의 트랙에 각기 다른 악기 성격을 부여 (베이스, 리드, 신스, 퍼커션 느낌)
SCALES = [
    [55.00, 65.41, 73.42, 82.41], # Track 1: Deep Bass (E-G-A-B)
    [110.00, 130.81, 146.83, 164.81], # Track 2: Mid Synth
    [220.00, 261.63, 293.66, 329.63], # Track 3: High Lead
    [440.00, 523.25, 587.33, 659.25]  # Track 4: Shimmer/Perc
]

def apply_electronic_synth(tone, sample_rate, brightness):
    n = len(tone)
    # 1. 일렉트로닉 특유의 날카로운 파형 (사각파 혼합)
    square = np.sign(tone) * np.abs(tone)
    tone = (tone * 0.7) + (square * 0.3 * (brightness / 255))
    
    # 2. ADSR: 일렉트로닉 특유의 톡 쏘는 Attack
    env = np.ones(n, dtype=np.float32)
    attack = int(n * 0.1) # 짧고 강한 시작
    release = int(n * 0.6) # 긴 잔향
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    
    # 3. Low Pass Filter 효과 (밝기에 따라 소리의 먹먹함 조절)
    # 실제 필터 대신 고주파 성분 제어로 시뮬레이션
    return tone * env

with st.sidebar:
    st.header("🎛 Synth Engine")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.divider()
    sensitivity = st.slider("빛 감지 민감도 (낮을수록 잘 잡힘)", 100, 250, 180)
    master_gain = st.slider("마스터 볼륨", 0.1, 2.0, 1.2)

if uploaded_file:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = 22050 
        
        # 4개 레이어 설정
        num_tracks = 4
        tracks_l = [np.zeros(int(sample_rate * (total_frames / fps)) + sample_rate) for _ in range(num_tracks)]
        tracks_r = [np.zeros(int(sample_rate * (total_frames / fps)) + sample_rate) for _ in range(num_tracks)]
        visual_data = [[] for _ in range(num_tracks)]
        
        prog = st.progress(0)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            # 일반화를 위해 밝기/크기/면적 추출
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 밝기 강조를 위해 민감도 적용
            _, thresh = cv2.threshold(gray, sensitivity, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_idx = int(i * (sample_rate / fps))
            t = np.linspace(0, 1/fps, int(sample_rate/fps), False).astype(np.float32)
            
            # 상위 4개 광원 분석
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:num_tracks]
            
            for idx, cnt in enumerate(sorted_cnts):
                area = cv2.contourArea(cnt)
                # 평균 밝기 계산
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_brightness = cv2.mean(gray, mask=mask)[0]
                
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx = int(M["m10"]/M["m00"])
                
                # [핵심] 면적에 따라 음높이 결정, 밝기에 따라 음색 결정
                scale = SCALES[idx]
                note_idx = int((area % 1000) / 250) % len(scale)
                freq = scale[note_idx]
                
                # 밝기와 면적을 조합한 볼륨 (작은 불빛도 밝으면 소리가 나게 설정)
                vol = (min(area / 1000, 0.6) * 0.5 + (mean_brightness / 255) * 0.5) * master_gain
                
                # 미디 스타일 합성음 생성
                tone = vol * np.sin(2 * np.pi * freq * t)
                tone = apply_electronic_synth(tone, sample_rate, mean_brightness)
                
                # 스테레오 팬닝
                pan_r = cx / frame.shape[1]
                pan_l = 1.0 - pan_r
                
                end_idx = start_idx + len(tone)
                if end_idx < len(tracks_l[0]):
                    tracks_l[idx][start_idx:end_idx] += tone * pan_l
                    tracks_r[idx][start_idx:end_idx] += tone * pan_r
                
                visual_data[idx].append(freq)
            
            # 데이터 동기화
            for j in range(len(sorted_cnts), num_tracks):
                visual_data[j].append(None)
            
            if i % 30 == 0: prog.progress(i / total_frames)

        # 믹싱
        master_l = np.sum(tracks_l, axis=0)
        master_r = np.sum(tracks_r, axis=0)
        master_stereo = np.vstack((master_l, master_r)).T
        
        # 안전한 노멀라이징
        peak = np.max(np.abs(master_stereo))
        if peak > 0: master_stereo = (master_stereo / peak) * 0.8
        audio_int16 = np.clip(master_stereo * 32767, -32768, 32767).astype(np.int16)

        # WAV 바이너리
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        # --- UI ---
        col1, col2 = st.columns([1, 1])
        with col1:
            st.header("🎞 Video & MIDI Player")
            st.video(uploaded_file)
            st.audio(wav_buf.getvalue())
            
            # 레이어별 저장 기능 복구
            st.subheader("Layer Export")
            selected_layer = st.selectbox("저장할 레이어 선택", [f"Track {i+1}" for i in range(num_tracks)])
            layer_idx = int(selected_layer.split()[-1]) - 1
            
            # 개별 레이어 WAV 생성
            l_buf = io.BytesIO()
            l_data = np.vstack((tracks_l[layer_idx], tracks_r[layer_idx])).T
            l_peak = np.max(np.abs(l_data))
            if l_peak > 0: l_data = (l_data / l_peak) * 0.8
            l_int16 = np.clip(l_data * 32767, -32768, 32767).astype(np.int16)
            with wave.open(l_buf, 'wb') as wf:
                wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate)
                wf.writeframes(l_int16.tobytes())
            
            st.download_button(f"💾 {selected_layer} 다운로드", l_buf.getvalue(), f"{selected_layer}.wav")

        with col2:
            st.header("📊 MIDI Timeline")
            time_axis = np.linspace(0, total_frames/fps, total_frames)
            fig = go.Figure()
            colors = ['#FF4B4B', '#1C83E1', '#00D1FF', '#7752FE']
            for i in range(num_tracks):
                fig.add_trace(go.Scatter(x=time_axis, y=visual_data[i], name=f"Track {i+1}", line=dict(color=colors[i], width=2)))
            
            fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Time (sec)", yaxis_title="Note Pitch", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"연산 오류: {e}")
