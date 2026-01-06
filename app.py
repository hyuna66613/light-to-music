import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Electro Sync Pro v3")
st.title("🚀 Electro Sync: Multi-Track Mixer & Sync")

# --- 악기별 성격 부여 (Bass, Mid, Lead, Perc) ---
SCALES = [
    [55.00, 65.41, 73.42, 82.41],   # Track 1: Bass
    [110.00, 130.81, 146.83, 164.81], # Track 2: Mid
    [220.00, 261.63, 293.66, 329.63], # Track 3: High
    [440.00, 523.25, 587.33, 659.25]  # Track 4: Perc
]

def apply_synth(tone, sample_rate, brightness):
    n = len(tone)
    square = np.sign(tone) * np.abs(tone)
    tone = (tone * 0.6) + (square * 0.4 * (brightness / 255))
    env = np.ones(n, dtype=np.float32)
    attack, release = int(n * 0.1), int(n * 0.6)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return tone * env

# --- 사이드바: 믹서 컨트롤 ---
with st.sidebar:
    st.header("🎛 믹서 콘솔")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.divider()
    
    # [핵심] 여러 개 선택 가능한 트랙 리스트
    # 이 리스트에서 선택한 것들만 합쳐서 소리가 납니다.
    target_tracks = st.multiselect(
        "🔊 플레이/믹싱할 트랙 선택",
        ["Track 1", "Track 2", "Track 3", "Track 4"],
        default=["Track 1", "Track 2", "Track 3", "Track 4"],
        help="선택한 트랙들만 합쳐져서 마스터 음원으로 출력됩니다."
    )
    
    st.divider()
    sensitivity = st.slider("빛 감지 민감도", 50, 255, 180)
    master_gain = st.slider("마스터 볼륨", 0.1, 3.0, 1.2)

if uploaded_file:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        sample_rate = 22050 
        
        # 4개 트랙 독립 배열
        raw_tracks_l = [np.zeros(int(sample_rate * duration) + sample_rate) for _ in range(4)]
        raw_tracks_r = [np.zeros(int(sample_rate * duration) + sample_rate) for _ in range(4)]
        visual_data = [[] for _ in range(4)]
        
        prog = st.progress(0)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, sensitivity, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            start_idx = int(i * (sample_rate / fps))
            t = np.linspace(0, 1/fps, int(sample_rate/fps), False).astype(np.float32)
            
            sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
            for idx, cnt in enumerate(sorted_cnts):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx = int(M["m10"]/M["m00"])
                
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                brightness = cv2.mean(gray, mask=mask)[0]
                
                freq = SCALES[idx][int((area % 1000) / 250) % 4]
                vol = (min(area / 1000, 0.6) * 0.5 + (brightness / 255) * 0.5) * master_gain
                
                tone = apply_synth(vol * np.sin(2 * np.pi * freq * t), sample_rate, brightness)
                
                pan_r = cx / frame.shape[1]
                pan_l = 1.0 - pan_r
                
                end_idx = start_idx + len(tone)
                if end_idx < len(raw_tracks_l[0]):
                    raw_tracks_l[idx][start_idx:end_idx] += tone * pan_l
                    raw_tracks_r[idx][start_idx:end_idx] += tone * pan_r
                visual_data[idx].append(freq)
            
            for j in range(len(sorted_cnts), 4):
                visual_data[j].append(None)
            
            if i % 30 == 0: prog.progress(i / total_frames)

        # --- [믹싱 로직] 선택된 트랙만 합산 ---
        master_l = np.zeros_like(raw_tracks_l[0])
        master_r = np.zeros_like(raw_tracks_r[0])
        
        for t_name in target_tracks:
            idx = int(t_name.split()[-1]) - 1
            master_l += raw_tracks_l[idx]
            master_r += raw_tracks_r[idx]
            
        master_stereo = np.vstack((master_l, master_r)).T
        peak = np.max(np.abs(master_stereo))
        if peak > 0: master_stereo = (master_stereo / peak) * 0.8
        audio_int16 = np.clip(master_stereo * 32767, -32768, 32767).astype(np.int16)

        # 마스터 WAV 생성
        master_wav = io.BytesIO()
        with wave.open(master_wav, 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        # --- 메인 UI 레이아웃 ---
        col_m, col_t = st.columns([1.5, 1])
        
        with col_m:
            st.header("🎞 동기화 플레이어")
            st.video(uploaded_file)
            st.write(f"🎶 현재 믹스: {', '.join(target_tracks)}")
            st.audio(master_wav.getvalue())
            st.download_button("💾 현재 믹스 다운로드", master_wav.getvalue(), "my_electro_mix.wav")

        with col_t:
            st.header("📊 트랙 타임라인")
            time_axis = np.linspace(0, duration, total_frames)
            fig = go.Figure()
            colors = ['#FF4B4B', '#1C83E1', '#00D1FF', '#7752FE']
            
            for i in range(4):
                # [그래프 연동] 선택한 트랙만 그래프에 표시
                if f"Track {i+1}" in target_tracks:
                    fig.add_trace(go.Scatter(
                        x=time_axis, y=visual_data[i], 
                        name=f"Track {i+1}", 
                        line=dict(color=colors[i], width=2),
                        connectgaps=False
                    ))
            
            fig.update_layout(
                template="plotly_dark", height=400,
                xaxis=dict(title="Time (s)", rangeslider=dict(visible=True)),
                yaxis=dict(title="Pitch (Hz)"),
                margin=dict(l=10, r=10, t=10, b=10),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        # 개별 트랙 전용 저장소
        st.divider()
        st.subheader("📁 개별 트랙 내보내기")
        ex_col1, ex_col2 = st.columns([1, 1])
        
        for i in range(4):
            with (ex_col1 if i < 2 else ex_col2):
                t_buf = io.BytesIO()
                t_data = np.vstack((raw_tracks_l[i], raw_tracks_r[i])).T
                t_peak = np.max(np.abs(t_data))
                if t_peak > 0: t_data = (t_data / t_peak) * 0.8
                t_int16 = np.clip(t_data * 32767, -32768, 32767).astype(np.int16)
                with wave.open(t_buf, 'wb') as wf:
                    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate)
                    wf.writeframes(t_int16.tobytes())
                st.download_button(f"Track {i+1}만 다운로드", t_buf.getvalue(), f"track_{i+1}.wav")

    except Exception as e:
        st.error(f"연산 중 오류: {e}")
