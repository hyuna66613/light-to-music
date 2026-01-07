import streamlit as st
import cv2
import numpy as np
import io
import wave
import plotly.graph_objects as go
from midiutil import MIDIFile  # MIDI 생성 라이브러리 추가

st.set_page_config(layout="wide", page_title="Optical to MIDI DAW")
st.title("🎹 Optical MIDI Composer: Export to GarageBand")

# --- 설정 ---
BPM = 120
SAMPLE_RATE = 22050
BEAT_SEC = 60 / BPM 
UNIT_SEC = BEAT_SEC / 2  # 8분 음표 단위 분석

# 주파수를 MIDI 노트 번호로 변환하는 함수
def freq_to_midi(freq):
    if freq <= 0: return 60 # 기본값 C4
    return int(12 * np.log2(freq / 440.0) + 69)

def generate_pro_wave(freq, duration, layer_idx, mood_v):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    if layer_idx == 0: # Bass
        wave_data = np.sin(2 * np.pi * freq * t)
    elif layer_idx == 3: # Bell
        wave_data = np.sin(2 * np.pi * freq * t + 0.5 * np.sin(2 * np.pi * freq * 2.01 * t))
    else: # Pluck
        wave_data = 0.6 * np.sin(2 * np.pi * freq * t) + 0.4 * np.sign(np.sin(2 * np.pi * freq * t))
    
    # Envelope
    n = len(tone := wave_data)
    env = np.exp(-np.linspace(0, 5, n))
    return (tone * env).astype(np.float32)

uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_len = total_frames / fps
    num_units = int(video_len / UNIT_SEC)
    
    # 오디오 데이터 및 MIDI 데이터 저장용
    tracks_l = [np.zeros(int(SAMPLE_RATE * video_len) + 500) for _ in range(4)]
    tracks_r = [np.zeros(int(SAMPLE_RATE * video_len) + 500) for _ in range(4)]
    midi_data = [[] for _ in range(4)] # [ (time, pitch, velocity), ... ]

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
            
            # 주파수 및 MIDI 노트 계산
            base_f = [65.4, 130.8, 261.6, 523.2][idx]
            freq = base_f + (area % 30)
            pitch = freq_to_midi(freq)
            velocity = int(np.clip((area / 5000) * 127, 40, 127)) # 면적을 MIDI 벨로시티로
            
            # 오디오 생성
            tone = generate_pro_wave(freq, UNIT_SEC, idx, 0.5)
            end_s = start_s + len(tone)
            if end_s < len(tracks_l[0]):
                tracks_l[idx][start_s:end_s] += tone * 0.5
                tracks_r[idx][start_s:end_s] += tone * 0.5
            
            # MIDI 데이터 기록 (박자 단위)
            midi_data[idx].append((u * 0.5, pitch, velocity)) # u * 0.5는 8분음표 박자 위치

        if u % 10 == 0: prog.progress(u / num_units)
    cap.release()

    # --- MIDI 파일 생성 ---
    midi_file = MIDIFile(4) # 4개 트랙
    for idx, track_notes in enumerate(midi_data):
        midi_file.addTempo(idx, 0, BPM)
        track_name = ["Bass", "Pluck", "Lead", "Bell"][idx]
        midi_file.addTrackName(idx, 0, track_name)
        
        for time, pitch, vel in track_notes:
            # duration을 0.5(8분음표)로 설정
            midi_file.addNote(idx, 0, pitch, time, 0.5, vel)

    midi_db = io.BytesIO()
    midi_file.writeFile(midi_db)

    # --- UI ---
    st.header("📂 Export to GarageBand")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. MIDI Export (강력 추천)")
        st.write("GarageBand에서 이 파일을 불러와서 **원하는 가상악기**를 입히세요.")
        st.download_button("💾 MIDI 파일 다운로드", midi_db.getvalue(), "optical_composition.mid")
        

    with col2:
        st.subheader("2. Master Audio")
        m_l, m_r = np.sum(tracks_l, axis=0), np.sum(tracks_r, axis=0)
        master = np.vstack((m_l, m_r)).T
        if np.max(np.abs(master)) > 0: master = (master / np.max(np.abs(master))) * 0.8
        m_io = io.BytesIO()
        with wave.open(m_io, 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE); wf.writeframes((master * 32767).astype(np.int16).tobytes())
        st.audio(m_io.getvalue())

    st.divider()
    st.subheader("📁 개별 트랙 오디오 모니터링")
    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            l_data = np.vstack((tracks_l[i], tracks_r[i])).T
            if np.max(np.abs(l_data)) > 0: l_data = (l_data / np.max(np.abs(l_data))) * 0.7
            l_io = io.BytesIO()
            with wave.open(l_io, 'wb') as wf:
                wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE); wf.writeframes((l_data * 32767).astype(np.int16).tobytes())
            st.caption(f"Track {i+1}")
            st.audio(l_io.getvalue())
