import streamlit as st
import cv2
import numpy as np
from scipy.io import wavfile
import io
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Musical Light DAW")
st.title("🎹 Musical Light: Harmonic Synth DAW")

# --- 음악적 설정: 마이너 펜타토닉 스케일 (밤의 몽환적인 느낌) ---
# 도(C), 미b(Eb), 파(F), 솔(G), 시b(Bb) 주파수 리스트
NOTES = [130.81, 155.56, 174.61, 196.00, 233.08, 
         261.63, 311.13, 349.23, 392.00, 466.16, 
         523.25, 622.25, 698.46, 783.99, 932.33]

def get_nearest_note(freq):
    return min(NOTES, key=lambda x: abs(x - freq))

def apply_envelope(tone, sample_rate):
    # 부드러운 시작(Attack)과 끝(Release) 처리
    n = len(tone)
    attack = int(sample_rate * 0.01)
    release = int(sample_rate * 0.05)
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return tone * env

with st.sidebar:
    st.header("🎛 Synth Engine")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    harmony_mode = st.select_slider("Harmony Style", options=["Deep", "Dreamy", "Sharp"])

if uploaded_file:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        sample_rate = 22050 
        max_tracks = 6
        tracks_audio_l = [[] for _ in range(max_tracks)] # 왼쪽 채널
        tracks_audio_r = [[] for _ in range(max_tracks)] # 오른쪽 채널
        tracks_visual = [[] for _ in range(max_tracks)]
        
        prog = st.progress(0)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            duration = 1.0 / fps
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_tracks]
            
            for idx, cnt in enumerate(sorted_contours):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                # 1. 주파수를 음계(Scale)에 맞춤
                raw_freq = 150 + ((frame.shape[0] - cy) * 1.5)
                note_freq = get_nearest_note(raw_freq)
                
                # 2. 배음 추가 (진짜 악기처럼 들리게 함)
                vol = min(area / 1500, 0.6)
                tone = vol * np.sin(2 * np.pi * note_freq * t) # 기본음
                tone += (vol * 0.3) * np.sin(2 * np.pi * (note_freq * 2) * t) # 옥타브 배음
                
                # 3. 부드러운 ADSR 적용
                tone = apply_envelope(tone, sample_rate)
                
                # 4. 팬닝(Panning): x좌표에 따른 입체 음향
                pan_r = cx / frame.shape[1]
                pan_l = 1 - pan_r
                
                tracks_audio_l[idx].append(tone * pan_l)
                tracks_audio_r[idx].append(tone * pan_r)
                tracks_visual[idx].append({'time': i/fps, 'freq': note_freq})
            
            for j in range(len(sorted_contours), max_tracks):
                tracks_audio_l[j].append(np.zeros_like(t))
                tracks_audio_r[j].append(np.zeros_like(t))
                tracks_visual[j].append({'time': i/fps, 'freq': 0})
            
            if i % 30 == 0: prog.progress(i / total_frames)

        # 믹싱 (스테레오)
        master_l = np.sum([np.concatenate(t) for t in tracks_audio_l], axis=0)
        master_r = np.sum([np.concatenate(t) for t in tracks_audio_r], axis=0)
        
        # 노멀라이징 및 스테레오 합치기
        master_stereo = np.vstack((master_l, master_r)).T
        master_stereo = (master_stereo / np.max(np.abs(master_stereo)) * 32767).astype(np.int16)

        # UI 출력 (생략된 부분은 이전과 동일)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.video(uploaded_file)
            st.audio(master_stereo, sample_rate=sample_rate)
        
        with col2:
            st.header("📊 Harmonic Timeline")
            # 주파수 그래프 시각화 (코드 동일)
            # ... (이전 시각화 코드와 동일하게 적용)

    except Exception as e:
        st.error(f"오류: {e}")
