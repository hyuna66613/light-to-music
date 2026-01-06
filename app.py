import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile
import io
import base64

# --- 1. 소리 생성 함수 (수학적 연산) ---
def generate_tone(frequency, duration, volume=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 사인파 생성 (빛의 청각화)
    tone = volume * np.sin(2 * np.pi * frequency * t)
    return tone

# --- 2. 페이지 설정 ---
st.set_page_config(layout="wide", page_title="Light Orchestrator")
st.title("🚌 Night Bus Light-to-Music")

# --- 3. 사이드바 (정보창 대체) ---
with st.sidebar:
    st.header("📊 Video Info")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.info("광원 위치가 높을수록 고음, 낮을수록 저음이 생성됩니다.")

# --- 4. 메인 화면 (3개 구역) ---
col_vid, col_snd = st.columns([1, 1])

if uploaded_file:
    # 임시 파일로 영상 읽기
    g = io.BytesIO(uploaded_file.read())
    with open("temp_video.mp4", "wb") as f:
        f.write(g.read())
    
    cap = cv2.VideoCapture("temp_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 정보창 업데이트
    with st.sidebar:
        st.write(f"FPS: {fps}")
        st.write(f"Total Frames: {total_frames}")

    # 소리 데이터 저장을 위한 딕셔너리
    # 레이어: Small(고음), Medium(중음), Large(저음)
    audio_layers = {"Small": [], "Medium": [], "Large": []}
    
    # 분석 프로세스 (샘플링: 10프레임당 1박자)
    progress_bar = st.progress(0)
    step = 10 
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        
        # 광원 추출 로직
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY) # 밝은 부분만 남기기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5: continue # 너무 작은 노이즈 제거
            
            # 중심점 찾기
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cy = int(M["m01"] / M["m00"])
            
            # 높이(cy)를 주파수로 변환 (위쪽이 고음)
            freq = 1000 - (cy * 1.5) # 간단한 매핑 공식
            duration = (1/fps) * step
            
            tone = generate_tone(freq, duration, volume=min(area/1000, 1.0))
            
            # 크기에 따라 레이어 분류
            if area < 50: audio_layers["Small"].append(tone)
            elif area < 200: audio_layers["Medium"].append(tone)
            else: audio_layers["Large"].append(tone)
        
        progress_bar.progress(i / total_frames)

    # 1번 창: 영상 재생
    with col_vid:
        st.header("📽 Video View")
        st.video(uploaded_file)

    # 2번 창: 소리 레이어 및 다운로드
    with col_snd:
        st.header("🎵 Sound Layers")
        
        final_audio_all = []
        for name, tones in audio_layers.items():
            if tones:
                layer_data = np.concatenate(tones)
                st.subheader(f"Layer: {name}")
                st.audio(layer_data, sample_rate=44100)
                
                # 다운로드 버튼 생성 (WAV 변환)
                buffer = io.BytesIO()
                wavfile.write(buffer, 44100, (layer_data * 32767).astype(np.int16))
                st.download_button(f"Download {name} Layer", buffer, f"{name}.wav")
                final_audio_all.append(layer_data)

        if final_audio_all:
            st.divider()
            st.button("🔥 Download All Layers (Mix)")

    cap.release()
