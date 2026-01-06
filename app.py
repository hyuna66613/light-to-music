import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile
import io
import os

# 페이지 설정
st.set_page_config(layout="wide", page_title="Light Orchestrator")
st.title("🚌 Night Bus Light-to-Music")

# --- 1. 사이드바 정보창 ---
with st.sidebar:
    st.header("📊 Video Info")
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=['mp4', 'mov', 'avi'])
    st.info("광원이 높을수록 고음, 낮을수록 저음이 생성됩니다.")

# --- 2. 메인 화면 레이아웃 ---
col_vid, col_snd = st.columns([1, 1])

if uploaded_file:
    try:
        # 임시 파일 저장 (안전한 방식)
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 24
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with st.sidebar:
            st.write(f"FPS: {fps}")
            st.write(f"Total Frames: {total_frames}")

        # 분석 설정 (성능을 위해 15프레임당 1박자 샘플링)
        step = 15 
        sample_rate = 44100
        audio_layers = {"Small": [], "Medium": [], "Large": []}
        
        st.write("✨ 광원을 분석하고 있습니다... 잠시만 기다려주세요.")
        progress_bar = st.progress(0)

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            # 빛 감지 로직
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            duration = (1.0 / fps) * step
            t = np.linspace(0, duration, int(sample_rate * duration), False)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 10: continue # 노이즈 제거
                
                # 중심점 찾기
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cy = int(M["m01"] / M["m00"])
                
                # 주파수 매핑 (밤 버스 느낌의 부드러운 사인파)
                freq = 880 - (cy * 1.2) 
                vol = min(area / 2000, 0.8)
                tone = vol * np.sin(2 * np.pi * freq * t)
                
                # 레이어 분류
                if area < 100: audio_layers["Small"].append(tone)
                elif area < 500: audio_layers["Medium"].append(tone)
                else: audio_layers["Large"].append(tone)
            
            progress_bar.progress(min(i / total_frames, 1.0))

        # 1번 창: 영상
        with col_vid:
            st.header("📽 Video View")
            st.video(uploaded_file)

        # 2번 창: 소리 레이어 (사용자 요청 반영)
        with col_snd:
            st.header("🎵 Sound Layers")
            
            combined_all = []
            
            for name, tones in audio_layers.items():
                if tones:
                    # 모든 음을 하나로 합침
                    layer_signal = np.concatenate(tones)
                    st.subheader(f"Layer: {name}")
                    
                    # 오디오 플레이어
                    st.audio(layer_signal, sample_rate=sample_rate)
                    
                    # 다운로드 버튼
                    buf = io.BytesIO()
                    wavfile.write(buf, sample_rate, (layer_signal * 32767).astype(np.int16))
                    st.download_button(label=f"Download {name} Layer", data=buf.getvalue(), file_name=f"{name}_layer.wav", mime="audio/wav")
                    
                    combined_all.append(layer_signal[:1000000]) # 믹스용 길이는 제한

            if combined_all:
                st.divider()
                st.button("🔥 Download All Layers (Mix Mode)")

        cap.release()
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
