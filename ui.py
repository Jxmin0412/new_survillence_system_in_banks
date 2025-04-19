# ui.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from anu_test import get_processed_frame

# Streamlit UI setup
st.set_page_config(layout="centered")
st.title("🔐 Smart Surveillance System – Bank Prototype")

FRAME_PLACEHOLDER = st.empty()

# Status indicators
fire_status = st.empty()
weapon_status = st.empty()
face_status = st.empty()

# Start video processing loop
run = st.checkbox("Start Surveillance", value=True)

if run:
    video_stream = get_processed_frame()

    for frame, fire_active in video_stream:
        # Show fire detection status
        if fire_active:
            fire_status.markdown(
                "🔥 **Fire Detection: ACTIVE**", unsafe_allow_html=True
            )
        else:
            fire_status.markdown("🔥 Fire Detection: Inactive", unsafe_allow_html=True)

        # Placeholder statuses (not yet implemented)
        weapon_status.markdown("🔫 Weapon Detection: Inactive")
        face_status.markdown("🧑 Face Recognition: Inactive")

        # Convert BGR to RGB and display
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_PLACEHOLDER.image(
            Image.fromarray(img),
            channels="RGB",
            caption="Surveillance Feed",
            use_container_width=True,
        )

        # ~33 FPS delay
        time.sleep(1 / 30)
