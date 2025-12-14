# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
import pandas as pd
import numpy as np
import cv2
from datetime import datetime

def box_center_xyxy(x1, y1, x2, y2):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

def inside_roi(cx, cy, roi):
    rx1, ry1, rx2, ry2 = roi
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

def draw_roi(frame_bgr, roi):
    rx1, ry1, rx2, ry2 = roi
    cv2.rectangle(frame_bgr, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
    cv2.putText(frame_bgr, "RESTRICTED ZONE", (rx1, max(ry1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def get_roi_pixels(w, h, roi_x1, roi_y1, roi_x2, roi_y2):
    rx1 = int(w * roi_x1 / 100)
    ry1 = int(h * roi_y1 / 100)
    rx2 = int(w * roi_x2 / 100)
    ry2 = int(h * roi_y2 / 100)
    return (min(rx1, rx2), min(ry1, ry2), max(rx1, rx2), max(ry1, ry2))

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection And Tracking using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

st.sidebar.header("🧩 New Features (A + B)")

# A) Restricted Zone
enable_roi = st.sidebar.checkbox("A) Restricted-zone alert", value=True)
watch_class = st.sidebar.text_input("Watch class (e.g., person)", value="person").strip().lower()

roi_x1 = st.sidebar.slider("ROI left (%)", 0, 95, 10)
roi_y1 = st.sidebar.slider("ROI top (%)", 0, 95, 10)
roi_x2 = st.sidebar.slider("ROI right (%)", 5, 100, 60)
roi_y2 = st.sidebar.slider("ROI bottom (%)", 5, 100, 60)

# B) Counting + Export
enable_count = st.sidebar.checkbox("B) Count objects per class", value=True)

if "roi_log" not in st.session_state:
    st.session_state.roi_log = []

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                result = res[0]
                boxes = result.boxes


                # YOLO renders boxes/masks to BGR image
                plotted_bgr = result.plot()
                h, w = plotted_bgr.shape[:2]

                # ---------- B) Counting ----------
                counts = {}
                if enable_count and result.boxes is not None and len(result.boxes) > 0:
                    names = result.names
                    for b in result.boxes:
                        cls_id = int(b.cls.item())
                        cls_name = str(names.get(cls_id, cls_id)).lower()
                        counts[cls_name] = counts.get(cls_name, 0) + 1

                # ---------- A) ROI Alert + logging ----------
                alert_triggered = False
                if enable_roi:
                    roi = get_roi_pixels(w, h, roi_x1, roi_y1, roi_x2, roi_y2)
                    draw_roi(plotted_bgr, roi)

                    if result.boxes is not None and len(result.boxes) > 0:
                        names = result.names
                        for b in result.boxes:
                            cls_id = int(b.cls.item())
                            cls_name = str(names.get(cls_id, cls_id)).lower()
                            conf = float(b.conf.item())

                            x1, y1, x2, y2 = b.xyxy[0].tolist()
                            cx, cy = box_center_xyxy(x1, y1, x2, y2)

                            if cls_name == watch_class and inside_roi(cx, cy, roi):
                                alert_triggered = True
                                st.session_state.roi_log.append({
                                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "class": cls_name,
                                    "confidence": round(conf, 3),
                                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
                                })

                    if alert_triggered:
                        cv2.putText(plotted_bgr, "ALERT: INTRUSION DETECTED!", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                # Show final image (convert BGR -> RGB)
                st.image(cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB),
                        caption="Result (A: ROI + B: Count)", use_container_width=True)

                # Show counts nicely
                if enable_count:
                    st.subheader("📊 Object Count (Feature B)")
                    if len(counts) == 0:
                        st.info("No objects detected.")
                    else:
                        df_count = pd.DataFrame([{"class": k, "count": v} for k, v in sorted(counts.items())])
                        st.dataframe(df_count, use_container_width=True)

                        st.download_button(
                            "Download counts CSV",
                            data=df_count.to_csv(index=False).encode("utf-8"),
                            file_name="object_counts.csv",
                            mime="text/csv"
                        )

                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")

st.subheader("📄 Intrusion Log (New Feature Output)")

if len(st.session_state.roi_log) == 0:
    st.info("No intrusion events logged yet.")
else:
    df = pd.DataFrame(st.session_state.roi_log)
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download intrusion log (CSV)",
        data=csv_bytes,
        file_name="intrusion_log.csv",
        mime="text/csv"
    )

if st.button("Clear log"):
    st.session_state.roi_log = []
    st.success("Log cleared.")
