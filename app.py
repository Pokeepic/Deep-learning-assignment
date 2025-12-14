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


st.sidebar.header("✨ New Feature: Class Filter + Export")

enable_filter = st.sidebar.checkbox("Enable class filter", value=True)
min_conf_export = st.sidebar.slider("Min confidence for export", 0.05, 0.95, float(confidence), 0.05)


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

                # 1) Build detections list first
                detections = []
                if boxes is not None and len(boxes) > 0:
                    names = result.names
                    for b in boxes:
                        cls_id = int(b.cls.item())
                        cls_name = str(names.get(cls_id, cls_id))
                        conf = float(b.conf.item())
                        x1, y1, x2, y2 = b.xyxy[0].tolist()

                        detections.append({
                            "class": cls_name,
                            "confidence": round(conf, 3),
                            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
                        })

                # 2) Make dataframe
                df_det = pd.DataFrame(detections)

                # 3) Multiselect classes (based on detections)
                if len(df_det) > 0:
                    all_classes = sorted(df_det["class"].unique().tolist())
                else:
                    all_classes = []

                selected_classes = st.sidebar.multiselect(
                    "Show only these classes",
                    options=all_classes,
                    default=all_classes
                )

                # 4) Filter dataframe
                if enable_filter and len(df_det) > 0 and len(selected_classes) > 0:
                    df_view = df_det[df_det["class"].isin(selected_classes)].copy()
                else:
                    df_view = df_det.iloc[0:0].copy()

                df_export = df_view[df_view["confidence"] >= min_conf_export].copy()

                # 5) Plot & show image (still shows all boxes; ok for now)
                plotted_bgr = result.plot()
                st.image(cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.info(f"Total detections: {len(df_det)} | After filter: {len(df_view)} | Export: {len(df_export)}")

                # 6) Show table + export
                st.subheader("📋 Filtered Detections Table")
                st.dataframe(df_view, use_container_width=True)

                st.download_button(
                    "Download detections CSV",
                    data=df_export.to_csv(index=False).encode("utf-8"),
                    file_name="detections.csv",
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


