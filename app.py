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


CLASS_COLORS = {
    "person": (255, 0, 0),    # blue in BGR
    "laptop": (0, 255, 0),    # green
    "bottle": (0, 255, 255),  # yellow
    "chair": (255, 0, 255),   # purple
}
DEFAULT_COLOR = (255, 255, 255)  # white fallback in BGR


def draw_filtered_boxes(img_rgb, result, selected_classes, min_conf=0.0):
    """
    img_rgb: numpy RGB image (H,W,3)
    result: YOLO result (res[0])
    selected_classes: list of class names to keep (lowercase)
    min_conf: confidence threshold to display
    """
    out = img_rgb.copy()
    names = result.names

    if result.boxes is None or len(result.boxes) == 0:
        return out

    for b in result.boxes:
        cls_id = int(b.cls.item())
        cls_name = str(names.get(cls_id, cls_id)).lower()
        conf = float(b.conf.item())

        if selected_classes and cls_name not in selected_classes:
            continue
        if conf < min_conf:
            continue

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # draw (OpenCV uses BGR, so convert quickly)
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.rectangle(out_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out_bgr,
            f"{cls_name} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    return out


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

    # Clear session state if new image uploaded
    if source_img is not None:
        if 'uploaded_file' not in st.session_state or st.session_state['uploaded_file'] != source_img.name:
            for key in ['detections_df', 'selected_classes', 'filtered_df', 'export_df', 'uploaded_image', 'result']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['uploaded_file'] = source_img.name

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         width="stretch")
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         width="stretch")
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     width="stretch")
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                result = res[0]
                boxes = result.boxes

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

                df_det = pd.DataFrame(detections)
                st.session_state['detections_df'] = df_det
                st.session_state['uploaded_image'] = uploaded_image
                st.session_state['result'] = result

    # Display detections if available
    if 'detections_df' in st.session_state:
        df_det = st.session_state['detections_df']
        if len(df_det) > 0:
            all_classes = sorted(df_det["class"].unique().tolist())
        else:
            all_classes = []

        if 'selected_classes' not in st.session_state:
            st.session_state['selected_classes'] = all_classes

        selected_classes = st.sidebar.multiselect(
            "Show only these classes",
            options=all_classes,
            default=st.session_state['selected_classes']
        )

        st.session_state['selected_classes'] = selected_classes

        if enable_filter and len(df_det) > 0 and len(selected_classes) > 0:
            df_view = df_det[df_det["class"].isin(selected_classes)].copy()
        else:
            df_view = df_det.iloc[0:0].copy()

        df_export = df_view[df_view["confidence"] >= min_conf_export].copy()

        st.session_state['filtered_df'] = df_view
        st.session_state['export_df'] = df_export

        img_rgb = np.array(st.session_state['uploaded_image'].convert("RGB"))
        filtered_img = draw_filtered_boxes(
            img_rgb,
            st.session_state['result'],
            [c.lower() for c in selected_classes] if enable_filter and selected_classes else [],
            min_conf_export
        )
        st.image(filtered_img, caption="Filtered Boxes View", width="stretch")

        st.info(f"Total detections: {len(df_det)} | After filter: {len(df_view)} | Export: {len(df_export)}")

        st.subheader("📋 Filtered Detections Table")
        st.dataframe(df_view, width="stretch")

        st.download_button(
            "Download detections CSV",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name="detections.csv",
            mime="text/csv"
        )

        with st.expander("Detection Results"):
            for index, row in df_view.iterrows():
                st.write(f"Class: {row['class']}, Confidence: {row['confidence']}, Box: ({row['x1']}, {row['y1']}, {row['x2']}, {row['y2']})")


elif source_radio == settings.VIDEO:
    # create video class filter using model.names
    all_classes = sorted([str(v).lower() for v in model.names.values()])
    video_selected = st.sidebar.multiselect(
        "Show only these classes (Video)",
        options=all_classes,
        default=all_classes
    )

    helper.play_stored_video(confidence, model, selected_classes=video_selected)
else:
    st.error("Please select a valid source type!")


