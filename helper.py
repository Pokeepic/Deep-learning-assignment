from ultralytics import YOLO
from helper import frames_to_gif_bytes
import streamlit as st
import cv2
import yt_dlp
import settings
import io
import imageio.v2 as imageio

CLASS_COLORS = {
    "person": (255, 0, 0),    # blue in BGR
    "laptop": (0, 255, 0),    # green
    "bottle": (0, 255, 255),  # yellow
    "chair": (255, 0, 255),   # purple
}
DEFAULT_COLOR = (255, 255, 255)  # white fallback in BGR


def draw_filtered_boxes(frame_bgr, result, selected_classes=None):
    """
    Draw only selected classes on a BGR frame.
    selected_classes: list[str] or None (None = draw all)
    """
    out = frame_bgr.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return out

    names = result.names  # dict id->name
    selected = set([c.lower() for c in selected_classes]) if selected_classes else None

    for b in result.boxes:
        cls_id = int(b.cls.item())
        cls_name = str(names.get(cls_id, cls_id)).lower()

        if selected is not None and cls_name not in selected:
            continue

        conf = float(b.conf.item())
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{cls_name} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    return out


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None, selected_classes=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    result0 = res[0]

    # draw ONLY selected classes (BGR)
    frame_drawn = draw_filtered_boxes(image, result0, selected_classes=selected_classes)

    st_frame.image(
        frame_drawn,
        caption="Detected Video",
        channels="BGR",
        width='stretch'
    )

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose a video...", list(settings.VIDEOS_DICT.keys()))
    video_path = str(settings.VIDEOS_DICT.get(source_vid))

    st.video(open(video_path, "rb").read())

    # --- GIF settings in sidebar ---
    st.sidebar.markdown("### 🎞️ Download as GIF")
    make_gif = st.sidebar.checkbox("Enable GIF export", value=True)
    gif_fps = st.sidebar.slider("GIF FPS", 2, 15, 8)
    sample_every = st.sidebar.slider("Sample every N frames", 1, 10, 2)
    max_gif_frames = st.sidebar.slider("Max GIF frames", 30, 300, 120)

    if st.sidebar.button("Detect Video Objects"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video.")
            return

        frame_i = 0
        gif_frames_rgb = []  # store RGB frames for GIF
        frame_placeholder = st.empty()

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_i += 1
            if frame_i % sample_every != 0:
                continue

            # YOLO prediction
            res = model.predict(frame_bgr, conf=conf, verbose=False)
            result = res[0]

            plotted_bgr = result.plot()  # BGR image with boxes/labels
            plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

            # show live preview
            frame_placeholder.image(plotted_rgb, caption="Detected frame", use_container_width=True)

            # collect for gif
            if make_gif and len(gif_frames_rgb) < max_gif_frames:
                gif_frames_rgb.append(plotted_rgb)

            # stop when enough frames for gif
            if make_gif and len(gif_frames_rgb) >= max_gif_frames:
                break

        cap.release()

        # Create download button
        if make_gif and len(gif_frames_rgb) > 0:
            gif_bytes = frames_to_gif_bytes(gif_frames_rgb, fps=gif_fps)
            st.download_button(
                "⬇️ Download Detected GIF",
                data=gif_bytes,
                file_name="detected_video.gif",
                mime="image/gif",
            )
        else:
            st.info("GIF export disabled or no frames collected.")

def frames_to_gif_bytes(frames_rgb, fps=8):
    """
    frames_rgb: list of numpy arrays in RGB format (H,W,3), dtype uint8
    returns: bytes of GIF
    """
    gif_buffer = io.BytesIO()
    duration = 1.0 / max(fps, 1)

    # imageio expects list of RGB images
    imageio.mimsave(gif_buffer, frames_rgb, format="GIF", duration=duration, loop=0)
    gif_buffer.seek(0)
    return gif_buffer.getvalue()