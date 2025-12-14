from ultralytics import YOLO
import streamlit as st
import cv2
import random
import yt_dlp
import settings


CLASS_COLORS = {
    "person": (255, 0, 0),    # blue in BGR
    "laptop": (0, 255, 0),    # green
    "bottle": (0, 255, 255),  # yellow
    "chair": (255, 0, 255),   # purple
}
DEFAULT_COLOR = (255, 255, 255)  # white fallback in BGR


def color_for_class(name):
    random.seed(hash(name) % 2**32)
    return tuple(random.randint(50, 255) for _ in range(3))


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
        color = color_for_class(cls_name)
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


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None, selected_classes=None, min_conf=0.0):
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

    # Draw filtered boxes
    result = res[0]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    selected_classes_lower = [c.lower() for c in selected_classes] if selected_classes else []
    filtered_img = draw_filtered_boxes(img_rgb, result, selected_classes_lower, min_conf)
    st_frame.image(filtered_img,
                   caption='Detected Video',
                   channels="RGB",
                   width="stretch"
                   )

def play_stored_video(conf, model, selected_classes=None, min_conf_export=0.0):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             selected_classes,
                                             min_conf_export
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
