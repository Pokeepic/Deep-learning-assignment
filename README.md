# 🎯 YOLOv8 Object Detection with Class Filtering & Export (Streamlit App)

This project implements an **object detection web application using YOLOv8 and Streamlit**.  
It supports **image and video detection**, **class-based filtering**, **confidence thresholding**, and **CSV export** of detected objects.

The goal of this project is to demonstrate practical usage of deep learning–based object detection with an interactive user interface.

---

## 🔍 Features

- ✅ YOLOv8 object detection
- ✅ Image and video input support
- ✅ Adjustable confidence threshold
- ✅ Class-based filtering (applies to image & video)
- ✅ Filtered bounding boxes visualization
- ✅ CSV export of filtered detections
- ✅ Simple, clean Streamlit UI

---

## 🖥️ Application Overview

### Main Interface
- Upload an **image** or select a **video**
- Adjust model confidence
- Select classes to display
- Export filtered detections as CSV

### Example Outputs
- Original image/video
- Filtered detection results
- Detection table (class, confidence, bounding box coordinates)

---

## 🧠 Model Used

- **YOLOv8 (Ultralytics)**
- Pre-trained on **COCO dataset**
- Supports 80 object classes

---

## 📦 Requirements

- Python **3.8+**
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas

### Install dependencies
```bash
pip install ultralytics streamlit opencv-python numpy pandas yt_dlp
