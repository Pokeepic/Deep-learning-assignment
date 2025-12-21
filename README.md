# YOLOv8 Object Detection with Class Filtering & Export (Streamlit App)

This project implements an **object detection web application using YOLOv8 and Streamlit**.  
It supports **image and video detection**, **class-based filtering**, **confidence thresholding**, and **CSV export** of detected objects.

The goal of this project is to demonstrate practical usage of deep learning–based object detection with an interactive user interface.

---

## Features

- ✅ YOLOv8 object detection
- ✅ Image and video input support
- ✅ Adjustable confidence threshold
- ✅ Class-based filtering (applies to image & video)
- ✅ Filtered bounding boxes visualization
- ✅ CSV export of filtered detections
- ✅ Simple, clean Streamlit UI

---

## Application Overview

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

## Model Used

- **YOLOv8 (Ultralytics)**
- Pre-trained on **COCO dataset**
- Supports 80 object classes

---

## Requirements

- Python **3.8+**
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas

### Install dependencies
```bash
pip install ultralytics streamlit opencv-python numpy pandas yt_dlp imageio
```

### Media & Dataset Credits

This repository includes images and videos obtained from publicly available
Kaggle datasets. All rights and licenses remain with the original creators.

---

4. **Video Project6**
   - Platform: Kaggle
   - Dataset Author: mahsa sanaei
   - Dataset URL: https://www.kaggle.com/datasets/snmahsa/driving-test

5. **Traffic Images of Vehicles**
   - Platform: Kaggle
   - Dataset Author: mdshahriyarhossain
   - Dataset URL: https://www.kaggle.com/datasets/therealshihab/traffic-detection-for-yolov5

---

## License Compliance Notes## Kaggle Datasets

1. **Video Project1 - Video Project3**
   - Platform: Kaggle
   - Dataset Author: GhazanfarLatif
   - Dataset URL: https://www.kaggle.com/competitions/pmu-cai-competition2024/data

2. **Video Project4**
   - Platform: Kaggle
   - Dataset Author: Geir Drange
   - Dataset URL: https://www.kaggle.com/datasets/mistag/short-videos

3. **Video Project5**
   - Platform: Kaggle
   - Dataset Author: Abhay Swaroop
   - Dataset URL: https://www.kaggle.com/datasets/deeplyft/driving-video-subset-50-with-object-tracking


- Assets are used strictly under the terms specified on each Kaggle dataset page.
- Attribution is provided where required by the dataset license.
- This project does not claim ownership of third-party datasets or media.
