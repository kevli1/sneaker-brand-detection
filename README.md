# Sneaker Brand Detection in Marathon Videos

## 📌 Project Overview
This project uses computer vision techniques to identify sneaker brands worn by runners in marathon videos. The system processes video frames to detect shoes, classify their brand logos, and track individuals across frames to generate an aggregate summary of brand presence.

## 🎯 What It Does
- Takes in raw marathon video footage as input.
- Extracts frames and detects sneakers using an object detection model.
- Classifies sneaker brands using a custom-trained image classifier.
- Tracks individuals using object tracking algorithms to avoid duplicate counts.
- Outputs a CSV file with frame-by-frame brand detections and summary visualizations.

## 🛠️ Project Structure
```
sneaker-brand-detection/
├── scripts/                  # Core scripts for inference and tracking
│   ├── nike_marathon_inference.py
│   ├── object_tracker_v5.py
│   └── retinanet_video.py
├── data/                     # Sample output data (CSV only)
│   └── IMG_0007_shoes_final.csv
├── model_results/            # Detection visual results & dashboard
│   ├── IMG_0007_keyframe.jpg
│   └── dashboard.png
├── models/                   # Model structure (prototxt)
│   └── MobileNetSSD_deploy.prototxt
└── README.md

```

## 🧠 Model Training & Improvement
The brand classifier was trained using custom-labeled sneaker images. One key challenge was identifying shoes with brand logos that had low contrast with the shoe color (e.g., black-on-black or white-on-white).

To address this, we performed error analysis on our test predictions and discovered that these low-contrast examples were consistently misclassified. We then sourced and added more of these challenging examples into the training dataset, which led to a significant performance improvement in those specific cases.

## 📊 Sample Output
- **CSV Output**: Frame-by-frame brand detection results (see `data/`)
- **Image Results**: Annotated detection examples (see `model_results/IMG_0007_keyframe.jpg`)
- **Dashboard**: Summary of brand distribution and athlete breakdown (see `model_results/dashboard.png`)
- *(Optional)* Output videos with detection overlays are omitted here for brevity.

## 💻 Tech Stack
- Python, OpenCV
- Custom CNN classifier
- Object detection (RetinaNet, MobileNet-SSD)
- Person tracking
- Power BI for final visualisation and summary analysis

## 🔒 Note
This project was originally developed in 2021. All data presented here is either mock or anonymized for demonstration purposes.
