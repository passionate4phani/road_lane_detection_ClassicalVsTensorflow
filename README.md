# Road Lane Detection for Autonomous Driving Assistance

An end-to-end project that detects road lanes from images, GIFs, and short videos via a **Streamlit** app.  
It implements and compares two approaches:

1. **Classical CV:** Canny Edge Detection + Probabilistic Hough Transform  
2. **Deep Learning:** U-Net segmentation (Tensorflow). If trained weights/model aren't provided, the app gracefully falls back to the classical mask.

## ✨ Features

- Upload **image / GIF / video** and run lane detection in-browser.
- **Choose approach**: Classical (Canny+Hough) vs Deep Learning (U-Net).
- Side-by-side visualization, with **download** of processed output.
- Includes **demo media** and generated screenshots/GIF in `outputs/`.

## 🗂 Project Structure

```
lane_detection_project/
├── README.md
├── requirements.txt
├── app.py
├── data/
│   ├── images/
│   ├── videos/
│   └── gifs/
├── outputs/
│   ├── classical/
│   └── deep_learning/
├── src/
│   ├── classical/
│   │   ├── preprocessing.py
│   │   ├── canny_hough.py
│   │   └── utils.py
│   ├── deep_learning/
│   │   ├── model.py
│   │   ├── inference.py
│   │   ├── train.py
│   │   ├── dataset.py
│   │   └── utils.py
│   └── visualization/
│       ├── plot_examples.py
│       └── make_gif.py
├── notebooks/
│   └── RoadLaneDetection.ipynb
└── models/
    └── best_model.keras
```

## 🚀 Quickstart

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the Streamlit app
streamlit run app.py
```


