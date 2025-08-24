# Road Lane Detection for Autonomous Driving Assistance

An end-to-end project that detects road lanes from images, GIFs, and short videos via a **Streamlit** app.  
It implements and compares two approaches:

1. **Classical CV:** Canny Edge Detection + Probabilistic Hough Transform  
2. **Deep Learning:** U-Net segmentation (PyTorch). If weights aren't provided, the app gracefully falls back to the classical mask.

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
    └── RoadLaneDetection.ipynb
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

Then open the browser tab that Streamlit prints, upload a small **image/GIF/video**, and pick **Classical** or **Deep Learning**.

> **Note:** For U-Net, place your trained weights as `models/unet_lane.pth`. If absent, the pipeline will fallback to classical edges to form a segmentation mask for visualization so that the comparison page still works for demo purposes.

## 🧠 Training (Optional)

- Use `src/deep_learning/train.py` with a lane dataset such as **TuSimple**.  
- Configure dataset roots in `dataset.py`.  
- The script saves checkpoints into `models/checkpoints/` and a final `models/unet_lane.pth`.

## 📊 Demo GIFs / Screenshots

Below are example outputs included in this repository (generated from synthetic demo media):

- `outputs/classical/demo_image_overlay.png`
- `outputs/classical/demo_video_classical.gif`
- `outputs/deep_learning/demo_video_unet.gif` *(will mirror classical edges unless you provide weights)*

![Classical Overlay](outputs/classical/demo_image_overlay.png)

## 🛣 Future Improvements

- Lane curvature & radius estimation
- Robust perspective transform + polynomial fitting
- Lane tracking with temporal smoothing (Kalman/EMA)
- On-road inference with webcam/RTSP support

---

**Author:** You 💚 — built for internship showcase.  
If this helps you, consider adding a ⭐ in your repo!
