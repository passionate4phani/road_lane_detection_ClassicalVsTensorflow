import streamlit as st
import numpy as np
import cv2
import tempfile
from PIL import Image, ImageSequence
from pathlib import Path
from src.classical.canny_hough import process_frame_classical
from src.deep_learning.inference import load_keras, segment_frame_keras
import imageio.v2 as imageio

st.set_page_config(page_title="Lane Detection", layout="wide")

st.title("🛣️ Road Lane Detection — Classical vs Tensorflow")

with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Choose Method", ["Classical (Canny + Hough)", "Deep Learning (Tensorflow)"])
    show_edges = st.checkbox("Show edges mask (debug)", False)
    st.markdown("---")
    st.caption("Upload a small image/GIF/video. Processing happens locally.")

uploaded = st.file_uploader("Upload image / GIF / short video", type=["png","jpg","jpeg","bmp","gif","mp4","avi","mov","mkv"])

col1, col2 = st.columns(2)

def bytes_to_tempfile(file, suffix):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(file.getbuffer())
    tfile.flush()
    return tfile.name

@st.cache_resource
def _load_keras_once():
    return load_keras(weights_path="models/best_model.keras")

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix in [".png",".jpg",".jpeg",".bmp"]:
        # Image path
        image = Image.open(uploaded).convert("RGB")
        frame = np.array(image)[:, :, ::-1]  # to BGR
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)
        if method.startswith("Classical"):
            overlay, dbg = process_frame_classical(frame, return_debug=True)
            res_rgb = overlay[:, :, ::-1]
            with col2:
                st.subheader("Detected Lanes")
                st.image(res_rgb, use_container_width=True)
                if show_edges and dbg is not None:
                    st.image(dbg, caption="Edges (debug)", use_container_width=True)
            outname = "output_classical.png"
            _, buf = cv2.imencode(".png", res_rgb[:, :, ::-1])
            st.download_button("Download output", data=buf.tobytes(), file_name=outname, mime="image/png")
        else:
            with st.spinner("Running Keras inference... ⏳"):
                keras = _load_keras_once()
                overlay, mask = segment_frame_keras(keras, frame)
            res_rgb = overlay[:, :, ::-1]
            with col2:
                st.subheader("Detected Lanes (Keras)")
                st.image(res_rgb, use_container_width=True)
                if show_edges and mask is not None:
                    st.image(mask, caption="Mask (debug)", use_container_width=True)
            outname = "output_unet.png"
            _, buf = cv2.imencode(".png", res_rgb[:, :, ::-1])
            st.download_button("Download output", data=buf.tobytes(), file_name=outname, mime="image/png")

    elif suffix == ".gif":
        gif_path = bytes_to_tempfile(uploaded, ".gif")
        frames = [np.array(f.convert("RGB"))[:, :, ::-1] for f in ImageSequence.Iterator(Image.open(gif_path))]
        processed = []
        if method.startswith("Classical"):
            for f in frames:
                ov, _ = process_frame_classical(f, return_debug=False)
                processed.append(ov[:, :, ::-1])  # to RGB for gif
        else:
            keras = _load_keras_once()
            progress = st.progress(0)
            for i, f in enumerate(frames):
                ov, _ = segment_frame_keras(keras, f)
                processed.append(ov[:, :, ::-1])
                progress.progress((i + 1) / len(frames))
            progress.empty()
        with col1:
            st.subheader("Original GIF")
            st.image(uploaded, use_container_width=True)
        with col2:
            st.subheader(f"Detected Lanes ({method})")
            st.image(processed[0], use_container_width=True)
        out_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
        imageio.mimsave(out_gif, processed, fps=10)
        with open(out_gif, "rb") as f:
            st.download_button("Download processed GIF", f, file_name=f"output_{method.lower().split()[0]}.gif", mime="image/gif")

    else:
        # Video case
        video_path = bytes_to_tempfile(uploaded, suffix)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not read the uploaded video.")
        else:
            frames = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            cap.release()

            processed = []
            if method.startswith("Classical"):
                for f in frames:
                    ov, _ = process_frame_classical(f, return_debug=False)
                    processed.append(ov)
            else:
                keras = _load_keras_once()
                progress = st.progress(0)
                for i, f in enumerate(frames):
                    ov, _ = segment_frame_keras(keras, f)
                    processed.append(ov)
                    progress.progress((i + 1) / len(frames))
                progress.empty()

            with col1:
                st.subheader("Original Video (first frame)")
                st.image(frames[0][:,:,::-1], use_container_width=True)
            with col2:
                st.subheader(f"Detected Lanes ({method}, first frame)")
                st.image(processed[0][:,:,::-1], use_container_width=True)

            out_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
            imageio.mimsave(out_gif, [p[:,:,::-1] for p in processed], fps=15)
            with open(out_gif, "rb") as f:
                st.download_button("Download processed GIF", f, file_name=f"output_{method.lower().split()[0]}.gif", mime="image/gif")
else:
    st.info("Upload an image, GIF, or a short video to get started.")