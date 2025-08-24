import streamlit as st
import numpy as np
import cv2
import tempfile
from PIL import Image, ImageSequence
from pathlib import Path
from src.classical.canny_hough import process_frame_classical
from src.deep_learning.inference import load_unet, segment_frame_unet
import imageio.v2 as imageio

st.set_page_config(page_title="Lane Detection", layout="wide")

st.title("🛣️ Road Lane Detection — Classical vs U-Net")

with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Choose Method", ["Classical (Canny + Hough)", "Deep Learning (U-Net)"])
    show_edges = st.checkbox("Show edges mask (debug)", False)
    st.markdown("---")
    st.caption("Upload a small image/GIF/video. Processing happens locally.")

uploaded = st.file_uploader("Upload image / GIF / short video", type=["png","jpg","jpeg","bmp","gif","mp4","avi","mov","mkv"])

col1, col2 = st.columns(2)
raw_bytes = None

def bytes_to_tempfile(file, suffix):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(file.getbuffer())
    tfile.flush()
    return tfile.name

@st.cache_resource
def _load_unet_once():
    return load_unet(weights_path="models/unet_lane.pth")

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
            # download
            outname = "output_classical.png"
            _, buf = cv2.imencode(".png", res_rgb[:, :, ::-1])
            st.download_button("Download output", data=buf.tobytes(), file_name=outname, mime="image/png")
        else:
            unet = _load_unet_once()
            overlay, mask = segment_frame_unet(unet, frame)
            res_rgb = overlay[:, :, ::-1]
            with col2:
                st.subheader("Detected Lanes (U-Net)")
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
            with col1:
                st.subheader("Original GIF")
                st.image(uploaded, use_container_width=True)
            with col2:
                st.subheader("Detected Lanes")
                st.image(processed[0], use_container_width=True)
            out_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
            imageio.mimsave(out_gif, processed, fps=10)
            with open(out_gif, "rb") as f:
                st.download_button("Download processed GIF", f, file_name="output_classical.gif", mime="image/gif")
        else:
            unet = _load_unet_once()
            for f in frames:
                ov, _ = segment_frame_unet(unet, f)
                processed.append(ov[:, :, ::-1])
            with col1:
                st.subheader("Original GIF")
                st.image(uploaded, use_container_width=True)
            with col2:
                st.subheader("Detected Lanes (U-Net)")
                st.image(processed[0], use_container_width=True)
            out_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
            imageio.mimsave(out_gif, processed, fps=10)
            with open(out_gif, "rb") as f:
                st.download_button("Download processed GIF", f, file_name="output_unet.gif", mime="image/gif")

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
                with col1:
                    st.subheader("Original Video (first frame)")
                    st.image(frames[0][:,:,::-1], use_container_width=True)
                with col2:
                    st.subheader("Detected Lanes (first frame)")
                    st.image(processed[0][:,:,::-1], use_container_width=True)

                out_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
                imageio.mimsave(out_gif, [p[:,:,::-1] for p in processed], fps=15)
                with open(out_gif, "rb") as f:
                    st.download_button("Download processed GIF", f, file_name="output_classical.gif", mime="image/gif")
            else:
                unet = _load_unet_once()
                for f in frames:
                    ov, _ = segment_frame_unet(unet, f)
                    processed.append(ov)
                with col1:
                    st.subheader("Original Video (first frame)")
                    st.image(frames[0][:,:,::-1], use_container_width=True)
                with col2:
                    st.subheader("Detected Lanes (U-Net, first frame)")
                    st.image(processed[0][:,:,::-1], use_container_width=True)
                out_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
                imageio.mimsave(out_gif, [p[:,:,::-1] for p in processed], fps=15)
                with open(out_gif, "rb") as f:
                    st.download_button("Download processed GIF", f, file_name="output_unet.gif", mime="image/gif")
else:
    st.info("Upload an image, GIF, or a short video to get started.")
