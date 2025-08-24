import imageio.v2 as imageio
import glob
import numpy as np

def frames_to_gif(frames_glob, out_path="out.gif", fps=10):
    files = sorted(glob.glob(frames_glob))
    frames = [imageio.imread(f) for f in files]
    imageio.mimsave(out_path, frames, fps=fps)
    return out_path
