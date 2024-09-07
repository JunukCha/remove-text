import cv2
import glob
import os
import os.path as osp
import torch
import numpy as np
from basicsr.utils import img2tensor, tensor2img
from basicsr.data.transforms import mod_crop


def save_video_from_frames(frames, save_path, fps=25):
    """
    Save a sequence of frames as a video.

    Args:
        frames (torch.Tensor): Tensor of frames, size (t, c, h, w), RGB, [0, 1].
        save_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.
    """
    # Convert the tensor to numpy images
    frames_np = [tensor2img(frame, rgb2bgr=True, min_max=(0, 1)) for frame in frames]
    
    # Get height, width from the first frame
    h, w = frames_np[0].shape[:2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    
    for frame in frames_np:
        out.write(frame)
    
    out.release()
    print(f"Video saved at {save_path}")

def read_video_seq(video_path, require_mod_crop=False, scale=1, return_imgname=False):
    """Read a sequence of frames from a video file.

    Args:
        video_path (str): Path to the video file.
        require_mod_crop (bool): Require mod crop for each frame.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname (bool): Whether to return frame names (timestamps). Default False.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
        list[str]: Returned frame name list (timestamps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video {video_path}")

    frames = []
    timestamps = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame.astype(np.float32) / 255.0
        if require_mod_crop:
            frame = mod_crop(frame, scale)
        frames.append(frame)
        if return_imgname:
            timestamps.append(f"frame_{frame_idx:04d}")
        frame_idx += 1

    cap.release()

    frames = img2tensor(frames, bgr2rgb=True, float32=True)
    frames = torch.stack(frames, dim=0)

    if return_imgname:
        return frames, timestamps
    else:
        return frames