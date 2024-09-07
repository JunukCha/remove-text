import cv2
import glob
import os
import os.path as osp
import tqdm
import torch
import argparse
from archs.mia_vsr_arch import MIAVSR
from basicsr.data.data_util import read_img_seq
from lib.miavsr.utils import read_video_seq, save_video_from_frames


def main(args):
    # -------------------- Configurations -------------------- #
    device = torch.device(args.device)
    model_path = args.model_path
    input_path = args.input_path
    save_folder = f'data'
    os.makedirs(save_folder, exist_ok=True)

    # set up the models
    model = MIAVSR( mid_channels=64,
                 embed_dim=120,
                 depths=[6,6,6,6],
                 num_heads=[6,6,6,6],
                 window_size=[3, 8, 8],
                 num_frames=3,
                 cpu_cache_length=100,
                 is_low_res_input=True,
                 use_mask=True,
                 spynet_path='checkpoints/spynet_sintel_final-3d2a1287.pth')
    model.load_state_dict(torch.load(model_path)['params'], strict=False)
    model.eval()
    model = model.to(device)

    if osp.isdir(input_path):
        imgs_lq, imgnames = read_img_seq(input_path, return_imgname=True)
        video_save_path = osp.join(input_path, "video_sr.mp4")
        fps = 25
    elif osp.isfile(input_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Error: Could not open video {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        imgs_lq, imgnames = read_video_seq(input_path, return_imgname=True)
        ext = osp.splitext(input_path)[1]
        video_save_path = input_path.replace(ext, f"_sr.mp4")

    # inference
    name_idx = 0
    imgs_lq = imgs_lq.unsqueeze(0).to(device)

    # Initialize an empty list to store output frames
    all_outputs = []

    # Assuming imgs_lq is a tensor of shape (N, C, H, W)
    # where N is the number of frames, you can process the video in chunks.
    chunk_size = 10  # Number of frames to process at a time

    # Process the video in chunks
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, imgs_lq.shape[1], chunk_size), desc="Run SR"):
            chunk = imgs_lq[:, i:i+chunk_size].to('cuda')
            outputs, _ = model(chunk)
            outputs = outputs.squeeze(0).cpu()  # Move the output to CPU
            all_outputs.append(outputs)
            torch.cuda.empty_cache()  # Clear the cache to free up GPU memory
    # Concatenate all output chunks along the time dimension (0 axis)
    final_outputs = torch.cat(all_outputs, dim=0)

    # Save the concatenated output frames as a video
    save_video_from_frames(final_outputs, video_save_path, fps=fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MIAVSR inference.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference (e.g., cuda, cpu).')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--input_path', type=str, required=True, help='Folder containing low-resolution input images.')
    
    args = parser.parse_args()
    main(args)
