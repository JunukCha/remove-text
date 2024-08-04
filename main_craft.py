import os, os.path as osp
import argparse
import cv2
import numpy as np
import tqdm
import torch
from torch.autograd import Variable
from collections import OrderedDict

import sys
import random
sys.path.append(osp.join(osp.dirname(__file__), "CRAFT-pytorch"))

# Importing CRAFT model related modules
from craft import CRAFT
from craft_utils import getDetBoxes, adjustResultCoordinates
from imgproc import resize_aspect_ratio, normalizeMeanVariance

def load_craft_model(model_path='craft_mlt_25k.pth'):
    """Load the CRAFT text detection model."""
    net = CRAFT()
    state_dict = torch.load(model_path, map_location='cpu')

    # Remove 'module.' prefix if it exists
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.'
        else:
            new_state_dict[k] = v
    
    net.load_state_dict(new_state_dict)
    net.eval()
    return net

def detect_text_boxes(image, net):
    """Detect text regions in the image and return bounding box coordinates."""
    # Image preprocessing
    image_resized, target_ratio, _ = resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio
    x = normalizeMeanVariance(image_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    x = Variable(x)

    # Text region detection
    with torch.no_grad():
        y, _ = net(x)
    
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Extract text boxes
    boxes, _ = getDetBoxes(score_text, score_link, 0.7, 0.4, 0.4, True)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    
    return boxes

def enlarge_random_boxes(boxes, enlarge_ratio=1, enlarge_factor=1.2):
    """Randomly select 10% of the boxes and enlarge them."""
    num_boxes_to_enlarge = int(len(boxes) * enlarge_ratio)
    indices_to_enlarge = random.sample(range(len(boxes)), num_boxes_to_enlarge)
    
    for idx in indices_to_enlarge:
        box = np.array(boxes[idx])
        center = box.mean(axis=0)
        new_box = (box - center) * enlarge_factor + center
        boxes[idx] = new_box.tolist()
    
    return boxes

def create_box_mask(image, boxes):
    """Create a mask based on the detected boxes."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for box in boxes:
        box = np.array(box, dtype=np.int32)
        cv2.fillPoly(mask, [box], 255)
    return mask

def plot_mask_on_image(image, mask):
    """Overlay the mask on the original image."""
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored[:, :, 1:] = 0  # Set mask color to red
    overlay = cv2.addWeighted(image, 1.0, mask_colored, 0.5, 0)
    return overlay

def save_mask(mask, output_dir, frame_index):
    """Save the mask as a PNG file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mask_path = os.path.join(output_dir, f"{frame_index:05d}.png")
    cv2.imwrite(mask_path, mask)

def process_video(video_path):
    """Process the video, detect and mask text regions, and save the results."""
    cap = cv2.VideoCapture(video_path)
    craft_net = load_craft_model()
    
    frame_index = 0

    base_name = osp.basename(video_path)
    file_name, file_ext = os.path.splitext(base_name)
    save_folder = osp.join("data", file_name)
    os.makedirs(save_folder, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_width, output_height = 432, 240

    for _ in tqdm.tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect text regions
        frame = cv2.resize(frame, (output_width, output_height))
        boxes = detect_text_boxes(frame, craft_net)

        # Randomly enlarge 10% of the boxes
        boxes = enlarge_random_boxes(boxes)

        # Create mask
        mask = create_box_mask(frame, boxes)
        
        # Save mask
        save_mask(mask, save_folder, frame_index)
        
        frame_index += 1

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Path to the input video file.")
    args = parser.parse_args()
    process_video(args.video_path)
