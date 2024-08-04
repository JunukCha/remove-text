import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def merge_videos(video_path, mask_dir):
    """Merge the original video, mask images, and result video side by side."""
    cap = cv2.VideoCapture(video_path)
    
    # Generate result video path by appending '_result' to the video file name
    base_name = os.path.basename(video_path)
    file_name, file_ext = os.path.splitext(base_name)
    result_path = os.path.join(os.path.dirname(video_path), f"{file_name}_result{file_ext}")
    
    if not os.path.exists(result_path):
        print(f"Result video file {result_path} does not exist.")
        return

    result_cap = cv2.VideoCapture(result_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    result_width = int(result_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    result_height = int(result_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate output video path by appending '_merged' to the original video file name
    output_path = os.path.join(os.path.dirname(video_path), f"{file_name}_merged{file_ext}")

    # Video writer for the merged video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (result_width * 3, result_height))

    for frame_index in tqdm(range(total_frames)):
        ret, frame = cap.read()
        ret_result, result_frame = result_cap.read()
        
        if not ret or not ret_result:
            break

        # Load the corresponding mask
        mask_path = os.path.join(mask_dir, f"{frame_index:05d}.png")
        if not os.path.exists(mask_path):
            print(f"Mask file {mask_path} does not exist.")
            break

        mask_img = cv2.imread(mask_path)

        # Resize original frame and mask to match the result frame size
        frame_resized = cv2.resize(frame, (result_width, result_height))
        mask_resized = cv2.resize(mask_img, (result_width, result_height))

        # Merge the frame, mask image, and result video frame side by side
        merged_frame = np.hstack((frame_resized, mask_resized, result_frame))
        out.write(merged_frame)

    cap.release()
    result_cap.release()
    out.release()
    print(f"Video saved at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge original video, mask images, and result video side by side.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the original video file.")
    args = parser.parse_args()

    # Determine the mask directory from the video path
    base_name = os.path.basename(args.video_path)
    file_name, _ = os.path.splitext(base_name)
    mask_dir = os.path.join("data", file_name)

    # Check if the mask directory exists
    if not os.path.exists(mask_dir):
        print(f"Mask directory {mask_dir} does not exist.")
    else:
        merge_videos(args.video_path, mask_dir)
