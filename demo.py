import cv2
import glob
import os
import os.path as osp
import torch
import argparse
from archs.mia_vsr_arch import MIAVSR
from basicsr.data.data_util import read_img_seq
from basicsr.utils import imwrite, tensor2img


def main(args):
    # -------------------- Configurations -------------------- #
    device = torch.device(args.device)
    model_path = args.model_path
    lr_folder = args.lr_folder
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
                 spynet_path='/data1/home/zhouxingyu/zhouxingyu_vsr/MIA-VSR/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth')
    model.load_state_dict(torch.load(model_path)['params'], strict=False)
    model.eval()
    model = model.to(device)

    subfolder_l = sorted(glob.glob(osp.join(lr_folder, '*')))

    # for each subfolder
    subfolder_names = []
    for subfolder in subfolder_l:
        subfolder_name = osp.basename(subfolder)
        subfolder_names.append(subfolder_name)

        # read lq and gt images
        imgs_lq, imgnames = read_img_seq(subfolder, return_imgname=True)

        # inference
        name_idx = 0
        imgs_lq = imgs_lq.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs, _ = model(imgs_lq)
            outputs = outputs.squeeze(0)
        # convert to numpy image
        for idx in range(outputs.shape[0]):
            img_name = imgnames[name_idx] + '.png'
            output = tensor2img(outputs[idx], rgb2bgr=True, min_max=(0, 1))
            imwrite(output, osp.join(save_folder, subfolder_name, f'{img_name}'))
            name_idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MIAVSR inference.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference (e.g., cuda, cpu).')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--lr_folder', type=str, required=True, help='Folder containing low-resolution input images.')
    
    args = parser.parse_args()
    main(args)
