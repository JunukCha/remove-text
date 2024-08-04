## Introduction
In this project, I implemented a method to remove text from videos, utilizing the powerful capabilities of Optical Character Recognition (OCR) and inpainting techniques. This implementation builds on the excellent work by [Youngmin Baek, CRAFT](https://github.com/clovaai/CRAFT-pytorch) for OCR and [Yanhong Zeng, STTN](https://github.com/researchmm/STTN) for inpainting.

## Install
To set up the environment, run the following script:

```
source scripts/install.sh
```

## Download Checkpoints
For the OCR task, refer to the [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) repository. You can download the general checkpoint and place the `craft_mlt_25k.pth` file in the root folder.

For the inpainting task, refer to the [STTN](https://github.com/researchmm/STTN) repository. Download the checkpoint and place it in `STTN/checkpoints/sttn.pth`.

Your final folder structure should look like this:

```
(root)
├── data
├── scripts
├── CRAFT-pytorch
├── STTN
│   └── checkpoints
│       └── sttn.pth
├── .gitignore
├── main_craft.py
├── requirements.txt
├── visualization.py
└── craft_mlt_25k.pth
```

## Inference
To save the text mask, run:

```
source scripts/save_text_mask.sh
```

To inpaint the region of the text mask, run:

```
source scripts/run_inpaint.sh
```

To merge the video with the original, the text mask, and the result, run:

```
source scripts/vis.sh
```

## Visualization of the Merged Video
<img src="https://github.com/user-attachments/assets/c2220ed0-fe52-4c97-b06f-5c2fe85899cc">
