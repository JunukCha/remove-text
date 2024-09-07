## Introduction
In this project, I implemented a method to remove text from videos, utilizing the powerful capabilities of Optical Character Recognition (OCR) and inpainting techniques. This implementation builds on the excellent work by [Youngmin Baek, CRAFT](https://github.com/clovaai/CRAFT-pytorch) for OCR and [Yanhong Zeng, STTN](https://github.com/researchmm/STTN) for inpainting.

## Install
To set up the environment, run the following script:

```
source scripts/install.sh
```

## Download Checkpoints
`scripts/download.sh`

Your final folder structure should look like this:

```
(root)
├── archs
├── checkpoints
├── data
├── scripts
├── lib
├── .gitignore
├── main_craft.py
├── main_inpaint.py
├── main_sr.py
├── requirements.txt
└── visualization.py
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

(Optional) Super resolution

```
source scripts/run_super_res.sh
```

## Visualization of the Merged Video
<img src="https://github.com/user-attachments/assets/c2220ed0-fe52-4c97-b06f-5c2fe85899cc">
