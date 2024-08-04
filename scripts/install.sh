conda create -n remove_subtitles python=3.10 -y
conda activate remove_subtitles

pip install torch torchvision
pip install -r requirements.txt

git clone https://github.com/clovaai/CRAFT-pytorch.git
git clone https://github.com/researchmm/STTN.git