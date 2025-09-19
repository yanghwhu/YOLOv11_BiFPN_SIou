# cd ultralytics_yolo
#  conda create -n yolo11 python=3.9
conda activate yolo11
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# CUDA 版本要和你当前设备 匹配

# pip install ultralytics



python mytrain.py