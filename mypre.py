import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt')
    results=model.predict(source='img/Area2.tif', imgsz=2048, device='0', save=True, save_txt=True)

