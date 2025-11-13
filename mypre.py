import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp11/weights/best.pt')
    results=model.predict(source='img/img1.png', imgsz=640, device='0', save=True, save_txt=True)

