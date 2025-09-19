# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model=r'..\ultralytics\cfg\models\11\yolo11_BiFPN.yaml')
    model.load('yolo11n.pt')
    model.train(data=r'..\ultralytics\cfg\datasets\tomb.yaml',
                imgsz=640,
                epochs=100,
                batch=32,
                workers=0,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
