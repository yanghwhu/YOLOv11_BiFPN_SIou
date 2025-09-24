import argparse
from ultralytics import YOLO


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--resume', action='store_true', help='resume training')
    # 可以添加其他需要的参数
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()

    model = YOLO(model=r'..\ultralytics\cfg\models\11\yolo11_BiFPN.yaml')
    model.load('yolo11s.pt')
    model.train(data=r'..\ultralytics\cfg\datasets\tomb.yaml',
                imgsz=640,
                epochs=opt.epochs,
                batch=opt.batch,
                workers=opt.workers,
                device=opt.device,
                optimizer='SGD',
                close_mosaic=10,
                resume=opt.resume,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )