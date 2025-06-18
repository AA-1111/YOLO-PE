from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model_yaml = r"E:\KAN\2\ultralytics\cfg\models\v8\MobileNetV4.yaml"

    # 训练模型
    data_yaml = r"E:\KAN\2\ultralytics\cfg\datasets\hat.yaml"
    pre_model = r"E:\KAN\2\yolov8n.pt"
    model = YOLO(model_yaml, task="detect").load(pre_model)
    result = model.train(
        data=data_yaml,
        epochs=300,
        imgsz=640,
        batch=16,
        workers=2
    )
