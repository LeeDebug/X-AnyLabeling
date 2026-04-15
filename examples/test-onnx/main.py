from ultralytics import YOLO

model = YOLO("./weights/best.onnx")
# 替换为你的验证集配置文件路径，自动计算mAP、Precision、Recall等指标
metrics = model.val(data="./dataset.yaml", device=0)
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
print(f"精确率: {metrics.box.mp}")
print(f"召回率: {metrics.box.mr}")