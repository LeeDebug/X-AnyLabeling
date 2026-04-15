from ultralytics import YOLO

# 自动下载 YOLOv8n (nano 最小最快版本) 到本地缓存
model = YOLO("yolov8x.pt")

# 如果你想下载其他大小，把 "yolov8n.pt" 改成：
# - yolov8n.pt (nano 最小最快版本)
# - yolov8s.pt (small 小模型，平衡速度精度)
# - yolov8m.pt (medium 中等模型)
# - yolov8l.pt (large 大模型，精度高)
# - yolov8x.pt (extra large 超大模型，精度最高)