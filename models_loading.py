# work_dataset/models_loading.py
import torch.cuda
from ultralytics import YOLO

areas_detector_weights = r"models_learning/runs/segment/segmentation_optimized/weights/best.pt"
series_weights = r"models_learning/runs/detect/series_detector/weights/best.pt"
readings_weights = r"models_learning/runs/detect/readings_detector/weights/best.pt"


print("инициализация весов YOLO моделей")
areas_model_yolo = YOLO(areas_detector_weights)
areas_model_yolo.eval()
readings_model_yolo = YOLO(readings_weights)
readings_model_yolo.eval()
series_model_yolo = YOLO(series_weights)
series_model_yolo.eval()

if torch.cuda.is_available():
    print("🔥 Используется GPU")
    # YOLO автоматически использует GPU, если он доступен
else:
    print("💻 Используется CPU")

print("✅ Модели загружены!")

def get_yolo_models():
    """
    :return: Возвращает уже загруженные модели YOLO
    """
    return areas_model_yolo, readings_model_yolo, series_model_yolo
