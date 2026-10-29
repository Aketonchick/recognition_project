import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import os
from os import PathLike

from typing import Callable
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # Устанавливаем бэкенд ДО импорта pyplot
# Matplotlib по умолчанию пытается использовать графический интерфейс (GUI) Tkinter,
# который привязан к главному потоку (main thread) программы.
# Когда ты запускаешь show_results_to_user в фоновом потоке через asyncio.to_thread,
# Matplotlib пытается создать окно или сохранить рисунок в этом дочернем потоке,
# а Tkinter запрещает это делать. Отсюда и падение сервера.
import matplotlib.pyplot as plt

# Цвета для классов (BGR для OpenCV)
COLORS = {
    'reading_mark': (0, 255, 0),  # reading_mark - зеленый
    'series_mark': (255, 0, 0)  # series_mark - синий
}

def lenghten(x1, y1, x2, y2, coefficient, downbigger=True):
    """
    Функция для расширения изображения.
    :return: Координаты левого верхнего угла и правого нижнего угла изображения.
    """
    x1 = x1 - coefficient
    y1 = y1 - coefficient
    x2 = x2 + coefficient
    if downbigger:
        y2 = y2 + int(coefficient * 1.5) # для лучшей видимости запятой
    return x1, y1, x2, y2


def crop_and_save_image(model, image_path: PathLike, output_path: PathLike, class_names: dict) -> list[PathLike]:
    """
    Функция обрабатывает изображения (находит все объекты и обрезаих их). В результаты мы получаем список ссылок на
    полученные изображения.
    :param model: модель yolo
    :param image_path: путь к изображению, объекты которого необходимо обрезать
    :param output_path: путь к папке, куда сохраняются обрезанные изображения
    :param class_names:
    :return: список ссылок на обрезанные изображения
    """
    image_name = os.path.basename(image_path)
    # Загружаем изображение
    image = cv2.imread(str(image_path))

    # Копируем изображение, чтобы не изменять оригинал
    img_copy = image.copy()

    with torch.no_grad():
        results = model(img_copy) #verbose=False, чтобы не было вывода информации

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    cropped_images_paths = []
    for i, (box, cls) in enumerate(zip(boxes[:2], classes[:2])):  # [:2] берем только 2 самых вероятных бокса (1 readings и 1 series)
        name = class_names[int(cls)]
        # Обрезка и сохранение
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        cropped_image = crop_image(img_copy, box,f"{image_name}_{name}_{i}.jpg")
        out_path = Path(output_path) / f"{image_name}_{name}_{i}.jpg"
        cv2.imwrite(str(out_path), cropped_image)  # сохраняем изображение out_path
        print(f"✅ Сохранено: {out_path}")

        cropped_images_paths.append(out_path) # сохраняем пути изображений в список (для дальнейшей работы)

    return cropped_images_paths


def crop_image(image, box, prefix: str):
    """Обрезает изображение по прямоугольнику и сохраняет."""
    x1, y1, x2, y2 = map(int, box)
    if "reading" in prefix: # расширяем картинку вниз для показаний (нужно найтим запятую)
        x1, y1, x2, y2 = lenghten(x1, y1, x2, y2, 12, downbigger=True)
    else:
        x1, y1, x2, y2 = lenghten(x1, y1, x2, y2, 12, downbigger=False)

    cropped_image = image[y1:y2, x1:x2]

    return cropped_image


def processPhotoAndGetRecognitionResults(image_file: str, output_folder: str, inference) -> None:
    """
    Функция для распознавания цифр на изображении. В результате получаем изображения, на которых написаны результаты
    распознавания (для демонстрации). Изображения сохраняются в заданную папку.

    :param image_file: путь к изображению
    :param output_folder: путь к папке, куда будут сохраняться обработанные изображения
    :param inference: полученные результаты обработки изображения
    """

    image = Image.open(image_file)
    # создаем фигуру и отображаем изображение
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)

    # добавляем текст
    ax.text(0, 0, inference, fontsize=10, color='green')

    # убираем оси, чтобы не мешали
    ax.axis("off")

    # сохраняем результат в папку
    os.makedirs(output_folder, exist_ok=True)

    image_name = os.path.basename(image_file)
    output_path = os.path.join(output_folder, image_name)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()
    print(f"Изображение сохранено в: {output_path}")

