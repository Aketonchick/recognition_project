# используется для автоматической разметки в CVAT с помощью уже обученной модели,
# для того чтобы загрузить изображения в видео память - необходимо разделить изображения на батчи, чтобы
# хватило памяти для обработки

import os
import math
import shutil
import re

current_path = r"D:\python\datasets\TEST"

images_dir_path = r"D:\python\datasets\TEST\ElectroMeters_train_dataset"
list_of_images = os.listdir(images_dir_path)

def natural_sort_key(s):
    """
    Извлекает число из строки вида 'img_XXX.jpg' для сортировки по возрастанию
    """
    # Ищем цифры в строке
    match = re.search(r'(\d+)', s)
    if match:
        return int(match.group(1))  # Возвращаем число как int
    return 0  # Если число не найдено

list_of_images.sort(key=natural_sort_key)

size_of_batch = 500

count_of_batch_folders = math.ceil(len(list_of_images) / size_of_batch)

for i in range(count_of_batch_folders):
    new_path = os.path.join(current_path, f"batch{i}")
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    for _ in range(size_of_batch):
        if len(list_of_images) == 0:
            break
        image = list_of_images[0]
        image_path = os.path.join(images_dir_path, image)
        list_of_images.pop(0)
        shutil.copy(image_path, new_path)

print("Задача выполнена")