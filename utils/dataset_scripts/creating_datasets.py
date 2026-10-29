import os
import random
import math

images_path = r"D:\python\FINAL_WORK\work_dataset\tests\input_images2"  # папка с исходным датасетом
amount_files = len(os.listdir(images_path)) # количество изображений в нашем датасете

# разбиваем исходный датасет на обучающую, тестовую и валидационную выборки
train_dataset = r"D:\python\datasets\ElectroMeters_train_dataset"
validation_dataset = r"D:\python\datasets\ElectroMeters_validation_dataset"
test_dataset = r"D:\python\datasets\Meters_test"

destination_path_dict = {"train_dataset": train_dataset,
             "validation_dataset": validation_dataset,
             "test_dataset": test_dataset}

coeff_split_dict = {"train_dataset": 0,
                   "validation_dataset": 1,
                   "test_dataset": 0}

for file_name in destination_path_dict:
    # создаем соответствующие папки
    if not os.path.exists(destination_path_dict[file_name]):
        os.makedirs(destination_path_dict[file_name])

def replacing_files(current_path: str, destination_path_dict: dict[str, str], coeff_split_dict: dict[str, float]) -> None:
    """
    :param current_path: текущая папка
    :param needed_path: новая папка
    :param coeff_split: коэффициент разбиения текущей выборки и перемещение файлов в указанную папку
    :return: None
    """
    amount_files = len(os.listdir(current_path)) # количество файлов в текущей папке

    for dataset in destination_path_dict:
        amount_replacing_files = math.floor(amount_files * coeff_split_dict[dataset]) # округляем в нижнюю сторону число файлов, которые необходимо перенести в новую папка
        list_of_files = os.listdir(current_path) # список названий файлов в текущей папке
        list_of_files = list_of_files.copy()

        image_count = 1
        while amount_replacing_files != 0:
            amount_replacing_files -= 1
            replacing_file = random.choice(list_of_files)                       # выбираем случайный файл из списка
            list_of_files.remove(replacing_file)                                # удаляем файл из списка файлов
            source = os.path.join(current_path, replacing_file)                        # путь текущего выбранного файла
            destination = os.path.join(destination_path_dict[dataset], f"img_{image_count}.jpg")      # новый путь для файла

            while os.path.exists(destination):
                image_count += 1
                destination = os.path.join(destination_path_dict[dataset], f"img_{image_count}.jpg")

            os.rename(source, destination)  # перемещение файла в папку


# img_path_source = r"D:\python\datasets\1"
# img_path_destination = r"D:\python\datasets\2"

# создаем датасеты для задачи детекции
# replacing_files(images_path, path_dict['train_dataset'], train_k)
replacing_files(images_path, destination_path_dict, coeff_split_dict)
