import torch.cuda
from ultralytics import YOLO
from pathlib import Path
import os
from os import PathLike
from utils.preprocessing_and_cropping import crop_and_save_image, processPhotoAndGetRecognitionResults
from models_loading import get_yolo_models
import time

import matplotlib.pyplot as plt
from PIL import Image

from typing import Callable
import threading
file_lock = threading.Lock()

# исходная модель училась определять следующие объекты
series_cls_objects_dict = {0: '1',
                           1: '2',
                           2: '3',
                           3: '4',
                           4: '5',
                           5: '6',
                           6: '7',
                           7: '8',
                           8: '9',
                           9: '0',
                           10: '-',
                           11: 'A',
                           12: 'D',
                           13: "Г",
                           14: "I",
                           15: "o",
                           16: "J"}

readings_cls_objects_dict = {0: '1',
                             1: '2',
                             2: '3',
                             3: '4',
                             4: '5',
                             5: '6',
                             6: '7',
                             7: '8',
                             8: '9',
                             9: '0',
                             10: '.',
                             11: 'T',
                             12: '(sigma)'}

CLASS_NAMES = ['reading_mark', 'series_mark']


def series_recogntion_model(model, image, cls_objects: dict = series_cls_objects_dict) -> str:
    """
    Функция для распознавания цифр на номере счетчика
    :param weights: веса полученные в результате обучения нашей нейронной сети
    :param image: изображения номера счетчика
    :param cls_objects: словарь для интерпретации номера класса с его значением
    :return: на выходе получаем строку из распознанных чисел
    """
    boxes, confs, classes = takeResults(model, image)
    classMarkAndCoordinates = formListOfMarksAndBoxes(classes, boxes, cls_objects, confs)

    # обрабатываем вторую строку номера
    low_line_series = []
    indexes = []
    flag = False
    for i in range(len(classMarkAndCoordinates) - 1):
        if classMarkAndCoordinates[i][1][1] - classMarkAndCoordinates[i + 1][1][1] < -15:
            low_line_series.append(classMarkAndCoordinates[i + 1][0])
            indexes.append(i + 1)
            flag = True
        elif classMarkAndCoordinates[i][1][1] - classMarkAndCoordinates[i + 1][1][1] > 15:
            flag = False
        elif flag == True:
            low_line_series.append(classMarkAndCoordinates[i + 1][0])
            indexes.append(i + 1)

    # добавляем вероятности к выводу
    confs = []
    for i in range(len(classMarkAndCoordinates)):
        confs.append(classMarkAndCoordinates[i][2])

    inference = ''
    for i in range(len(classMarkAndCoordinates)):
        if i in indexes:
            continue
        inference += classMarkAndCoordinates[i][0]  # добавляем цифру в выводную строку

    # добавляем цифры 2-ой строки в конец
    inference = inference + ''.join(low_line_series)
    # обработка двойного -- -> -
    while "--" in inference:
        inference = inference.replace("--", "-")

    # return inference, confs
    return inference

def readings_recogntion_model(model, image, cls_objects: dict = readings_cls_objects_dict) -> str:
    """
    Функция для распознавания цифр на показаниях счетчика
    :param weights: веса полученные в результате обучения нашей нейронной сети
    :param image: изображения номера счетчика
    :param cls_objects: словарь для интерпретации номера класса с его значением
    :return: на выходе получаем строку из распознанных чисел
    """
    boxes, confs, classes = takeResults(model, image)
    classMarkAndCoordinates = formListOfMarksAndBoxes(classes, boxes, cls_objects, confs)

    # добавляем вероятности к выводу
    confs = []
    for i in range(len(classMarkAndCoordinates)):
        confs.append(classMarkAndCoordinates[i][2])

    inference = ''
    for i in range(len(classMarkAndCoordinates)):
        inference += classMarkAndCoordinates[i][0]  # добавляем цифру в выводную строку

    # обработка двойной .. -> .
    while ".." in inference:
        inference = inference.replace("..", ".")

    # дополнительно выделяем регистр T и наличие суммы(sigma)
    inference, sigma, T = processTvalueAndSigma(inference)

    # inference, sigma, T, confs
    return inference, sigma, T

def takeResults(model, image, result_number: int = 0) -> tuple:
    # Выполняем предсказание
    with torch.no_grad():
        results = model.predict(image, iou=0.2, agnostic_nms=True)
        # iou отвечает за разделение боксов, наиболее вероятный бокс в области становится главным.
        # Именно этот бокс мы считаем за класс. Чем ближе iou к нулю, тем меньше становится порог удаления маловероятных
        # боксов, наслаивающихся на самый вероятный бокс.

    # Извлекаем координаты боксов
    result = results[result_number]
    boxes = result.boxes.xyxy  # Координаты в формате x1, y1, x2, y2
    confs = result.boxes.conf  # Уверенность
    classes = result.boxes.cls  # ID классов

    return boxes, confs, classes

def formListOfMarksAndBoxes(classes, boxes, cls_objects, confs) -> list[str, list]:
    classMarkAndItsCoordinate = []

    for i, number in enumerate(classes):
        classMarkAndItsCoordinate.append((cls_objects[int(number)], boxes[i], confs[i]))

    classMarkAndItsCoordinate.sort(
        key=lambda x: x[1][0])  # сортируем боксы по оси X (крайний левый - первый, крайний правый - последний)

    # обработка T и ближайшего к нему числа
    processCoordinatesTvalue(classMarkAndItsCoordinate)

    return classMarkAndItsCoordinate

def processTvalueAndSigma(string_number) -> tuple[str, str, str]:
    string_number, T = processTvalue(string_number)
    string_number, sigma = processSigma(string_number)

    return string_number, sigma, T

def processTvalue(string_number) -> tuple[str, str]:
    new_string_number = string_number
    list_positions = ["T0", "T1", "T2", "T3", "T4"]
    T = "nonT"
    extra_checking = True
    for position in list_positions:
        if position in string_number:
            T = position
            new_string_number = string_number.replace(T, "")
            extra_checking = False
            break

    if extra_checking:           # когда мы убедились, что в регистре нету числа, то можно попробовать обработать T (саму букву)
        if "T" in string_number:
            T = "T"
            new_string_number = string_number.replace(T, "")

    return new_string_number, T

def processSigma(string_number) -> tuple[str, str]:
    new_string_number = string_number
    sigma = "nonsigma"
    if "(sigma)" in string_number:
        new_string_number = string_number.replace("(sigma)", "")
        sigma = "sigma"

    return new_string_number, sigma

def processCoordinatesTvalue(list_of_boxes: list) -> list | tuple[int, int]:  # processCoordinatesTvalueAndClosestNumber2T
    """
    Функция для нахождения позиции Т и ближайшего числа к Т (нахождение его позиции в отсортированном списке боксов)
    :param list_of_boxes: список списков названий классов и координат боксов
    :return: позиция T и ближайшего числа к T в списке или возвращает обратно список
    """
    positionT, texistance = detectPositionT(list_of_boxes)
    positionSigma, sigmaexistance = detectPositionSigma(list_of_boxes)
    comparison_difference = []

    if texistance is True:
        for j in range(len(list_of_boxes)):
            if j == positionT:
                continue
            elif sigmaexistance is True and j == positionSigma:
                continue
            comparison_difference.append((abs(list_of_boxes[j][1][1] - list_of_boxes[positionT][1][1]), j))
        comparison_difference.sort(key=lambda x: x[0])
        closest_number_to_T = comparison_difference[0]
        distance_closest_number_to_T = closest_number_to_T[0]
        position_closest_number_to_T = closest_number_to_T[1]

        if distance_closest_number_to_T > 20: # в случае если T одна, то ближайшее число будет в показаниях (исключаем этот случай)
            valueT = list_of_boxes[positionT]
            list_of_boxes.pop(positionT)
            list_of_boxes.append(valueT)
            return list_of_boxes
        else:
            # удаляем из списка T и ближайшее к нему число и выделяем их отдельно
            valueT = list_of_boxes[positionT]
            value_closest_number_to_T = list_of_boxes[position_closest_number_to_T]

            list_of_boxes[positionT] = 0
            list_of_boxes[position_closest_number_to_T] = 0

            while 0 in list_of_boxes:
                list_of_boxes.remove(0)
            list_of_boxes.append(valueT)
            list_of_boxes.append(value_closest_number_to_T)
            return list_of_boxes
    else:
        return list_of_boxes


def detectPositionT(list_of_boxes: list) -> tuple[int, bool]:
    positionT = None
    texistance = False
    for i in range(len(list_of_boxes)):
        if list_of_boxes[i][0] == "T":
            positionT = i
            texistance = True
            break

    return positionT, texistance

def detectPositionSigma(list_of_boxes: list) -> tuple[int, bool]:
    positionSigma = None
    sigmaexistance = False
    for i in range(len(list_of_boxes)):
        if list_of_boxes[i][0] == "(sigma)":
            positionSigma = i
            sigmaexistance = True
            break

    return positionSigma, sigmaexistance


def get_recogntion_results(lits_of_paths: list[PathLike]) -> dict:
    _, readings_model_yolo, series_model_yolo = get_yolo_models()
    results_dict = {
        "readings": [],
        "series": []
    }
    for path in lits_of_paths:
        path = str(path) # преобразуем WindowsPath в строку
        if "_reading_" in path:
            result = readings_recogntion_model(readings_model_yolo, path)
            processPhotoAndGetRecognitionResults(path, "images/reading_cropped_images", result)
            results_dict["readings"].append(result)
        elif "_series_" in path:
            result = series_recogntion_model(series_model_yolo, path)
            processPhotoAndGetRecognitionResults(path, "images/series_cropped_images", result)
            results_dict["series"].append(result)

    return results_dict

def show_results_to_user(image: PathLike, output_path: PathLike):
    """
    :param image: изображение, которое подается на вход
    :param output_path: путь к папке, куда сохраняются обрезанные изображения
    :return: словарь: результаты распознавания для каждого объекта (текст)
    """
    areas_model_yolo, _, _ = get_yolo_models()
    cropped_images_paths = crop_and_save_image(areas_model_yolo, image, output_path, CLASS_NAMES)
    results_dict = get_recogntion_results(cropped_images_paths)
    return results_dict

def main2():
    """
    Программа, в которой обрабатываются изображения из назначенной папки. На выходе получаем папку с изображениями,
    на которых написаны результаты распознавания
    """
    areas_model_yolo, readings_model_yolo, series_model_yolo = get_yolo_models()

    output_folder_path = r"results"
    input_folder_path = r"tests/input_images"

    input_images_list = os.listdir(input_folder_path)
    for image in input_images_list:
        image_file = os.path.join(input_folder_path, image)
        crop_and_save_image(areas_model_yolo, image_file, output_folder_path, CLASS_NAMES)

    readings_path_to_folder_results = "images/reading_cropped_images"
    series_path_to_folder_results = "images/series_cropped_images"
    for image in os.listdir(output_folder_path):
        if "reading" in image:
            image_file = os.path.join(output_folder_path, image)
            processPhotoAndGetRecognitionResults(image_file, readings_path_to_folder_results, readings_recogntion_model,
                                                 readings_model_yolo, readings_cls_objects_dict)
        elif "series" in image:
            image_file = os.path.join(output_folder_path, image)
            processPhotoAndGetRecognitionResults(image_file, series_path_to_folder_results, series_recogntion_model,
                                                 series_model_yolo, series_cls_objects_dict)

    print("end!")


def main():
    output_path = "results"
    while True:
        image_path = input("Введите ссылку на изображение:")
        results_dict = show_results_to_user(image_path, output_path)
        print(f"{"="*10}РЕЗУЛЬТАТЫ{"="*10}")
        for key in results_dict:
            print(f"{key}: {results_dict[key]}")
        print()
        print("Хотите продолжить программу?{y/n}")
        answer = input()
        if answer == "N" or answer == "n":
            break


import time
if __name__ == "__main__":
    main()



