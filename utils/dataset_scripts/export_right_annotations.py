import os
import shutil

path_of_labels = r"D:\python\CVAT2YOLO-main\my_dataset_yolo4\labels" # текстовые файлы с разметкой, имя файла - имя фотографии
path_of_images = "D:\python\CVAT2YOLO-main\my_dataset_yolo2"  # путь, по которому можно найти обрезанные изображения

work_folder = "D:\python\CVAT2YOLO-main\work_dataset\images_series"

images_links_path_dict = {"test": "Test.txt",
                          "train": "Train.txt",
                          "val": "Validation.txt"}

images_path_dict = {"test": "results_crops_test",
                    "train": "results_crops_train",
                    "val": "results_crops_val"}



for path in images_links_path_dict:
    folder = os.path.join(path_of_labels, path)
    list_of_files = os.listdir(folder)

    for annotation in list_of_files:
        image_name = os.path.basename(annotation)
        image_name = os.path.splitext(image_name)[0]
        image_name = image_name + ".jpg"
        image_file = os.path.join(path_of_images, images_path_dict[path], image_name)

        new_path = os.path.join(work_folder, path)
        shutil.copy(image_file, new_path)

