import os
import shutil

current_path = r"D:\python\CVAT2YOLO-main\my_dataset_yolo3"
destination_path = r"D:\python\CVAT2YOLO-main\work_dataset\images"

real_dataset_images_dict = {"test": r"D:\python\CVAT2YOLO-main\my_dataset_yolo2\results_crops_test",
                            "train": r"D:\python\CVAT2YOLO-main\my_dataset_yolo2\results_crops_train",
                            "val": r"D:\python\CVAT2YOLO-main\my_dataset_yolo2\results_crops_val"}

images_path_dict = {"test": "Test.txt",
                    "train": "Train.txt",
                    "val": "Validation.txt"}

for images_set in images_path_dict:
    text_file = os.path.join(current_path, images_path_dict[images_set])
    count_of_files = 1
    with open(text_file, 'r') as f:
        links = f.read().split("\n")
        while "" in links:
            links.remove("")
        for link in links: # ссылка на изображение (data/obj_Test_data/img_1.jpg)
            img = os.path.basename(link)
            link = os.path.join(real_dataset_images_dict[images_set], img) # настоящий адрес изображения
            new_path = os.path.join(destination_path, images_set, f"img_{count_of_files}.jpg")
            shutil.copy(link, new_path)
            count_of_files += 1
            # if images_set == "train" and count_of_files in [207, 271, 401]:
            #     count_of_files += 1
