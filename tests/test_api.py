from locust import HttpUser, task, between
import os

folder = r"D:\python\FINAL_WORK\work_dataset\tests\input_images"
list_of_files = os.listdir(folder) * 10  # Умножаем для большего объема

class MeterAPITester(HttpUser):
    wait_time = between(1, 2)

    def on_start(self):
        """Этот метод вызывается при старте каждого пользователя."""
        # Каждый пользователь получает свою КОПИЮ списка
        self.my_files = list_of_files.copy()
        # Перемешиваем, чтобы пользователи не брали изображения в одном порядке
        import random
        random.shuffle(self.my_files)

    @task
    def upload_image(self):
        if not self.my_files:
            return
        # Берем последний элемент (это быстрее, чем pop(0))
        img_name = self.my_files.pop()
        with open(f"input_images/{img_name}", "rb") as img:
            self.client.post(
                "/predictions",
                files={"file": img},
                name="/predictions"
            )