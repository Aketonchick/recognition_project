import asyncio
from fastapi import FastAPI, UploadFile, File
import shutil
import uvicorn
from main import show_results_to_user
from utils.counter import Counter

app = FastAPI()
count_of_images = Counter("config.json", "count_of_user_images")

@app.post("/predictions", tags=["Обработка изображения"])
async def predict(file: UploadFile = File(...)):
    # 1. Сохраняем файл
    file_location = f"images/user_images/{count_of_images.get_value()}.jpg"
    count_of_images()  # увеличиваем счетчик изображений

    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_folder = "results"

    # 2. Запускаем тяжелую обработку в отдельном потоке
    result = await asyncio.to_thread(show_results_to_user, file_location, output_folder)


    # 3. Возвращаем результат
    return {"Результаты": result}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, workers=2)


