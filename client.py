import requests
from pathlib import Path
SERVER_URL = "https://ca68df5a7262.ngrok-free.app/"  # Адрес, на котором находится сервер (выведет при запуске)
files = ["test.pdf", "182.jpg"]            # Файл, который хотим обработать
OUTPUT_FILE = "result_all.md"             # Файл для сохранения результата
files_to_upload = []  # Список, в который будут записаны кортежи для отправки на сервер
for file_path in files:
    files_to_upload.append(
        ('files', (Path(file_path).name, open(file_path, 'rb'))))  # Записываем кортежи в правильном формате
    if not files_to_upload:
        print("Нет файлов для отправки")
        exit()

print("Отправка запроса")
response = requests.post(f"{SERVER_URL}/download/", files=files_to_upload) # Отправляем запрос
if response.status_code == 200:
    with open(OUTPUT_FILE, "wb") as f:
        f.write(response.content)
    print(f"Результат сохранен в {OUTPUT_FILE}")
else:
    print(f"Ошибка сервера: {response.status_code} - {response.text}")
for file in files_to_upload:
    file[1][1].close()  # Закрываем все файлы (открывали при отправке)