import requests

SERVER_URL = "https://someadress12.ngrok-free.app/"  # Адрес, на котором находится сервер (выведет при запуске)
FILE_PATH = "text.jpeg"            # Файл, который хотим обработать
OUTPUT_FILE = "result_jpg.md"             # Файл для сохранения результата

with open(FILE_PATH, 'rb') as f:
    files = {'files': (FILE_PATH, f)}
    print("Отправка запроса")
    response = requests.post(f"{SERVER_URL}/download/", files=files)

if response.status_code == 200:
    print("Ответ получен успешно. Сохранение результата")
    with open(OUTPUT_FILE, 'wb') as f:
        f.write(response.content)
    print(f"Результат сохранен в {OUTPUT_FILE}")

else:
    print(f"Ошибка: {response.status_code} - {response.text}")