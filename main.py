!pip install - q torch transformers accelerate pymupdf Pillow !mkdir - p. / offload

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import fitz  # PyMuPDF
from PIL import Image
from google.colab import files
import gc
from datetime import datetime
import json

PROMPT = ("This is the image of one page of document. Just return the plain text"
          "of this document as if you were reading it naturally. ALL tables should be presented in HTML format. "
          "If there are images or figures, present them as <Image>(left,top),(right,bottom)</Image> "
          "Present all titles and headings as H1 headings. Do not hallucinate")


# Промпт используемый по всему коду вынесен в константу

# Максимально подчищаем память в связи с низким количеством ресурсов
def clean_gpu():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    print(f"GPU память свободна: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.1f} GB")


clean_gpu()

try:  # Запускаем модель с экономией по памяти
    processor = AutoProcessor.from_pretrained(
        "ChatDOC/OCRFlux-3B",
        trust_remote_code=True,
        padding_side='left'
    )
    model = AutoModelForImageTextToText.from_pretrained(
        "ChatDOC/OCRFlux-3B",
        device_map="auto",
        offload_folder="./offload",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    raise


def process_pdf(file_path: str, question: str = PROMPT, is_pdf=True) -> str:
    try:
        full_text = ""
        doc = fitz.open(file_path)
        pix = doc[0].get_pixmap(dpi=200)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image.save(f"page.png")
        # Прочитал изображение страницы, проверил на корректность (не пустоту), сохранил в формате для модели
        if image.size[0] == 0 or image.size[1] == 0:
            print("Нулевой размер изображения на странице")
            return
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://page.png"},
                    {"type": "text", "text": PROMPT}
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs_ids = model.generate(**inputs, temperature=0.0, max_new_tokens=4096, do_sample=False)
        generated_ids = outputs_ids[:, inputs.input_ids.shape[1]:]
        output_text = \
        processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        # Получаем распознанный текст от модели, после чего добавляем его в общую переменную
        try:
            result_data = json.loads(output_text)  # Обрабатываем в форме json
            if isinstance(result_data,
                          dict) and 'natural_text' in result_data:  # Извлекаем поле natural text - остальные технические: например язык, наличие таблиц и картинок
                full_text += result_data['natural_text']
            else:
                full_text += output_text  # Если вдруг получаем не json или json без нужного поля (что само по себе странно), просто пишем то, что получили
        except json.JSONDecodeError:
            full_text += output_text
        del inputs, outputs_ids, image  # Подчищаем за собой хранилище, в том числе картинку
        clean_gpu()
        return full_text.strip()
    except Exception as page_error:
        print(f"Ошибка обработки страницы: {page_error}")
        full_text += f"\n\n[Ошибка на странице]"
        if 'inputs' in locals():
            del inputs  # Подчищаем за собой хранилище
        clean_gpu()
        return full_text.strip()


if __name__ == "__main__":
    print("Загрузите PDF файл для извлечения текста (до 5MB). Рекомендованные языки: английский/китайский:")
    uploaded = files.upload()
    if not uploaded:
        print("Файл не загружен")
    else:
        pdf_name = next(iter(uploaded))
        result = process_pdf(pdf_name, PROMPT)
        print("Результат обработки:")
        print(result)  # Пока вывожу в консоль для проверки
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # текст называем по времени
        filename = f"extracted_text_{timestamp}.md"
        with open(filename, "w", encoding="utf-8") as md_file:
            md_file.write(result)  # Сохраняем
        files.download(filename)  # Загрузка для коллаба
    if 'processor' in globals():
        del processor
    if 'model' in globals():
        del model
    clean_gpu()  # Подчищаем за собой память