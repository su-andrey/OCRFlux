!pip install -q torch transformers accelerate pymupdf Pillow
!mkdir - p. / offload

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import fitz  # PyMuPDF
from PIL import Image
from google.colab import files
import gc
from datetime import datetime

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


def process_pdf(file_path: str, question: str = PROMPT) -> str:  # Обработка (сканирование) пдф
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page_num in range(min(2, len(doc))):  # Пока больше двух страниц за раз не обрабатываю, обработка постранично
            print(f"page №{page_num + 1} in process")
            pix = doc[page_num].get_pixmap(dpi=200)
            if pix.width == 0 or pix.height == 0:
                print(f"Пустая страница {page_num + 1}")
                continue
            # Прочитал изображение страницы, проверил на корректность (не пустоту), сохранил в формате для модели
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            if image.size[0] == 0 or image.size[1] == 0:
                print(f"Нулевой размер изображения на странице {page_num + 1}")
                continue
            image.save(f"page_{page_num + 1}.png")
            messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": f"file://page_{page_num + 1}.png"},
                                    {"type": "text", "text": PROMPT}
                                ],
                            },
                            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text],images=[image], padding=True, return_tensors="pt")
            inputs=inputs.to(model.device)
            outputs_ids = model.generate(**inputs, temperature=0.0,max_new_tokens=4096, do_sample=False)
            generated_ids = [
                outputs_ids[len(input_ids):]
                for input_ids, outputs_ids in zip(inputs.input_ids, outputs_ids)
            ]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # Получаем распознанный текст от модели, после чего добавляем его в общую переменную
            full_text += f"\n\nСтраница {page_num + 1}:\n{output_text}"
        return full_text  
    except Exception as page_error:
      print(f"Ошибка обработки страницы №{page_num + 1}: {page_error}")
      full_text += f"\n\n[Ошибка на странице №{page_num + 1}]"
      if 'inputs' in locals():
          del inputs # Подчищаем за собой хранилище
      if 'outputs' in locals():
          del outputs
      clean_gpu() # Стоит подчищать и при успешном заверешении, сделаю в следующем коммите. Будет важно при батчах
      return full_text


if __name__ == "__main__":
    print("Загрузите PDF файл для извлечения текста(до 5MB). Рекомендованные языки: английский/китайский:")
    uploaded = files.upload()
    if not uploaded:
        print("Файл не загружен")
    else:
        pdf_name = next(iter(uploaded))
        result = process_pdf(pdf_name, PROMPT)
        print("Результат обработки:")
        print(result) # Пока вывожу в консоль для проверки
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # текст называем по времени
        name = f"extracted_text_{timestamp}.md"
        with open(name, "w", encoding="utf-8") as md_file:
          md_file.write(result) # Сохраняем
        files.download(name) # Загрузка для коллаба
    if 'processor' in globals():
        del processor
    if 'model' in globals():
        del model
    clean_gpu() #  Подчищаем за собой память