#!pip install -q torch transformers accelerate pymupdf Pillow fastapi python-multipart nest-asyncio pyngrok
#!mkdir -p ./offload


import gc
import json
import os
import tempfile
from datetime import datetime
from typing import List

import fitz  # PyMuPDF
import nest_asyncio
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pyngrok import ngrok
from transformers import AutoModelForImageTextToText, AutoProcessor

nest_asyncio.apply() # Разрешаем множественные лупы в колаб

PROMPT = ("This is the image of one page of document. Just return the plain text"
          "of this document as if you were reading it naturally. ALL tables should be presented in HTML format. "
          "If there are images or figures, present them as <Image>(left,top),(right,bottom)</Image> "
          "Present all titles and headings as H1 headings. Do not hallucinate")
# Промпт используемый по всему коду вынесен в константу

# Максимально подчищаем память в связи с низким количеством ресурсов
app = FastAPI()


def clean_gpu():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    print(f"GPU память свободна: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.1f} GB")


processor, model = None, None  # Предобъявляем модель и процессор перед запуском

@app.on_event("startup")  # Запускаем модель при старте сервера с экономией по памяти
async def startup_event():
    global processor, model
    clean_gpu()
    try:
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


def convert_to_md(file_path: str, question: str = PROMPT, is_pdf=True) -> str:
    try:
        full_text = ""
        doc = fitz.open(file_path)
        for num in range(len(doc)):
            pix = doc[num].get_pixmap(dpi=200)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            if is_pdf:
                image.save("page.png")
                file_path = "page.png"
            # Прочитал изображение страницы, проверил на корректность (не пустоту), сохранил в формате для модели
            if image.size[0] == 0 or image.size[1] == 0:
                print("Нулевой размер изображения на странице")
                return
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{file_path}"},
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


@app.post("/download/")
async def download_result(files: List[UploadFile] = File(...)):
    results = {}
    for file in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file.close()
                temp_path = temp_file.name
            results[file.filename] = convert_to_md(temp_path, file.filename[-4:] == '.pdf')
            os.unlink(temp_path)  # Удаляю самостоятельно, для надёжности в случае ошибок (поэтому выше delete=False)
        except Exception as e:
            results[file.filename] = f"Ошибка: {str(e)}"
    output_filename = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        for filename, content in results.items():
            f.write(content)
            f.write("\n\n" + ("-" * 50) + "\n\n")
    return FileResponse(output_filename, filename=output_filename)


if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(8000)  # Настраиваем тунель для доступа и запускаем сервер
    print(f"Public URL: {ngrok_tunnel.public_url}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
