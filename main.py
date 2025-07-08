!pip
install - q
torch
transformers
accelerate
pymupdf
import fitz
import torch
from google.colab import files
from transformers import AutoModelForImageTextToText, AutoTokenizer

# Очень важно по максимуму чистить память, так как её критически не хватает
torch.cuda.empty_cache()


def clean_memory():
    torch.cuda.empty_cache()
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()
    print("Память GPU очищена")


clean_memory()

try:
    tokenizer = AutoTokenizer.from_pretrained("ChatDOC/OCRFlux-3B")
    model = AutoModelForImageTextToText.from_pretrained(
        "ChatDOC/OCRFlux-3B",
        device_map="balanced",
        offload_folder="./offload",
        torch_dtype=torch.float16
    )
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки: {e}")
    raise


# Пытаемся загрузить модель, если не получается отлавливаем исключение и красиво его выводим
# 99% ошибок на этом этапе в моём случае - CUDA out of memory

# Пока я выбираю текст из pdf с помощью библиотеки, а потом суммаризирую, но это неправильное использование данной
# Модели, но для теста сделал его, так как пока картинки не могу сделать по ресурсам
def process_pdf(pdf_path: str,
                question: str = "Извлеки ключевую информацию") -> str:  # По умолчанию просим найти важное
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"  # Считываем полный текст документа
        inputs = tokenizer(
            f"{question}\n\n{full_text}",
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        ).to(model.device)  # Так как инпут большой, описываю его отдельно
        with torch.no_grad():
            outputs = model.generate(
                **inputs,  # Выполняем запрос к модели
                max_new_tokens=200,
                temperature=0.3,
                do_sample=False
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Ошибка обработки: {str(e)}"  # Так же красиво отлавливаем ошибку, опять-таки чаще всего CUDA OOM


# Загрузка написана, как и весь код для colab
if __name__ == "__main__":
    print("Загрузите PDF-файл:")
    uploaded = files.upload()
    pdf_name = next(iter(uploaded))
    result = process_pdf(pdf_name, "Выдели основные тезисы документа и важные цифровые данные")
    print("\nРезультат:")
    print(result)
    torch.cuda.empty_cache()
    clean_memory()
