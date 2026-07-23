FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Скачиваем данные и обучаем модель при сборке (опционально, но можно сделать отдельно)
# Лучше заранее обучить локально и скопировать модели, чтобы не тратить время на сборку
RUN python src/data_loader.py && python src/train_model.py

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]