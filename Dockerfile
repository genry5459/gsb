FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей (если понадобятся компиляторы)
RUN apt-get update && apt-get install -y --no-install-recommends     gcc     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Не запускаем от root в продакшене
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

CMD ["python", "gold_micro_scalper_unified.py"]
