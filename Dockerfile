FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends     gcc     && rm -rf /var/lib/apt/lists/*

# Копируем зависимости первыми для кэширования
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Делаем start.sh исполняемым
RUN chmod +x start.sh

# Railway сам подхватит startCommand из railway.toml,
# но на всякий случай укажем ENTRYPOINT
ENTRYPOINT ["./start.sh"]
