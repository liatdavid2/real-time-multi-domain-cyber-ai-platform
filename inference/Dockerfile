FROM python:3.10-slim

WORKDIR /app

# 1. copy ONLY requirements
COPY inference/requirements.txt .

# 2. install deps (this will be cached)
RUN pip install --no-cache-dir -r requirements.txt

# 3. copy code AFTER install
COPY inference /app
COPY training /app/training

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]