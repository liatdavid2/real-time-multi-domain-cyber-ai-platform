FROM python:3.10-slim

WORKDIR /app

# Prevent python buffering (better logs)
ENV PYTHONUNBUFFERED=1

# Install system deps (optional but useful)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 1. copy ONLY requirements (for caching)
COPY inference/requirements.txt .

# 2. install python deps
RUN pip install --no-cache-dir -r requirements.txt

# 3. copy code AFTER install
COPY inference /app
COPY training /app/training

# expose port
EXPOSE 8000

# run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]