FROM python:3.10-slim

WORKDIR /app

# copy only requirements first (important!)
COPY requirements.txt .

# install dependencies (cached unless requirements changed)
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of the code
COPY . .

CMD ["python", "retrain_watcher.py"]