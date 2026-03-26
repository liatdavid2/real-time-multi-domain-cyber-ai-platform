FROM python:3.10-slim

WORKDIR /app

COPY monitoring/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY monitoring /app

CMD ["python", "app.py"]