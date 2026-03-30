# syntax=docker/dockerfile:1

FROM python:3.10-slim

WORKDIR /app

# copy only requirements
COPY inference/requirements.txt .

# install deps with cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --prefer-binary

# copy code AFTER install
COPY inference /app
COPY training /app/training

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]