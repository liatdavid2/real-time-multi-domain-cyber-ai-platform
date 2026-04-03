FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app

# copy requirements
COPY training/requirements.txt ./requirements.txt

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy training code only (important!)
COPY training/ .

ENV GIT_PYTHON_REFRESH=quiet

ENTRYPOINT ["python", "run.py"]