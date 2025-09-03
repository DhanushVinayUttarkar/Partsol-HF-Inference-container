# App container: FastAPI + Gunicorn + HF pipelines
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
RUN pip install --upgrade pip
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy app source
COPY app /app/app

EXPOSE 8000

# Env worker count
ENV WORKERS=2

# Start Gunicorn with Uvicorn workers
CMD ["/bin/sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -w ${WORKERS:-2} -b 0.0.0.0:8000 app.main:app --timeout 180"]
