Partsol – Hugging Face Inference Container (NGINX + Gunicorn + FastAPI)

Hello and welcome to the demo on **Hugging Face Inference Container** with parallelism to handle multiple requests

It supports **multiple parallel requests** via **Gunicorn** (multi-worker) behind **NGINX**.  
Includes:
- A minimal UI to test the model's task handling without any code.
- A **POST API** (`/api/infer` and `/api/infer-image`) proxied through NGINX.
- A **Jupyter notebook** (`notebooks/client_demo.ipynb`) that sends requests and prints responses.

---

## 1) Prerequisites (Windows)

- **Docker Desktop** (WSL2 backend recommended)

> If ports 8080 or 8000 are taken on your machine, adjust in the compose file.

---

## 2) Quick Start

Open **PowerShell** in this project folder and run:

```powershell
docker compose up --build
```

Wait for the images to build and both containers to start.  
Then open: **http://localhost:8080** for the UI.

**Health checks:**
- UI: http://localhost:8080/
- API: http://localhost:8080/api/infer (POST)
- Health: http://localhost:8080/api/health (GET: 200 OK)

Stop the stack with:
```powershell
docker compose down
```

---

## 3) How it works (High level)

- **FastAPI app** (Python) exposes:
  - `POST /infer` for text tasks (text-classification, text-generation, summarization, question-answering, fill-mask, token-classification).
  - `POST /infer-image` for image-classification with file upload.
  - Pipelines are **cached** in-process keyed by `(task, model_id)` for speed, with a **Lock** per pipeline for safe use.
- **Gunicorn** starts multiple **Uvicorn** workers (ASGI), enabling **parallel request handling**.
- **NGINX** sits in front, serving the static UI and proxying `/api/*` to the app. This also gives you an easy place to add TLS, rate limits, etc.
- **HF model cache** is persisted in a Docker **volume** (`hf-cache`) so models are downloaded once and reused across restarts.


## 4) Why the distilbert-base-uncased-finetuned-sst-2-english model?

The default for text classification is **`distilbert-base-uncased-finetuned-sst-2-english`**:
- **Small and fast**: ideal for CPU-only containers (since not every system has access to a GPU).
- **Well-known** and **widely benchmarked**: reviewers on HF recognize its behavior.
- **Clear, easy-to-interpret outputs**: It make demos self-explanatory and easy to understand (Positive/Negative).

We also demonstrate **`gpt2`** for trext-generation and **`google/vit-base-patch16-224`** for image classification to show **multi-task flexibility**.

---

## 6) Change concurrency

By default the app runs with 2 Gunicorn workers. Increase by setting `WORKERS`:

```yaml
# docker-compose.yml
services:
  app:
    environment:
      - WORKERS=4
```

Then:
```bash
docker compose up --build
```

Each worker keeps its own pipeline cache. NGINX can also be scaled horizontally with multiple `app` replicas behind it.

---

## 7) Project Structure

```
.
├── app/
│   └── main.py                # FastAPI app + pipeline cache
├── nginx/
│   └── nginx.conf             # Serves UI & proxies /api/* to app
├── notebooks/
│   └── client_demo.ipynb      # Requests demo
├── ui/
│   └── index.html             # Minimal UI
├── Dockerfile                 # App image
├── requirements.txt
└── docker-compose.yml         # NGINX + app + HF cache volume
```
