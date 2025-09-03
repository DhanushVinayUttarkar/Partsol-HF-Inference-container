from typing import Any, Dict, Optional, Tuple, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline, Pipeline  # Hugging face model
from PIL import Image
from io import BytesIO
from threading import Lock

# ---- FastAPI app ----
app = FastAPI(title="Partsol HF Inference Container", version="1.0.0")

try:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

# Tasks
SUPPORTED_TASKS = {
    "text-classification",
    "text-generation",
    "summarization",
    "question-answering",
    "fill-mask",
    "token-classification",
}

# default for demo
DEFAULT_MODELS = {
    "text-classification": "distilbert-base-uncased-finetuned-sst-2-english",
    "text-generation": "gpt2",
    "summarization": "facebook/bart-large-cnn",
    "question-answering": "distilbert-base-cased-distilled-squad",
    "fill-mask": "distilroberta-base",
    "token-classification": "dslim/bert-base-NER",
    "image-classification": "google/vit-base-patch16-224",
}

class InferenceRequest(BaseModel):
    task: str = Field(..., description="HF pipeline task (e.g. 'text-classification', 'summarization', etc.)")
    inputs: Any = Field(..., description="The input for the model (string, list of strings, or dict depending on task).")
    model_id: Optional[str] = Field(None, description="Hugging Face model id; defaults to a small, fast model for the task.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Optional generation/classification params.")

class ModelCache:
    """Simple per-process pipeline cache with per-entry locks for safe concurrent use."""
    def __init__(self):
        self._cache: Dict[Tuple[str, str], Tuple[Pipeline, Lock]] = {}
        self._global_lock = Lock()

    def get(self, task: str, model_id: Optional[str]) -> Tuple[Pipeline, Lock]:
        if task not in SUPPORTED_TASKS and task != "image-classification":
            raise HTTPException(status_code=400, detail=f"Unsupported task '{task}'. Supported: {sorted(SUPPORTED_TASKS)}")

        resolved_model = model_id or DEFAULT_MODELS.get(task)
        if not resolved_model:
            raise HTTPException(status_code=400, detail=f"No default model configured for task '{task}', please provide model_id.")

        key = (task, resolved_model)
        if key in self._cache:
            return self._cache[key]

        with self._global_lock:
            if key in self._cache:
                return self._cache[key]
            try:
                pipe = pipeline(task=task, model=resolved_model, device=-1)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load pipeline for task='{task}', model='{resolved_model}': {e}")
            lock = Lock()
            self._cache[key] = (pipe, lock)
            return self._cache[key]

model_cache = ModelCache()

@app.get("/")
def root():
    return {"message": "Partsol HF Inference Container - FastAPI", "health": "/health", "infer": "/infer", "infer_image": "/infer-image"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer")
def infer(req: InferenceRequest):
    task = req.task
    pipe, lock = model_cache.get(task, req.model_id)

    # The HF pipeline accepts various shapes; we pass through inputs & parameters.
    try:
        with lock:
            result = pipe(req.inputs, **req.parameters)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")

    return {"task": task, "model_id": req.model_id or DEFAULT_MODELS.get(task), "result": result}

@app.post("/infer-image")
async def infer_image(model_id: Optional[str] = None, file: UploadFile = File(...)):
    # Separate endpoint for image-classification due to file upload
    task = "image-classification"
    pipe, lock = model_cache.get(task, model_id)

    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        with lock:
            result = pipe(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")

    return {"task": task, "model_id": model_id or DEFAULT_MODELS.get(task), "result": result}
