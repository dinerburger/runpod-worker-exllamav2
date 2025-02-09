import os
import logging
import glob
import json
from huggingface_hub import snapshot_download
from utils import timer_decorator

BASE_DIR="/"
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", "/runpod-volume/")

@timer_decorator
def download(name, revision, cache_dir):
    return snapshot_download(name, revision=revision, cache_dir=cache_dir, local_dir=f"{MODEL_BASE_PATH}/{name}")

if __name__ == "__main__":
    cache_dir = os.getenv("HF_HOME")
    model_name, model_revision = os.getenv("MODEL_NAME"), os.getenv("MODEL_REVISION") or None
    cache_type = os.getenv("KV_CACHE_QUANT", "FP16")
    model_path = download(model_name, model_revision, cache_dir)   
    metadata = {
        "MODEL_NAME": model_name,
        "MODEL_REVISION": model_revision,
        "KV_CACHE_QUANT": cache_type
    }   
    
    with open(f"{BASE_DIR}/local_model_args.json", "w") as f:
        json.dump({k: v for k, v in metadata.items() if v not in (None, "")}, f)