# -*- coding: utf-8 -*-
"""
export_onnx_model.py -- One-time export of the embedding model to ONNX format.

Run once in the arm64 chroot to save the exported model locally:
    sudo chroot /opt/arm64-env bash -c 'cd /home/pi/redis-robot && python3 export_onnx_model.py'

After this, the model loads from disk in ~2 seconds instead of re-exporting (~13 seconds).
"""
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "redis/langcache-embed-v3-small"
LOCAL_PATH = "/home/pi/redis-robot/models/langcache-embed-v3-small-onnx"

print(f"Exporting {MODEL_NAME} to ONNX...")
print(f"Save path: {LOCAL_PATH}")

# Load with ONNX backend (triggers export from PyTorch)
model = SentenceTransformer(MODEL_NAME, backend="onnx")

# Save the full model (including ONNX weights) locally
os.makedirs(LOCAL_PATH, exist_ok=True)
model.save(LOCAL_PATH)

# Verify it loads from local path
print("Verifying local load...")
model2 = SentenceTransformer(LOCAL_PATH, backend="onnx")
vec = model2.encode("test", normalize_embeddings=True)
print(f"Output dim: {len(vec)}")
print(f"Model saved to {LOCAL_PATH}")
print("Done. Update registry.py MODEL_NAME to this local path.")
