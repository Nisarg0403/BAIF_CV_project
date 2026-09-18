import sys
import os
import psutil
import gc
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

process = psutil.Process(os.getpid())
initial_rss = process.memory_info().rss / (1024 * 1024)
print(f"Initial process memory: {initial_rss:.2f} MB")

from backend.main import app
from backend.inference import process_image_file

startup_rss = process.memory_info().rss / (1024 * 1024)
print(f"Startup process memory (after app import, before first request): {startup_rss:.2f} MB")

img_path = os.path.join(PROJECT_ROOT, "data", "BAIF_IMAGES", "105730429112", "105730429112_LL_1.jpg")
with open(img_path, "rb") as f:
    img_bytes = f.read()

print("\n--- FIRST REQUEST ---")
res1 = process_image_file(img_bytes, side_name="Left Side Profile", model_engine="deeplab")
post_req1_rss = process.memory_info().rss / (1024 * 1024)
print(f"Process memory after Request #1: {post_req1_rss:.2f} MB")

print("\n--- SECOND REQUEST (Re-using cached models) ---")
res2 = process_image_file(img_bytes, side_name="Left Side Profile", model_engine="deeplab")
post_req2_rss = process.memory_info().rss / (1024 * 1024)
print(f"Process memory after Request #2: {post_req2_rss:.2f} MB")

gc.collect()
post_gc_rss = process.memory_info().rss / (1024 * 1024)
print(f"Process memory after GC collect: {post_gc_rss:.2f} MB")
