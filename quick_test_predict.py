"""Run a quick prediction using the same pipeline as the Flask app without starting the server.

The script finds the first image under archive/.../PetImages/(Cat|Dog) and runs predict_image from app.py.
"""
import os
import glob
import sys

from app import predict_image

ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), "archive")

# search for any jpg/png in Cat or Dog folder
patterns = [os.path.join(ARCHIVE_DIR, "**", "PetImages", "Cat", "*.*"),
            os.path.join(ARCHIVE_DIR, "**", "PetImages", "Dog", "*.*")]

candidates = []
for p in patterns:
    candidates.extend(glob.glob(p, recursive=True))

if not candidates:
    print("No sample images found under archive/.../PetImages/Cat or Dog. Place some images there and retry.")
    sys.exit(2)

sample = candidates[0]
print(f"Using sample image: {sample}")
try:
    label = predict_image(sample)
    print(f"Prediction: {label}")
except Exception as e:
    print(f"Error during prediction: {e}")
