import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from skimage.feature import hog
from joblib import load

# ================= CONFIG =================
IMG_SIZE = (64, 64)
MODEL_PATH = "svm_model.pkl"
# ==========================================

# Load trained model
try:
    model = load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except:
    print("❌ Failed to load model!")
    exit()

# Extract features (must match training!)
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

# Predict uploaded image
def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    try:
        img = cv2.imread(file_path)
        if img is None:
            result_label.config(text="❌ Invalid image file!", fg="red")
            return

        img_resized = cv2.resize(img, IMG_SIZE)
        feat = extract_features(img_resized).reshape(1, -1)
        pred = model.predict(feat)[0]
        label = "🐱 Cat" if pred == 0 else "🐶 Dog"

        # Display uploaded image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((250, 250))
        img_tk = ImageTk.PhotoImage(img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk

        # Show prediction result
        result_label.config(text=f"Prediction: {label}", fg="lime")
    except Exception as e:
        result_label.config(text=f"⚠️ Error: {str(e)}", fg="red")

# ================= UI =================
root = tk.Tk()
root.title("🐾 Cat vs Dog Classifier 🐾")
root.geometry("420x550")
root.configure(bg="#2C2C2C")

title = tk.Label(root, text="🐾 Cat vs Dog Classifier 🐾", font=("Arial", 18, "bold"), bg="#2C2C2C", fg="gold")
title.pack(pady=15)

btn = tk.Button(root, text="📂 Upload Image", command=upload_image,
                font=("Arial", 14, "bold"), bg="gold", fg="black", cursor="hand2", relief="raised")
btn.pack(pady=10)

panel = tk.Label(root, bg="#444", width=250, height=250)
panel.pack(pady=20)

result_label = tk.Label(root, text="Upload an image to predict 🐾", font=("Arial", 14, "bold"),
                        bg="#2C2C2C", fg="#AAAAAA")
result_label.pack(pady=10)

footer = tk.Label(root, text="Made with ❤️ using SVM", font=("Arial", 10, "italic"),
                  bg="#2C2C2C", fg="#AAAAAA")
footer.pack(side="bottom", pady=10)

root.mainloop()
