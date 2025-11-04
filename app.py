from flask import Flask, render_template, request, flash
import cv2
import numpy as np
from skimage.feature import hog
from joblib import load
import os
from werkzeug.utils import secure_filename
import tempfile

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev_secret_key")

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "svm_model.pkl")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}
IMG_SIZE = (64, 64)

# Load model once at startup
print(f"Attempting to load model from: {MODEL_PATH}")
try:
    model = load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"ERROR: Could not load model from {MODEL_PATH}")
    print(f"Error details: {str(e)}")


def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def predict_image(image_path):
    print(f"\nPrediction steps for: {image_path}")
    
    if model is None:
        raise RuntimeError("Model not loaded - check earlier load errors")

    print("Reading image...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from: {image_path}")

    print(f"Resizing image from {img.shape} to {IMG_SIZE}...")
    img_resized = cv2.resize(img, IMG_SIZE)
    
    print("Converting to grayscale...")
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    print("Extracting HOG features...")
    feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    
    print("Running model prediction...")
    pred = model.predict([feat])[0]
    result = "Cat 🐱" if int(pred) == 0 else "Dog 🐶"
    print(f"Prediction result: {result}")
    
    return result


@app.route('/health')
def health():
    """Simple health endpoint to check model availability."""
    return {"model_loaded": model is not None}


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        print("Form submitted with POST")
        
        if 'file' not in request.files:
            print("No file in request.files")
            flash('Please select a file')
            return render_template('index.html', error="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            print("Empty filename submitted")
            flash('No file selected')
            return render_template('index.html', error="No file selected")

        if file and allowed_file(file.filename):
            try:
                print(f"Processing file: {file.filename}")
                
                # Check if model is loaded
                if model is None:
                    raise RuntimeError("Model is not loaded. Please check if svm_model.pkl exists.")
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.'+file.filename.rsplit('.', 1)[1].lower(), delete=False) as tmp:
                    print(f"Saving to temp file: {tmp.name}")
                    file.save(tmp.name)
                    
                    if not os.path.exists(tmp.name):
                        raise IOError("Failed to save uploaded file")
                        
                    # Run prediction
                    print("Running prediction...")
                    label = predict_image(tmp.name)
                    print(f"Prediction result: {label}")
                    
                    # Clean up
                    try:
                        os.unlink(tmp.name)
                    except:
                        pass  # Ignore cleanup errors
                    
                if not label:
                    raise ValueError("Could not determine prediction")
                    
                return render_template('index.html', label=label)
                                     
            except Exception as e:
                error_msg = str(e)
                print(f"Error during prediction: {error_msg}")
                flash("Sorry, there was an error processing your image. Please try again.")
                return render_template('index.html', error=error_msg)
        else:
            flash('Unsupported file type')

    return render_template('index.html', filename=None, label=None)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    # When running locally, show a helpful message if model missing
    if model is None:
        print("Model not found. Place your trained svm_model.pkl next to app.py")
    try:
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting Flask: {e}")
        import traceback
        traceback.print_exc()
