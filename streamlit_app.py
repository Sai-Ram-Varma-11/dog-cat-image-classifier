import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
from joblib import load
import os
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="🐱 Cat vs Dog Classifier 🐶",
    page_icon="🐾",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #8e44ad, #3498db);
        }
        .css-1v0mbdj.ebxwdo61 {
            border-radius: 20px;
            padding: 2rem;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        .css-10trblm.e16nr0p30 {
            color: white;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "svm_model.pkl")
    try:
        return load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_image(image):
    """Make prediction on an image using our SVM model"""
    model = load_model()
    
    if model is None:
        st.error("Model not loaded. Please check if svm_model.pkl exists.")
        return None
        
    # Convert PIL Image to cv2 format
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
    # Resize
    img_resized = cv2.resize(img_array, (64, 64))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features
    feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
              cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    
    # Predict
    pred = model.predict([feat])[0]
    return "Cat 🐱" if int(pred) == 0 else "Dog 🐶"

# Title
st.title("🐾 Cat vs Dog Classifier")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Make prediction
    with col2:
        with st.spinner("Analyzing image... 🔍"):
            prediction = predict_image(image)
            if prediction:
                st.success(f"Prediction: {prediction}")
                
                # Show confidence message
                if "Cat" in prediction:
                    st.markdown("I think this is a cute cat! 😺")
                else:
                    st.markdown("I think this is a good boy/girl! 🐕")
            
# Add some helpful instructions
st.markdown("""
---
### Instructions
1. Click the 'Browse files' button above
2. Select a cat or dog image from your computer
3. Wait for the prediction!

The classifier works best with clear, front-facing photos of cats and dogs.
""")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and scikit-learn")