import os
import json
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model and parameters
def load_model_and_params():
    try:
        # Load the model
        model_path = 'model/sports_classifier_model.keras'
        model = tf.keras.models.load_model(model_path)
        
        # Load the preprocessing parameters
        params_path = 'model/preprocessing_params.json'
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        print("Model and parameters loaded successfully!")
        return model, params
    except Exception as e:
        print(f"Error loading model or parameters: {e}")
        raise

# Load model and parameters at startup
print("Loading model and parameters...")
model, params = load_model_and_params()

# Get parameters
IMG_SIZE = params.get('image_size', 64)  # Default to 64 if not specified
MEAN = np.array(params['mean'])
STD = np.array(params['std'])
CLASS_NAMES = params['class_names']

@app.route('/')
def index():
    # Clear uploads folder to save space (optional)
    for file in os.listdir(UPLOAD_FOLDER):
        if file != '.gitkeep':  # Keep this file
            os.remove(os.path.join(UPLOAD_FOLDER, file))
    
    return render_template('index.html', class_names=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', class_names=CLASS_NAMES, error="No file part")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', class_names=CLASS_NAMES, error="No selected file")
    
    if file and allowed_file(file.filename):
        # Save the file with a unique name to avoid conflicts
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the image
            img = cv2.imread(filepath)
            if img is None:
                return render_template('index.html', class_names=CLASS_NAMES, 
                                      error="Could not read image. Please try another file.")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Preprocess
            img_preprocessed = (img_resized - MEAN) / STD
            img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
            
            # Make prediction
            predictions = model.predict(img_preprocessed)[0]
            predicted_class_idx = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = float(predictions[predicted_class_idx]) * 100
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_predictions = [
                (CLASS_NAMES[i], float(predictions[i]) * 100)
                for i in top_indices
            ]
            
            return render_template(
                'result.html',
                filename=filename,
                predicted_class=predicted_class,
                confidence=confidence,
                predictions=top_predictions
            )
        except Exception as e:
            return render_template('index.html', class_names=CLASS_NAMES, 
                                  error=f"Error processing image: {str(e)}")
    
    return render_template('index.html', class_names=CLASS_NAMES, 
                          error="Invalid file type. Please upload a JPG, JPEG or PNG image.")

if __name__ == '__main__':
    app.run(debug=True)