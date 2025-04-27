from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

import tempfile
from werkzeug.utils import secure_filename


# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = load_model('asl_resnet50_model.h5')  # rename accordingly

# Define image size (must match your training image size)
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Class labels (ensure the order matches your training generator class_indices)
class_labels = ['O', 'R']

# Route for home page (upload form)
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400
    
    file = request.files['file']

    if file.filename == '':
        return "No selected file!", 400

    if file:
        # Create a temporary file using tempfile
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_path = temp_file.name
            # Save the uploaded file content to the temp file
            file.save(temp_file)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # rescaling like during training
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = round(np.max(predictions) * 100, 2)

        return render_template('result.html', 
                               prediction=predicted_class, 
                               confidence=confidence, 
                               image_path=file_path)

    return "Something went wrong!", 500

if __name__ == '__main__':
    app.run()