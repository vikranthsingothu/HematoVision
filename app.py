from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the model
model = load_model('model.h5') # Make sure your model is trained on (224, 224, 3)

# Class names (adjust based on your model's output)
class_names = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0 # Normalize
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]

            return render_template('result.html', prediction=predicted_class, image_url=url_for('static', filename=f'uploads/{filename}'))

        except Exception as e:
            return f"Error processing image: {str(e)}"

    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
