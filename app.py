from flask import Flask, render_template, request
from werkzeug.utils import secure_filename  # Add this import
import numpy as np
import os
import uuid
from flask import send_from_directory
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load your trained model
model = tf.keras.models.load_model("model.h5")

# Define the disease classes with their names and information
DISEASE_CLASSES = {
    0: {'name': 'Coccidiosis', 'info': 'A parasitic disease affecting the intestinal tract.'},
    1: {'name': 'Healthy', 'info': 'No signs of disease detected.'},
    2: {'name': 'Newcastle Disease', 'info': 'A highly contagious viral disease affecting birds.'},
    3: {'name': 'Salmonella', 'info': 'A bacterial infection causing digestive problems.'}
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

    
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'poultry_image' not in request.files:
            return render_template('predict.html', error="No file part")

        file = request.files['poultry_image']
        if file.filename == '':
            return render_template('predict.html', error="No selected file")

        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4().hex}{ext}"  # Unique filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
        pred = model.predict(img_array)
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = float(np.max(pred, axis=1)[0])
                
                # Get disease info
        disease = DISEASE_CLASSES.get(pred_class, {'name': 'Unknown', 'info': ''})
                
        return render_template('predict.html', 
                                    prediction=disease['name'],
                                    confidence=f"{confidence*100:.2f}%",
                                    info=disease['info'],
                                    image_path=filename)
    
    return render_template('predict.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)