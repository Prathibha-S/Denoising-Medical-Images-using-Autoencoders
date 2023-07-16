from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import os
import tensorflow as tf
import cv2
import numpy as np
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.static_folder = 'static'
model = tf.keras.models.load_model('denoising_model/finetuned_denoising_autoencoder.h5')

def denoise_image(file):
    filename = secure_filename(file.filename)
    image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (120, 120))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)
    denoised_image = model.predict(image)
    denoised_image = (denoised_image[0, :, :, 0] * 255).astype('uint8')
    return denoised_image

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    file = request.files['imagefile']
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = denoise_image(file)
        img = Image.fromarray(img)
        img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'denoised_'+filename))
        return render_template('result.html', filename=filename)
    else:
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
