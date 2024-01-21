from flask import Flask, render_template, request
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import os
import cv2
import tensorflow as tf


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('about.html')

@app.route('/save', methods=['POST'])
def save():
    data_url = request.form['data']
    image_data = base64.b64decode(data_url.split(',')[1])
    
    
    img = Image.open(BytesIO(image_data))
    
    
    img_resized = img.resize((28, 28))
    gray_image = img_resized.convert('L')

    
    image_array = np.array(gray_image)

    
    
    lower_threshold = 80
    upper_threshold = 240

    
    
    darkened_pixels = np.where(
    (image_array >= lower_threshold) & (image_array <= upper_threshold),
    image_array // 10,  
    image_array  
    )

    
    darkened_image = Image.fromarray(darkened_pixels.astype(np.uint8))
    model = tf.keras.models.load_model('saved_mod<')


    
    darkened_image.save('drawing.png')
    img = cv2.imread('./drawing.png'.format(1))[:,:,0]
    img = np.invert(np.array([img]))

    prediction = model.predict(img)
    predicted_number = np.argmax(prediction)
    print(predicted_number)
    
    return str(predicted_number)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
