from flask import Flask , request , jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from warnings import filterwarnings
filterwarnings('ignore')

model = load_model('digit_recognition_model.h5')

app = Flask(__name__)
@app.route('/' , methods = ['GET'])
def hello ():
    return jsonify({'message': 'Hello, AI OCR is running'})

@app.route('/predict' , methods = ['POST'])
def predict():
    img  = request.files.get('image')
    img = Image.open(img).convert('L')
    img = img.resize((28,28))
    img = np.array(img).reshape(1,28,28,1).astype('float32') / 255.0
    res = model.predict(img)   
    predicted_label = np.argmax(res)
    return jsonify({'predicted': int(predicted_label), 'confidence': float(np.max(res))})

if __name__ == '__main__':
    app.run(port = 5000 , debug = True , use_reloader = False)