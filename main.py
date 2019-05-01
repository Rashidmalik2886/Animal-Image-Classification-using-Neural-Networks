
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/classifier.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')



def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
                return key

def model_predict(path ,model):
    test_image = image.load_img(path, 
                            target_size = (124,124))
    test_image = image.img_to_array(test_image)
    test_image/=255
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    answer = np.argmax(result)
    my_dict = {'Cat': 0, 'Cow': 1, 'Dear': 2, 'Dog': 3, 'Elephant': 4, 'Goat': 5, 'Horse': 6, 'Lion': 7, 'Panda': 8, 'Sheep': 9, 'Wolf': 10}
    print(result)
    new_test = result[0]
    print(new_test)
    act_value = max(new_test)
    print(act_value)
    val_index = new_test.argmax()
    result = get_key(val_index, my_dict)
    if act_value > 0.7:
        result = result
    elif act_value > 0.4:
        result = "Not Sure but may be " + result
    else:
        result = "Sorry, Can't recognise"
#print("result")
    return result

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.abspath("")
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)


        return result
    return None

if __name__ == '__main__':
    app.run(port=5002, debug=True,use_reloader=False)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
