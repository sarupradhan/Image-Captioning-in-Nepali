from flask import Flask, render_template,request

from googletrans import Translator
import os
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

import cv2
import numpy as np


import numpy as np
from tqdm.notebook import tqdm # how much data is process till now
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input # extract features from image data.
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input , Dense , LSTM , Embedding , Dropout , add





def idx_to_word(integer, tokenizer):
      for word, index in tokenizer.word_index.items():
            if index == integer:
                  return word
            return None





def predict_caption(model, image, tokenizer, max_length):
      # add start tag for generation process
      in_text = 'startseq'
      # iterate over the max length of sequence
      for i in range(max_length):
            # encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad the sequence
            sequence = pad_sequences([sequence], max_length)
            # predict next word
            yhat = model.predict([image, sequence], verbose=0)
            # get index with high probability
            yhat = np.argmax(yhat)
            # convert index to word
            word = idx_to_word(yhat, tokenizer)
            # stop if word not found
            if word is None:
                  break
            # append word as input for generating next word
            in_text += " " + word
            # stop if we reach end tag
            if word == 'endseq':
                  break

      # Remove startseq and endseq tokens
      in_text = in_text.replace('startseq', '').replace('endseq', '').strip()

      # Translate the predicted caption from English to Nepali
      translator = Translator()
      translated_caption = translator.translate(in_text, src='en', dest='ne').text

      return translated_caption


#-----------------------------------------------------------flask---------------------------------------
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def Home():
      return render_template('index.html')



@app.route('/caption',methods=['GET', 'POST'])
def caption():

      # Load model
      saved_model_path = 'C:\\Users\\LEVEL51PC\\Desktop\\imageCaptionWeb\\imgmodel.h5'  # Adjust the file name as needed
      model = load_model(saved_model_path)

      # Load model
      saved_model_path = 'C:\\Users\\LEVEL51PC\\Desktop\\imageCaptionWeb\\vgg16_model.h5'  # Adjust the file name as needed
      vgg16_model = load_model(saved_model_path)

      # load Features and tokenizers
      tokenizer_path = 'C:\\Users\\LEVEL51PC\\Desktop\\imageCaptionWeb\\tokenizer.pkl'
      with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)

      max_length_path = 'C:\\Users\\LEVEL51PC\\Desktop\\imageCaptionWeb\\max_length.txt'
      with open(max_length_path, 'r') as f:
            max_length = int(f.read())

      with open(os.path.join("C:\\Users\\LEVEL51PC\\Desktop\\imageCaptionWeb", 'imgfeatures.pkl'), 'rb') as f:
            features = pickle.load(f)
      
      image_folder = 'demo'
      image_path = os.path.abspath(image_folder)
      
      # load image
      image = load_img(image_path, target_size=(224, 224))
      # convert image pixels to numpy array
      image = img_to_array(image)

      # reshape data for model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      # preprocess image for vgg
      image = preprocess_input(image)
      # extract features
      feature = vgg16_model.predict(image)
      print(feature)
      # predict from the trained model
      caption=predict_caption(model, feature, tokenizer, max_length)
      image = Image.open(image_path)
      print(caption)
      file = request.files['file1']
      filename = secure_filename(file.filename)
      file_path = os.path.join('image_path', filename)
      file.save(file_path)

      
      # Example: Processing an image using OpenCV
      img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

      return render_template('caption.html')

if __name__ == "__main__":
      app.run(debug = True)

