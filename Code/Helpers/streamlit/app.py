import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image, ImageOps
from streamlit_image_select import image_select
from tensorflow.keras.models import load_model
import zipfile
import tensorflow_hub as hub
import time

# function to load and cache pretrained model
@st.cache_resource
def load_model_stream():
    path = "../../../Models/tiny_conv_model2"
    model = load_model(path)
    return model

# function to preprocess an image and get a prediction from the model
def get_prediction(model, image):
    
    open_image = Image.open(image)
    resized_image = open_image.resize((256, 256))
    np_image = np.array(resized_image)
    reshaped = np.expand_dims(np_image, axis=0)

    predicted_prob = model.predict(reshaped)[0][0]
    
    if predicted_prob >= 0.5:
        return f"Not hot dog, Confidence: {str(predicted_prob)[:4]}"
    else:
        return f"Hot dog, Confidence: {str(1 - predicted_prob)[:4]}"

def upload_mode():

  st.header("Detector Mode")
  st.subheader("Upload an Image to Make a Prediction")

  # upload an image
  uploaded_image = st.file_uploader("Upload your own image to test the model:", type=['jpg', 'jpeg'])

  # when an image is uploaded, display image and run inference
  if uploaded_image is not None:
    st.image(uploaded_image)
    st.text(get_prediction(classifier, uploaded_image))

def model_compare():

  st.header("Comparing Models")
  st.subheader("Our models comparisons!")
  st.image("../../../Images/accuracy_AUC.png")
  st.image("../../../Images/Precision_Recall.png")

st.set_page_config(layout="wide")

# load model
classifier = load_model_stream()

st.title("Hot Dog Checker")

st.write('Use the sidebar to select a page to view.')

page = st.sidebar.selectbox('Select Mode',['Upload Image','Model Comparison']) 

if page == 'Model Comparison':
  model_compare()
else:
  upload_mode()



   










