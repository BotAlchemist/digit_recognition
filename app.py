# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 18:56:55 2022

@author: Sumit
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import glob


IMAGE_SIZE=32
filename= 'digit_train.h5'



img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)
    #img= img.resize((IMAGE_SIZE*IMAGE_SIZE))#
    img.save("Unseen/test.jpg")
    
    all_files= glob.glob("Unseen/*")
    
    X_unseen=[]
    for i_ in all_files:
        image=  tf.keras.preprocessing.image.load_img(
            path= i_,
            color_mode='grayscale',
            target_size=(IMAGE_SIZE,IMAGE_SIZE),
            )
        
        image_arr= tf.keras.preprocessing.image.img_to_array(image).reshape(IMAGE_SIZE, IMAGE_SIZE)
        X_unseen.append(image_arr)
    
    X_unseen= np.array(X_unseen)
    
    # Scaling
    X_unseen = X_unseen / 255
    
    loaded_model= load_model(filename)
    
    y_predicted= loaded_model.predict(X_unseen)
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    
    st.write("Predicted: ", y_predicted_labels[0])
    st.write("Confidence: ", max(y_predicted[0])*100)
    
    
    


