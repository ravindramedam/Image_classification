import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os

#  Title
st.header('Image Classification Model')

#  Load model (set compile=False if you’re only predicting)
model = load_model(
    r'C:\Users\RAVI\OneDrive\Documents\OneDrive\Desktop\personals\RESUME\projects\image_claasifier\Image_classify.keras',
    compile=False
)

#  Categories
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic',
    'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion',
    'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato',
    'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
    'turnip', 'watermelon'
]

img_height = 180
img_width = 180

#  User input for image path
image_path = st.text_input('Enter image file path', 'Banana.jpg')

if os.path.exists(image_path):
    #  Load and prepare image
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, axis=0)  # Batch dimension

    #  Model prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])  # First batch element

    # Display result
    st.image(image_path, width=200)
    st.write(f'Veg/Fruit in image is: **{data_cat[np.argmax(score)]}**')
    st.write(f'Accuracy: **{np.max(score)*100:.2f}%**')

else:
    st.error("❌ File not found. Please check the path or filename.")