import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


## Load the dataset and word index
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

## Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

## function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


## function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

##Streamlit app
import streamlit as st

st.title("Movie Review Sentiment Analysis")

# User input
user_input = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    preprocess_input = preprocess_text(user_input)
    prediction = model.predict(preprocess_input)
    st.write("Sentiment:", "Positive" if prediction[0][0] > 0.5 else "Negative")
    st.write("Prediction Score:", prediction[0][0])
else:
    st.write("Please enter a review.")
