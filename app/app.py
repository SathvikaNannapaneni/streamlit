import streamlit as st
import pickle
import numpy as np

# Load the trained logistic regression model
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Logistic Regression Prediction App")

# Add user inputs for each feature
mean_radius = st.number_input("Mean Radius")
mean_texture = st.number_input("Mean Texture")
mean_perimeter = st.number_input("Mean Perimeter")
mean_area = st.number_input("Mean Area")

# Prepare input data as a numpy array
input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area]])

# Predict the output
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.write("The predicted class is: Malignant")
    else:
        st.write("The predicted class is: Benign")