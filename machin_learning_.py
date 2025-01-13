# Step 4: Create a Streamlit App

import streamlit as st
import joblib

# Load the saved model
model = joblib.load('iris_model.pkl')

st.title('Iris Flower Prediction App')

# Input features
sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, step=0.5)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button('Predict'):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.write(f'Predicted Class: {prediction[0]}')
