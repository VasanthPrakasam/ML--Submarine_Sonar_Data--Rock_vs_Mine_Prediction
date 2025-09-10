import streamlit as st
import numpy as np
import pickle

# Load the trained Logistic Regression model
with open('Support_Vector_Machine_best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app layout
st.title("SONAR Mine vs Rock Prediction")
st.write("Enter the SONAR Readings to predict whether it's a Mine or a Rock.")

input_data = st.text_input('Enter comma-separated SONAR Reading Values here')
# When the user clicks the predict button
if st.button('Predict'):
    # Prepare input data
    input_data_np_array = np.asarray(input_data.split(','), dtype=float)
    reshaped_input = input_data_np_array.reshape(1, -1)
    # Predict and show result
    prediction = model.predict(reshaped_input)
    if prediction[0] == 'R':
        st.write('This Object is Rock')
    else:
        st.write('The Object is Mine')
        
# 0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111, 0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797, 0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857, 0.1307,0.2604,0.5122,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744, 0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324, 0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0032,0.0044