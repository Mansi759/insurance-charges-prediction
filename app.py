
import sklearn
import streamlit as st
import pickle
import numpy as np


# Function to load pickled files
def load_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load encoders and model
encoder_sex = load_pickle("encoder_sex.pkl")
encoder_smoker = load_pickle("encoder_smoker.pkl")
encoder_region = load_pickle("encoder_region.pkl")
ridge_model = load_pickle("ridge_model.pkl")

# Streamlit UI
st.title("🚑 Insurance Charges Prediction")
st.write("Enter the details below to predict the insurance charges.")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)

# Categorical Inputs
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# ✅ Function to Encode Inputs
def encode_input(encoder, value):
    try:
        return encoder.transform([value])[0]  # LabelEncoder case
    except:
        return encoder.transform([[value]]).toarray()[0]  # OneHotEncoder case

# Encode categorical variables correctly
encoded_sex = encode_input(encoder_sex, sex)
encoded_smoker = encode_input(encoder_smoker, smoker)
encoded_region = encode_input(encoder_region, region)

# Convert all features into a single array
input_data = np.hstack(([age, bmi, children], encoded_sex, encoded_smoker, encoded_region)).reshape(1, -1)

# Predict button
if st.button("🔮 Predict Insurance Charges"):
    prediction = ridge_model.predict(input_data)
    st.success(f"💰 Predicted Insurance Charges: **${prediction[0]:,.2f}**")
