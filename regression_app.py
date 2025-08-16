# STREAMLIT APP FOR ANN REGRESSION MODEL PREDICTING ESTIMATED SALARY

import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# LOad the trained model
model=tf.keras.models.load_model("Regression_model.h5")

# Load the encoders and scaler
with open ("label_encoder_gender.pkl",'rb') as file:
    label_encoder_gender=pickle.load(file)
with open("oneHotencode_geo.pkl",'rb') as file:
    oneHotencode_geo = pickle.load(file)
with open ("scaler_reg.pkl", 'rb') as file:
    scaler_reg= pickle.load(file)

## STREAMLIT APP

st.title("CUSTOMER SALARY PREDICTION")

geography=st.selectbox("Geography", oneHotencode_geo.categories_[0])
age=st.slider("Age", 18,92)
gender=st.selectbox('Gender', label_encoder_gender.classes_)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit score")
exited=st.selectbox("Exited",[0,1])
tenure=st.slider("Tenure",0,10)
num_of_products= st.slider("Number of products", 1,4)
has_cr_card= st.selectbox("Has credit card?",[0,1])
is_active_member=st.selectbox("Is active member?", [0,1])

# Prepare the input data
input_data= pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[exited]
})

# One-hot encode Geography column

geo_encoded= oneHotencode_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=oneHotencode_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data= pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled= scaler_reg.transform(input_data)

# PRedict the estimated salary
prediction = model.predict(input_data_scaled)
predicted_salary=prediction[0][0]

st.write(f"Predicted estimated salary : $ {predicted_salary:.2f}")