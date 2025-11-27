import streamlit as st
import numpy as np
import pandas as pd
import joblib

#Lets load the joblib instances over here
with open('pipeline.joblib','rb') as file:
    preprocess = joblib.load(file)

with open('model.joblib','rb') as file:
    model = joblib.load(file)


#Lets take the inpput from the user
st.title('Help NGO Organization')
st.subheader('This application will help to identify the development category of the country using socio economic factors.Original data has been clustered using KMeans.')