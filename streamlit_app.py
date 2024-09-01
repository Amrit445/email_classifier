import streamlit as st
import pandas as pd

st.title('E-MAIL SPAM DETECTION')

df=pd.read_csv('https://raw.githubusercontent.com/Amrit445/email_classifier/master/spam.csv')

print(df.shape)

df.drop_duplicates(inplace=True)
print(df.shape)
