import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nlkt

spam_data=pd.read_csv('https://raw.githubusercontent.com/Amrit445/email_classifier/master/spam.csv')

spam_data.rename(columns={'Category':'target','Message':'text'},inplace=True)

encoder=LabelEncoder()
spam_data['target']=encoder.fit_transform(spam_data['target'])

spam_data=spam_data.drop_duplicates(keep='first')


