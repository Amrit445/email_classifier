import streamlit as st
import pandas as pd


spam_data=pd.read_csv('https://raw.githubusercontent.com/Amrit445/email_classifier/master/spam.csv')

spam_data.rename(columns={'Category':'target','Message':'text'},inplace=True)
spam_data
