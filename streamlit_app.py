import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
spam_data=pd.read_csv('https://raw.githubusercontent.com/Amrit445/email_classifier/master/spam.csv')

spam_data.rename(columns={'Category':'target','Message':'text'},inplace=True)

encoder=LabelEncoder()
spam_data['target']=encoder.fit_transform(spam_data['target'])

spam_data=spam_data.drop_duplicates(keep='first')

spam_data['num_characters']=spam_data['text'].apply(len)

spam_data['num_words']=spam_data['text'].apply(lambda x:len(nltk.word_tokenize(x)))

spam_data['num_sentences']=spam_data['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)

print(transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight ,k? I've cried enough today."))
