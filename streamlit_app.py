import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

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


spam_data['transformed_text']=spam_data['text'].apply(transform_text)

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(spam_data['transformed_text']).toarray()

y = spam_data['target'].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

mnb = MultinomialNB()

mnb.fit(X_train,y_train)
y_pred1 = mnb.predict(X_test)

st.title("Email Spam Detector")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = mnb.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

