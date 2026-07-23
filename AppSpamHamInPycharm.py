import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import re
nltk.download('stopwords')
nltk.download('punkt_tab')


# Function for text processing
def text_processing_f(text):
    # 1-convert to lower case
    text = text.lower()

    # 2-Tokenize the words
    text = nltk.word_tokenize(text)

    # remove the special characters
    text = [word for word in text if word.isalnum()]

    # 3-remove stopwords(eg a,is,the,for,on etc)
    # remove Punctuations(!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~)
    text = [word for word in text
            if word not in stopwords.words('english') and
            word not in string.punctuation]

    # 6-Stemming (eg. Dance,Dancing,danced convert to dance)
    text = [ps.stem(word) for word in text]

    return " ".join(text)

vc = joblib.load(open(r'C:/Users/Dell/PyCharmMiscProject/.venv/PROJECTS/vectorizerTfidf.joblib','rb'))
model = joblib.load(open(r'C:/Users/Dell/PyCharmMiscProject/.venv/PROJECTS/2nd-SpamHamEmailsTrainedModel.joblib','rb'))


st.title("Spam Email Detector App")
input_email = st.text_area("Enter or Paste the text of Email")

# Preprocess the
processed_input_text = text_processing_f(input_email)
# Vectorize
vector_input_text = vc.transform([processed_input_text])
# Predict
final_result = model.predict(vector_input_text)[0]
# Display
if st.button("Predict"):
    if final_result == 1:
        st.header("This is a Spam Email")
    else:
        st.header("This is not a Spam Email")