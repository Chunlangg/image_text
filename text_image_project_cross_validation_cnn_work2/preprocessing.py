import re
import emoji
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# download useful file
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def clean_text(text):
    """
    clear text 
    """
    if isinstance(text, float):
        # if it is float，transfer to text
        text = str(text)
    text = text.lower()  # change to lower case 
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove link
    text = emoji.demojize(text)     # change the emoje to text

    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation 
    text = re.sub(r'\d+', '', text)  # remove numbers 
    return text

def tokenize_and_remove_stopwords(text):
    """
    remove the stopwords 
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words]

def preprocess_data(X_train, X_test):
    """
    preprocessing for the X_train & X_test
    """
    X_train_processed = [' '.join(tokenize_and_remove_stopwords(clean_text(text))) for text in X_train]
    X_test_processed = [' '.join(tokenize_and_remove_stopwords(clean_text(text))) for text in X_test]
    return X_train_processed, X_test_processed
