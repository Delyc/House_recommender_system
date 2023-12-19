import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Downloading NLTK stopwords for language processing
nltk.download('stopwords')

# Downloading NLTK wordnet for lexical analysis
nltk.download('wordnet')


# Defining a function to remove punctuation from text
PUNCTUATION = string.punctuation

def remove_punctuation(text: str):
    """
    Removes punctuation from the input text.

    Parameters:
    - text (str): Input text containing punctuation.

    Returns:
    - str: Text with punctuation removed.
    """
    return text.translate(str.maketrans('', '', PUNCTUATION))

# Defining a function to remove URLs from text
def remove_urls(text):
    """
    Removes URLs from the input text.

    Parameters:
    - text (str): Input text containing URLs.

    Returns:
    - str: Text with URLs removed.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# Creating a WordNet Lemmatizer instance
lemmatizer = WordNetLemmatizer()

# Defining a function to lemmatize text
def lemmatize_text(text):
    """
    Lemmatizes the words in the input text.

    Parameters:
    - text (str): Input text to be lemmatized.

    Returns:
    - str: Lemmatized text.
    """
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """
    Removes stopwords from the input text.

    Parameters:
    - text (str): Input text containing stopwords.

    Returns:
    - str: Text with stopwords removed.
    """
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])