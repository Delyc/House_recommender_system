
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from scipy.sparse import hstack

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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

# STOPWORDS = set(stopwords.words('english'))
# def remove_stopwords(text):
#     """
#     Removes stopwords from the input text.

#     Parameters:
#     - text (str): Input text containing stopwords.

#     Returns:
#     - str: Text with stopwords removed.
#     """
#     return " ".join([word for word in str(text).split() if word not in STOPWORDS])

class HouseRecommenderSystem:
    def __init__(self, df) -> None:
        self.df_original = df
        self.df_processed = None
        self.cosine_similarity = None
        # Defining a list of categorical features in the dataset
        self._categorical_features = ['city', 'homeType']
    
    def preprocess(self, df):
        # Dropping specified columns from the DataFrame
        df = df.drop(['homeImage', 'latestPriceSource', 'numOfPhotos', 'zpid', 'latest_saledate', 'streetAddress', 'latitude', 'longitude'], axis=1)

        # Removing duplicate rows from the DataFrame
        df.drop_duplicates(inplace=True)

        df['description'] = df['description'].fillna('')

        # Applying text preprocessing steps to the 'description' column
        df['description'] = df['description'].apply(remove_punctuation)
        # df['description'] = df['description'].apply(remove_stopwords)
        df['description'] = df['description'].apply(remove_urls)
        df['description'] = df['description'].apply(lemmatize_text)

        # Scaling numerical features using StandardScaler
        numerical_features = df.select_dtypes(exclude=['object', 'bool']).columns
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

        # Encoding categorical features using LabelEncoder
        for column in self._categorical_features:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
        
        return df

    def fit(self):
        self.df_processed = self.preprocess(self.df_original)


        # Creating a TF-IDF Vectorizer with specified parameters
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')

        # Filling null values in the 'description' column and transforming it using TF-IDF
        self.df_processed['description'] = self.df_processed['description'].fillna('')
        tfidf_matrix = tfidf.fit_transform(self.df_processed['description'])

        # Creating a DataFrame without the 'description' column
        non_text_df = self.df_processed.drop('description', axis=1)

        # Converting the non-text DataFrame to a sparse format
        sdf = non_text_df.astype('Sparse[int64, 0]')

        # Combining the sparse non-text DataFrame with the TF-IDF matrix
        combined_sdf = hstack([sdf, tfidf_matrix])

        # Calculating cosine similarity using linear kernel on the combined sparse matrix
        self.cosine_similarity = linear_kernel(combined_sdf, combined_sdf)
    
    def recommend(self, house_id):
            # Creating a list of tuples containing house indices and their cosine similarity scores
        sim_scores = list(enumerate(self.cosine_similarity[house_id - 1]))
        # Sorting the list based on cosine similarity in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Selecting the top 5 similar houses
        sim_scores = sim_scores[1:10]

        # print("sim scores: ", sim_scores)

        # Extracting the indices of the recommended houses
        house_indices = [x[0] for x in sim_scores]
        # Returning the original DataFrame with the recommended houses
        return self.df_original.iloc[house_indices], sim_scores



