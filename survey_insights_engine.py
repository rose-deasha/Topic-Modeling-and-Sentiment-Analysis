import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary nltk data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
domain_stop_words = {'thank', 'please', 'feel', 'best', "I've", "ive", "I'm", "im", 'regards'}
stop_words.update(domain_stop_words)

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if len(token) > 2]
    return ' '.join(tokens)

# Assume that the file path will be provided later
survey_responses = pd.read_csv('your/file/path.csv')

# Replace any infinity values with NaN in the data
survey_responses.replace([np.inf, -np.inf], np.nan, inplace=True)

# Example of how to preprocess the feedback text
processed_responses = [preprocess_text(response) for response in survey_responses['Feedback']]
print(processed_responses)

# TF-IDF Vectorizer
def create_tfidf(processed_responses):
    tfidf = TfidfVectorizer(max_df=0.85, min_df=2, ngram_range=(1, 3))
    doc_term_matrix = tfidf.fit_transform(processed_responses)
    return doc_term_matrix, tfidf.get_feature_names_out()

# Function to print topics
def print_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

# LDA topic modeling
def perform_lda(doc_term_matrix, num_topics):
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_output = lda_model.fit_transform(doc_term_matrix)
    return lda_model, lda_output

# WordCloud generation
def create_wordcloud(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        top_words = {feature_names[i]: topic[i] for i in topic.argsort()[:-num_top_words - 1:-1]}
        wordcloud = WordCloud(background_color='white').generate_from_frequencies(top_words)
        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_idx + 1} Word Cloud')
        plt.savefig(f'topic_{topic_idx+1}_wordcloud.png')
        plt.show()

# Sentiment analysis with VADER
sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = sia.polarity_scores(text)
    return score['compound']

# Example of how to analyze sentiment
vader_sentiments = [vader_sentiment(response) for response in survey_responses['Feedback']]
plt.figure(figsize=(12, 6))
sns.histplot(vader_sentiments, bins=20, edgecolor='black')
plt.title('Sentiment Distribution of Survey Responses (VADER)')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.savefig('vader_sentiment_distribution.png')
plt.show()
