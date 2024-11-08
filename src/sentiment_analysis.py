# src/sentiment_analysis.py

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Scarica il lexicon di VADER
nltk.download('vader_lexicon')

def perform_sentiment_analysis(tweets_df):
    """
    Esegue l'analisi del sentiment sulla colonna 'cleaned_text' del DataFrame tweets_df.
    
    Parameters:
    - tweets_df (pd.DataFrame): DataFrame contenente i tweet pre-processati con la colonna 'cleaned_text'.
    
    Returns:
    - pd.DataFrame: DataFrame con un'ulteriore colonna 'sentiment_score'.
    """
    sid = SentimentIntensityAnalyzer()
    
    # Calcola i punteggi di sentiment
    tweets_df['sentiment_score'] = tweets_df['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    
    return tweets_df
