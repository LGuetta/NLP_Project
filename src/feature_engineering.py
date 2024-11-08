# src/feature_engineering.py

import pandas as pd

def create_features(price_df, tweets_df):
    """
    Combina i dati dei prezzi di BTC con i punteggi di sentiment e crea nuove feature.
    
    Parameters:
    - price_df (pd.DataFrame): DataFrame dei prezzi di BTC con indice datetime.
    - tweets_df (pd.DataFrame): DataFrame dei tweet con punteggi di sentiment e indice datetime.
    
    Returns:
    - pd.DataFrame: DataFrame unito con nuove feature.
    """
    # Resample dei punteggi di sentiment a intervalli di 15 minuti prendendo la media
    sentiment_resampled = tweets_df['sentiment_score'].resample('15T').mean().rename('avg_sentiment')
    
    # Resample del conteggio dei tweet a intervalli di 15 minuti
    tweet_counts = tweets_df['sentiment_score'].resample('15T').count().rename('tweet_count')
    
    # Unisci i dati dei prezzi con i punteggi di sentiment e il conteggio dei tweet
    merged_df = price_df.join([sentiment_resampled, tweet_counts])
    
    # Riempie i valori mancanti dei sentiment con 0
    merged_df['avg_sentiment'].fillna(0, inplace=True)
    merged_df['tweet_count'].fillna(0, inplace=True)
    
    # Crea feature aggiuntive
    merged_df['sentiment_ma_15'] = merged_df['avg_sentiment'].rolling(window=4).mean()  # Media mobile di 1 ora
    merged_df['price_change'] = merged_df['close'].pct_change()
    merged_df['price_ma_15'] = merged_df['close'].rolling(window=4).mean()
    
    # Rimuove le righe con valori mancanti risultanti dal rolling
    merged_df.dropna(inplace=True)
    
    return merged_df
