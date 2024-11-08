# src/data_preprocessing.py

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from nltk.tokenize import wordpunct_tokenize  # Importa wordpunct_tokenize

# Scarica i pacchetti NLTK necessari
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_price_data(file_path):
    """
    Carica e pre-processa i dati dei prezzi di Bitcoin.

    Parameters:
    - file_path (str): Percorso al file CSV dei prezzi.

    Returns:
    - pd.DataFrame: DataFrame pre-processato con indice datetime.
    """
    dtype_spec = {
        'unix': 'int64',
        'date': 'str',
        'symbol': 'str',
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'float64',
        'volume_from': 'float64',
        'tradecount': 'int64'
    }

    # Carica i dati
    price_df = pd.read_csv(file_path, dtype=dtype_spec, encoding='utf-8')
    print("Bitcoin Price Data Loaded Successfully.")

    # Rimuovi eventuali spazi nei nomi delle colonne
    price_df.columns = price_df.columns.str.strip()

    # Converti 'unix' in datetime
    if 'unix' in price_df.columns:
        price_df['datetime'] = pd.to_datetime(price_df['unix'], unit='ms', errors='coerce')
        if price_df['datetime'].isnull().any():
            print("Warning: Alcuni valori 'unix' non sono stati convertiti correttamente.")
        price_df.set_index('datetime', inplace=True)
        print("Set 'datetime' as index for BTC price data.")
    else:
        raise KeyError("'unix' column not found in Bitcoin price data.")

    # Ordinare l'indice datetime
    price_df.sort_index(inplace=True)
    print("Indice datetime ordinato in modo crescente.")

    # Rimuovere duplicati nell'indice
    initial_count = len(price_df)
    price_df = price_df[~price_df.index.duplicated(keep='first')]
    final_count = len(price_df)
    print(f"Rimosso {initial_count - final_count} duplicati nell'indice datetime.")

    # Controlla e gestisci i valori mancanti
    print("\nMissing values in Bitcoin price data:")
    print(price_df.isnull().sum())

    # Forward fill per gestire i valori mancanti
    price_df.ffill(inplace=True)

    return price_df

def preprocess_text(text, stop_words, lemmatizer):
    """
    Pulisce e pre-processa il testo dei tweet.

    Parameters:
    - text (str): Testo del tweet.
    - stop_words (set): Set di stopwords in inglese.
    - lemmatizer (WordNetLemmatizer): Oggetto lemmatizzatore.

    Returns:
    - str: Testo pre-processato.
    """
    if pd.isnull(text):
        return ""

    # Converti in minuscolo
    text = text.lower()

    # Rimuovi URL, menzioni e hashtag
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#\w+', '', text)

    # Rimuovi caratteri non alfabetici
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Rimuovi punteggiatura e numeri
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])

    # Tokenizzazione usando wordpunct_tokenize
    tokens = wordpunct_tokenize(text)

    # Lemmatizzazione
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Rimuovi stopwords
    tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(tokens)

def check_time_gaps(df, freq='15min'):
    """
    Controlla se ci sono buchi temporali nel DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame con indice datetime.
    - freq (str): Frequenza attesa (default: '15min').

    Returns:
    - None
    """
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    missing_times = expected_index.difference(df.index)
    if not missing_times.empty:
        print(f"\nCi sono {len(missing_times)} timestamp mancanti nel dataset:")
        print(missing_times)
    else:
        print("\nNon ci sono timestamp mancanti nel dataset.")

def load_and_clean_tweets(file_path):
    """
    Carica e pre-processa i dati dei tweet.

    Parameters:
    - file_path (str): Percorso al file CSV dei tweet.

    Returns:
    - pd.DataFrame: DataFrame pre-processato con testo pulito e indice datetime.
    """
    # Definisci i tipi di dati, usando 'boolean' per colonne booleani nullable
    dtype_spec = {
        'user_name': 'str',
        'user_location': 'str',
        'user_description': 'str',
        'user_created': 'str',
        'user_followers': 'Int64',  # Use 'Int64' for nullable integers
        'user_friends': 'Int64',
        'user_favourites': 'Int64',
        'user_verified': 'boolean',  # Use 'boolean' for nullable booleans
        'text': 'str',
        'hashtags': 'str',
        'source': 'str',
        'is_retweet': 'boolean'  # Use 'boolean' for nullable booleans
    }

    # Carica i dati
    tweets_df = pd.read_csv(file_path, parse_dates=['date'], encoding='utf-8', dtype=dtype_spec, low_memory=False)
    print("Tweets Data Loaded Successfully.")

    # Rimuovi eventuali spazi nei nomi delle colonne
    tweets_df.columns = tweets_df.columns.str.strip()

    # Imposta 'date' come indice datetime
    if 'date' in tweets_df.columns:
        tweets_df.set_index('date', inplace=True)
        print("Set 'date' as datetime index for tweets data.")
    else:
        raise KeyError("'date' column not found in Tweets data.")

    # Ordinare l'indice datetime
    tweets_df.sort_index(inplace=True)
    print("Indice datetime dei tweet ordinato in modo crescente.")

    # Verifica se l'indice è monotono
    if not tweets_df.index.is_monotonic_increasing:
        raise ValueError("L'indice datetime dei tweet non è monotono crescente dopo l'ordinamento.")
    else:
        print("Indice datetime dei tweet è monotono crescente.")

    # Controlla e gestisci i valori mancanti
    print("\nMissing values in Tweets data:")
    print(tweets_df.isnull().sum())

    # Rimuovi tweet con testo mancante
    initial_count = len(tweets_df)
    tweets_df = tweets_df.dropna(subset=['text'])
    final_count = len(tweets_df)
    print(f"Dropped {initial_count - final_count} tweets with missing text.")

    # Inizializza stopwords e lemmatizzatore
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Preprocessa il testo
    tweets_df['cleaned_text'] = tweets_df['text'].apply(lambda x: preprocess_text(x, stop_words, lemmatizer))
    print("Preprocessed tweet texts.")

    # Stampa l'intervallo di date
    print(f"\nTweets date range: {tweets_df.index.min()} to {tweets_df.index.max()}")

    return tweets_df

def main():
    # Definisci i percorsi dei file
    btc_price_path = Path('data/Binance_BTCUSDT_2022_minute.csv')
    tweets_path = Path('data/NLP_Bitcoin_tweets.csv')

    # Carica e pre-processa i dati dei prezzi
    price_df = load_price_data(btc_price_path)

    # Carica e pre-processa i dati dei tweet
    tweets_df = load_and_clean_tweets(tweets_path)

    # Salva i dati puliti (opzionale)
    price_df.to_csv('outputs/cleaned_price_data.csv')
    tweets_df.to_csv('outputs/cleaned_tweets_data.csv')
    print("Cleaned data saved to 'outputs/' directory.")

    # Filtra per le due settimane specificate nel dataset dei tweet
    # Imposta start_date e end_date basandoti sull'intervallo disponibile
    start_date = '2022-11-07 16:27:36'
    end_date = '2022-11-21 23:59:59'

    # Stampa l'intervallo di date disponibile nei dati
    print(f"\nPrice Data Date Range: {price_df.index.min()} to {price_df.index.max()}")
    print(f"Tweets Data Date Range: {tweets_df.index.min()} to {tweets_df.index.max()}")

    # Verifica che le date esistano nel dataset
    if not (price_df.index.min() <= pd.to_datetime(start_date) <= price_df.index.max()) or \
       not (price_df.index.min() <= pd.to_datetime(end_date) <= price_df.index.max()):
        raise ValueError("L'intervallo di date specificato non esiste nel dataset dei prezzi.")

    if not (tweets_df.index.min() <= pd.to_datetime(start_date) <= tweets_df.index.max()) or \
       not (tweets_df.index.min() <= pd.to_datetime(end_date) <= tweets_df.index.max()):
        raise ValueError("L'intervallo di date specificato non esiste nel dataset dei tweet.")

    # Filtra i dati per l'intervallo di date
    price_df_filtered = price_df[start_date:end_date]
    tweets_df_filtered = tweets_df[start_date:end_date]

    # Verifica se i DataFrame filtrati non sono vuoti
    if price_df_filtered.empty:
        raise ValueError("Il DataFrame dei prezzi filtrato è vuoto. Controlla l'intervallo di date.")
    if tweets_df_filtered.empty:
        raise ValueError("Il DataFrame dei tweet filtrato è vuoto. Controlla l'intervallo di date.")

    # Resample a 15 minuti
    price_resampled = price_df_filtered.resample('15min').ffill()
    tweets_resampled = tweets_df_filtered.resample('15min').agg({
        'cleaned_text': ' '.join,
        'text': 'count'  # Counting number of tweets
    }).rename(columns={'text': 'tweet_count'})

    # Verifica l'ordine degli indici dopo il resampling
    price_resampled.sort_index(inplace=True)
    tweets_resampled.sort_index(inplace=True)

    # Controlla eventuali buchi temporali dopo il resampling
    check_time_gaps(price_resampled, freq='15min')
    check_time_gaps(tweets_resampled, freq='15min')

    # Unisci i dataset
    merged_df = pd.merge(price_resampled, tweets_resampled, left_index=True, right_index=True, how='inner')

    # Gestisci eventuali valori mancanti
    merged_df['tweet_count'] = merged_df['tweet_count'].fillna(0)

    # Rimuovi eventuali duplicati dopo la fusione
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

    # Salva il dataset unito
    merged_df.to_csv('outputs/merged_data_november_07_to_21.csv')  # Puoi rinominare il file se preferisci
    print("Merged and resampled data saved as 'merged_data_november_07_to_21.csv'.")

if __name__ == "__main__":
    main()
