# src/data_preprocessing.py

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from nltk.tokenize import wordpunct_tokenize  # Import wordpunct_tokenize

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_price_data(file_path):
    """
    Load and preprocess Bitcoin price data.

    Parameters:
    - file_path (str): Path to the price CSV file.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame with datetime index.
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

    # Load the data
    price_df = pd.read_csv(file_path, dtype=dtype_spec, encoding='utf-8')
    print("Bitcoin Price Data Loaded Successfully.")

    # Remove any spaces in column names
    price_df.columns = price_df.columns.str.strip()

    # Convert 'unix' to datetime
    if 'unix' in price_df.columns:
        price_df['datetime'] = pd.to_datetime(price_df['unix'], unit='ms', errors='coerce')
        if price_df['datetime'].isnull().any():
            print("Warning: Some 'unix' values were not converted correctly.")
        price_df.set_index('datetime', inplace=True)
        print("Set 'datetime' as index for BTC price data.")
    else:
        raise KeyError("'unix' column not found in Bitcoin price data.")

    # Sort the datetime index
    price_df.sort_index(inplace=True)
    print("Datetime index sorted in ascending order.")

    # Remove duplicates in the index
    initial_count = len(price_df)
    price_df = price_df[~price_df.index.duplicated(keep='first')]
    final_count = len(price_df)
    print(f"Removed {initial_count - final_count} duplicates in the datetime index.")

    # Check and handle missing values
    print("\nMissing values in Bitcoin price data:")
    print(price_df.isnull().sum())

    # Forward fill to handle missing values
    price_df.ffill(inplace=True)

    return price_df

def preprocess_text(text, stop_words, lemmatizer):
    """
    Clean and preprocess tweet text.

    Parameters:
    - text (str): Tweet text.
    - stop_words (set): Set of English stopwords.
    - lemmatizer (WordNetLemmatizer): Lemmatizer object.

    Returns:
    - str: Preprocessed text.
    """
    if pd.isnull(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs, mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#\w+', '', text)

    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove punctuation and numbers
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])

    # Tokenization using wordpunct_tokenize
    tokens = wordpunct_tokenize(text)

    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(tokens)

def check_time_gaps(df, freq='15min'):
    """
    Check for temporal gaps in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with datetime index.
    - freq (str): Expected frequency (default: '15min').

    Returns:
    - None
    """
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    missing_times = expected_index.difference(df.index)
    if not missing_times.empty:
        print(f"\nThere are {len(missing_times)} missing timestamps in the dataset:")
        print(missing_times)
    else:
        print("\nThere are no missing timestamps in the dataset.")

def load_and_clean_tweets(file_path):
    """
    Load and preprocess tweet data.

    Parameters:
    - file_path (str): Path to the tweets CSV file.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame with cleaned text and datetime index.
    """
    # Define data types, using 'boolean' for nullable boolean columns
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

    # Load the data
    tweets_df = pd.read_csv(file_path, parse_dates=['date'], encoding='utf-8', dtype=dtype_spec, low_memory=False)
    print("Tweets Data Loaded Successfully.")

    # Remove any spaces in column names
    tweets_df.columns = tweets_df.columns.str.strip()

    # Set 'date' as datetime index
    if 'date' in tweets_df.columns:
        tweets_df.set_index('date', inplace=True)
        print("Set 'date' as datetime index for tweets data.")
    else:
        raise KeyError("'date' column not found in Tweets data.")

    # Sort the datetime index
    tweets_df.sort_index(inplace=True)
    print("Datetime index of tweets sorted in ascending order.")

    # Check if the index is monotonic
    if not tweets_df.index.is_monotonic_increasing:
        raise ValueError("The datetime index of tweets is not monotonically increasing after sorting.")
    else:
        print("Datetime index of tweets is monotonically increasing.")

    # Check and handle missing values
    print("\nMissing values in Tweets data:")
    print(tweets_df.isnull().sum())

    # Remove tweets with missing text
    initial_count = len(tweets_df)
    tweets_df = tweets_df.dropna(subset=['text'])
    final_count = len(tweets_df)
    print(f"Dropped {initial_count - final_count} tweets with missing text.")

    # Initialize stopwords and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Preprocess the text
    tweets_df['cleaned_text'] = tweets_df['text'].apply(lambda x: preprocess_text(x, stop_words, lemmatizer))
    print("Preprocessed tweet texts.")

    # Print the date range
    print(f"\nTweets date range: {tweets_df.index.min()} to {tweets_df.index.max()}")

    return tweets_df

def main():
    # Define file paths
    btc_price_path = Path('data/Binance_BTCUSDT_2022_minute.csv')
    tweets_path = Path('data/NLP_Bitcoin_tweets.csv')

    # Load and preprocess price data
    price_df = load_price_data(btc_price_path)

    # Load and preprocess tweet data
    tweets_df = load_and_clean_tweets(tweets_path)

    # Save the cleaned data (optional)
    price_df.to_csv('outputs/cleaned_price_data.csv')
    tweets_df.to_csv('outputs/cleaned_tweets_data.csv')
    print("Cleaned data saved to 'outputs/' directory.")

    # Filter for the specified two-week period in the tweet dataset
    # Set start_date and end_date based on the available range
    start_date = '2022-11-07 16:27:36'
    end_date = '2022-11-21 23:59:59'

    # Print the available date ranges in the data
    print(f"\nPrice Data Date Range: {price_df.index.min()} to {price_df.index.max()}")
    print(f"Tweets Data Date Range: {tweets_df.index.min()} to {tweets_df.index.max()}")

    # Verify that the specified dates exist in the dataset
    if not (price_df.index.min() <= pd.to_datetime(start_date) <= price_df.index.max()) or \
       not (price_df.index.min() <= pd.to_datetime(end_date) <= price_df.index.max()):
        raise ValueError("The specified date range does not exist in the price dataset.")

    if not (tweets_df.index.min() <= pd.to_datetime(start_date) <= tweets_df.index.max()) or \
       not (tweets_df.index.min() <= pd.to_datetime(end_date) <= tweets_df.index.max()):
        raise ValueError("The specified date range does not exist in the tweet dataset.")

    # Filter the data for the specified date range
    price_df_filtered = price_df[start_date:end_date]
    tweets_df_filtered = tweets_df[start_date:end_date]

    # Check if the filtered DataFrames are not empty
    if price_df_filtered.empty:
        raise ValueError("The filtered price DataFrame is empty. Check the date range.")
    if tweets_df_filtered.empty:
        raise ValueError("The filtered tweet DataFrame is empty. Check the date range.")

    # Resample to 15 minutes
    price_resampled = price_df_filtered.resample('15min').ffill()
    tweets_resampled = tweets_df_filtered.resample('15min').agg({
        'cleaned_text': ' '.join,
        'text': 'count'  # Counting number of tweets
    }).rename(columns={'text': 'tweet_count'})

    # Sort the indices after resampling
    price_resampled.sort_index(inplace=True)
    tweets_resampled.sort_index(inplace=True)

    # Check for any temporal gaps after resampling
    check_time_gaps(price_resampled, freq='15min')
    check_time_gaps(tweets_resampled, freq='15min')

    # Merge the datasets
    merged_df = pd.merge(price_resampled, tweets_resampled, left_index=True, right_index=True, how='inner')

    # Handle any missing values
    merged_df['tweet_count'] = merged_df['tweet_count'].fillna(0)

    # Remove any duplicates after merging
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

    # Save the merged dataset
    merged_df.to_csv('outputs/merged_data_november_07_to_21.csv')  # You can rename the file if preferred
    print("Merged and resampled data saved as 'merged_data_november_07_to_21.csv'.")

if __name__ == "__main__":
    main()
