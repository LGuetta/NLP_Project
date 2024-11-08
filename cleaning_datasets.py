import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from pathlib import Path  # Import pathlib

# Define data types for BTC price data
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

# ------------------------ Data Loading ------------------------

# Define file paths using pathlib
btc_price_path = Path('C:/Users/loren/Desktop/Università/NLP/Project/Datasets/Binance_BTCUSDT_2022_minute.csv')
tweets_path = Path('C:/Users/loren/Desktop/Università/NLP/Project/Datasets/bitcoin-tweets-2022.csv')


# Load Bitcoin price data without parsing dates initially
btc_price_df = pd.read_csv(btc_price_path, 
                           dtype=dtype_spec, 
                           encoding='utf-8')

# Verify the first few rows
print("Bitcoin Price Data Preview:")
print(btc_price_df.head())

# ------------------------ Data Preprocessing ------------------------

# Strip whitespace from column names to avoid issues
btc_price_df.columns = btc_price_df.columns.str.strip()

# Check if 'unix' column exists
if 'unix' in btc_price_df.columns:
    # Convert 'unix' to datetime (assuming 'unix' is in milliseconds)
    btc_price_df['datetime'] = pd.to_datetime(btc_price_df['unix'], unit='ms', errors='coerce')
    
    # Check for conversion errors
    if btc_price_df['datetime'].isnull().any():
        print("\nWarning: Some 'unix' values could not be converted to datetime.")
    
    # Set 'datetime' as the index
    btc_price_df.set_index('datetime', inplace=True)
    print("\nSet 'datetime' as index for BTC price data.")
else:
    print("\nError: 'unix' column not found in BTC price data.")
    exit()

# Check for missing values after datetime conversion
print("\nMissing values in BTC price data after datetime conversion:")
print(btc_price_df.isnull().sum())

# Forward fill missing values
btc_price_df.ffill(inplace=True)  # Updated to use ffill() to avoid FutureWarning

# Load tweets data with datetime parsing
tweets_df = pd.read_csv(tweets_path, parse_dates=['datetime'], encoding='utf-8')

# Verify the first few rows
print("\nTweets Data Preview:")
print(tweets_df.head())

# Check for missing values
print("\nMissing values in tweets data:")
print(tweets_df.isnull().sum())

# Filter to English tweets if 'language' column exists
if 'language' in tweets_df.columns:
    tweets_df = tweets_df[tweets_df['language'] == 'en']
    print("\nFiltered to English tweets.")
else:
    print("\n'language' column not found. Proceeding without language filtering.")

# Drop duplicate tweets based on the 'text' column if it exists
if 'text' in tweets_df.columns:
    initial_count = len(tweets_df)
    tweets_df.drop_duplicates(subset='text', inplace=True)
    final_count = len(tweets_df)
    print(f"\nDropped {initial_count - final_count} duplicate tweets.")
else:
    print("\n'text' column not found. Proceeding without dropping duplicates.")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define text cleaning function
def clean_text(text):
    """
    Cleans the input text by removing URLs, user mentions, hashtags, punctuation,
    stopwords, and performing lemmatization.
    
    Parameters:
    - text (str): The tweet text to clean.
    
    Returns:
    - str: The cleaned and lemmatized text.
    """
    if pd.isnull(text):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references and '#'
    text = re.sub(r'\@\w+|\#','', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize using wordpunct_tokenize to avoid 'punkt_tab' error
    tokens = wordpunct_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Apply cleaning function to tweets if 'text' column exists
if 'text' in tweets_df.columns:
    tweets_df['cleaned_tweet'] = tweets_df['text'].apply(clean_text)
    print("\nApplied text cleaning to tweets.")
else:
    tweets_df['cleaned_tweet'] = ""
    print("\n'text' column not found. 'cleaned_tweet' column created with empty strings.")

# Set 'datetime' as datetime index for tweets
if 'datetime' in tweets_df.columns:
    tweets_df.set_index('datetime', inplace=True)
    print("\nSet 'datetime' as index for tweets data.")
else:
    print("\n'datetime' column not found. Cannot set as index.")

# ------------------------ Resampling ------------------------

# Resample Tweets Data to 15-Minute Intervals
# Count of tweets per 15-minute interval
tweet_counts = tweets_df.resample('15T').size()

# Combine tweets within each interval by concatenating cleaned tweets
tweets_15min = tweets_df.resample('15T').agg({'cleaned_tweet': ' '.join})

# Add tweet counts to the resampled tweets dataframe
tweets_15min['tweet_count'] = tweet_counts

# Verify the resampled tweets data
print("\nResampled Tweets Data Preview:")
print(tweets_15min.head())

# Resample Bitcoin prices to 15-Minute Intervals using forward fill
# Handle potential column name case sensitivity
if 'close' in btc_price_df.columns:
    btc_price_15min = btc_price_df['close'].resample('15T').ffill()
    print("\nResampled BTC prices using 'close' column.")
elif 'Close' in btc_price_df.columns:
    btc_price_15min = btc_price_df['Close'].resample('15T').ffill()
    print("\nResampled BTC prices using 'Close' column.")
else:
    raise KeyError("Column 'close' or 'Close' not found in BTC price data.")

# Merge tweets and BTC price data on the datetime index
merged_df = pd.merge(tweets_15min, btc_price_15min, left_index=True, right_index=True, how='inner')

# Check for missing values after merging
print("\nMissing values after merging:")
print(merged_df.isnull().sum())

# Handle missing values
# Option 1: Fill NaN tweet counts with 0
merged_df['tweet_count'].fillna(0, inplace=True)

# Option 2: Drop rows with any remaining missing values
merged_df.dropna(inplace=True)

# Verify the final merged data
print("\nFinal Merged Data Preview:")
print(merged_df.head())

# ------------------------ Feature Engineering ------------------------

# Define features and target
X = merged_df[['cleaned_tweet', 'tweet_count']]
y = merged_df['close'] if 'close' in merged_df.columns else merged_df['Close']

# Verify shapes
print("\nFeatures and Target Shapes:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Feature Engineering for 'cleaned_tweet' using TF-IDF
tfidf = TfidfVectorizer(max_features=500)  # Adjust 'max_features' based on dataset size
X_tfidf = tfidf.fit_transform(X['cleaned_tweet']).toarray()

# Combine TF-IDF features with 'tweet_count'
X_final = np.hstack((X_tfidf, X[['tweet_count']].values))

# Verify the shape of the final feature set
print(f"\nFinal Feature Set Shape: {X_final.shape}")

# ------------------------ Train-Test Split ------------------------

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42)

# Verify the shapes of the splits
print("\nTrain-Test Split Shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")
