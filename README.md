# Bitcoin Sentiment Analysis

## Overview

This project investigates the impact of Bitcoin-related tweet sentiment on Bitcoin (BTC) price movements. Using NLP techniques, including sentiment analysis, we analyze over 1.3 million tweets from December 2022 to predict the direction of Bitcoin's hourly returns. The project leverages minute-level BTC price data to explore potential correlations and insights into market sentiment.

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Analysis](#analysis)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## IMPORTANT WARNING

The data folder is too large to be included in this repository. You can download the full dataset directly from the following link: [Google Drive](https://drive.google.com/drive/folders/1xoaTUISduJwv0dqQumHL2INuhyTlSd7W?usp=drive_link)

## Data

### 1. Bitcoin Price Data
- **Source**: Binance BTCUSDT data
- **Description**: Minute-level and hourly BTC price information from 2022.
- **Key Files**:
  - `Binance_BTCUSDT_1h.csv`: Hourly BTC prices.
  - `Binance_BTCUSDT_2022_minute.csv`: Minute-level BTC prices, used for generating 15- and 30-minute returns.
  
### 2. Bitcoin Tweets Data
- **Source**: [Kaggle - Bitcoin Tweets 2021-2022](https://www.kaggle.com/datasets/hiraddolatzadeh/bitcoin-tweets-2021-2022)
- **Description**: Contains 1.3 million Bitcoin-related tweets from December 2022.
- **Columns**:
  - `datetime`: Timestamp of the tweet
  - `date`: Date of the tweet
  - `username`: Twitter user handle
  - `text`: Content of the tweet
- **Indication**: The file might not be included in the repository since its dimension exceeds the limit. You can download it for free on the link above.

## Project Structure

```
bitcoin_sentiment_analysis/
│
├── data/                              # Data files
│   ├── raw/                           # Original datasets
│   │   ├── Binance_BTCUSDT_2022_minute.csv
│   │   └── bitcoin-tweets-2022.csv
│   ├── processed/                     # Cleaned and preprocessed data. Look google drive for full datasets.
│       ├── BTCUSDT_1H.csv
│       ├── BTCUSDT_15m.csv
│       ├── BTCUSDT_30m.csv
│       └── cleaned_price_data.csv
│
├── notebooks/                         # Jupyter notebooks for exploration analysis, data pre-processing and models
│   ├── EDA.ipynb
│   ├── Fin_data_group.ipynb
│   ├── Merging_sent_fin.ipynb
│   ├── 15m_NN.ipynb                   
│   ├── 30m_NN.ipynb
│   ├── 1H_NN.ipynb
│   ├── Models_15m.ipynb
│   ├── Models_1H.ipynb
│   ├── Models_30m.ipynb
│   ├── Tweets_filtering.ipynb
│   ├── Tweets_preprocessing.ipynb
│   ├── Vader+Elkulako_Sentiment_grouping.ipynb
│   └── kk08_sentiment.ipynb
│
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
├── README.md                          # Project documentation
├── LICENSE                            # MIT License

```
# Description

This section provides an overview of the primary files and their respective roles within the project:

## Notebooks
- **`EDA.ipynb`**: Exploratory Data Analysis (EDA) for understanding the dataset, visualizing data distributions, and identifying trends. This includes sentiment distribution graphs from various sentiment analysis models.
- **`Fin_data_group.ipynb`**: Prepares and processes financial data for subsequent analysis, including calculations of Bitcoin returns over different timeframes.
- **`Merging_sent_fin.ipynb`**: Merges sentiment analysis data with financial data to create a comprehensive dataset for model training.
- **`Models_15m.ipynb`, `Models_1H.ipynb`, `Models_30m.ipynb`**: Implements Random Forest, XGBoost, and SVM models for predicting Bitcoin price direction based on different time intervals.
- **`Tweets_filtering.ipynb`**: Filters the initial dataset of tweets, retaining those relevant to Bitcoin sentiment analysis.
- **`Tweets_preprocessing.ipynb`**: Preprocesses tweets by removing irrelevant elements such as usernames, URLs, punctuation, and common stopwords.
- **`Vader+Elkulako_Sentiment_grouping.ipynb`**: Applies VADER and ElKulako models to extract sentiment from the tweets, assigning sentiment categories (e.g., bearish, neutral, bullish).
- **`kk08_sentiment.ipynb`**: Utilizes the kk08 CryptoBERT model to analyze the sentiment of the tweets, categorizing them as positive or negative.

## Reports
- **`final_report.pdf`**: A comprehensive report detailing the project's objectives, methodology, results, analysis, and conclusions. It includes related work, preprocessing, sentiment analysis methods, and the evaluation of different models.


### Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/LGuetta/NLP_Project.git
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Run the Main Script**
   ```bash
   python main.py
   ```
   This script will:
   - Load and preprocess the data.
   - Perform sentiment analysis.
   - Engineer features.
   - Train the predictive model.
   - Save the outputs.

2. **Explore Notebooks**
   - Navigate to the `notebooks/` directory and open `EDA.ipynb` to perform EDA and visualize data insights.

### Analysis
- **Sentiment Analysis**: Uses both BERT "kk08 cryptobert" for crypto-specific sentiment and VADER for general sentiment scoring.
- **Feature Engineering**: Combines sentiment scores, tweet counts, and Bitcoin price returns over various time intervals.
- **Lagged Features**: Includes lagged sentiment scores to capture sentiment shifts over time.

### Model Training
- **Models**: Random Forest, XGBoost, and SVM, chosen for their effectiveness in binary classification tasks.
- **Hyperparameter Tuning**: Applied grid search or random search to optimize model parameters.
- **Evaluation Metrics**: Models were evaluated based on Accuracy, ROC-AUC, and F1-score.

### Results
The project demonstrated that tweet sentiment has a moderate correlation with Bitcoin price movements. Sentiment scores, especially when lagged, were shown to contribute predictive power for determining the return sign of Bitcoin prices in hourly intervals.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m "Add YourFeature"
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
Project made by: Claudia Jurado, Ivan Isaenko, and Lorenzo Guetta

- [Kaggle Datasets](https://www.kaggle.com/datasets/hiraddolatzadeh/bitcoin-tweets-2021-2022)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Gensim for Topic Modeling](https://radimrehurek.com/gensim/)

