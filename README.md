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
│   ├── processed/                     # Cleaned and preprocessed data
│       ├── BTCUSDT_1H.csv
│       ├── BTCUSDT_15m.csv
│       ├── BTCUSDT_30m.csv
│       └── cleaned_price_data.csv
│
├── notebooks/                         # Jupyter notebooks for exploration and analysis
│   ├── EDA.ipynb
│   ├── Fin_data_group.ipynb
│   ├── Merging_sent_fin.ipynb
│   ├── Models_15m.ipynb
│   ├── Models_1H.ipynb
│   ├── Models_30m.ipynb
│   ├── Tweets_filtering.ipynb
│   ├── Tweets_preprocessing.ipynb
│   ├── Vader+Elkulako_Sentiment_grouping.ipynb
│   └── kk08_sentiment.ipynb
│
├── src/                               # Source code
│   ├── data/                          # Data loading and preprocessing scripts
│   │   ├── data_preprocessing.py
│   │   └── utils.py
│   ├── features/                      # Feature engineering scripts
│       └── feature_engineering.py
│   ├── models/                        # Model training and evaluation scripts
│       ├── model_training.py
│       └── RNN_training.py
│   ├── sentiment/                     # Sentiment analysis scripts
│       ├── sentiment_analysis.py
│       └── Vader+Elkulako_Sentiment_grouping.py
│
├── models/                            # Serialized models
│   └── lda_model.pkl
│
├── outputs/                           # Generated outputs and final results
│   ├── cleaned_tweets_data.csv
│   ├── final_model.pkl
│   ├── merged_data_november_07_to_21.csv
│   └── sentiment_scores.csv
│
├── reports/                           # Project reports
│   └── final_report.pdf
│
│
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
└── main.py                            # Main script for orchestrating the workflow
```

## Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/bitcoin_sentiment_analysis.git
   cd bitcoin_sentiment_analysis
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

## Usage

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

## Analysis
- Sentiment Analysis: Uses both BERT "kk08 cryptobert" for crypto-specific sentiment and VADER for general sentiment scoring.
- Feature Engineering: Combines sentiment scores, tweet counts, and Bitcoin price returns over various time intervals.
- Lagged Features: Includes lagged sentiment scores to capture sentiment shifts over time.
  
## Model Training
- Models: Random Forest, XGBoost, and SVM, chosen for their effectiveness in binary classification tasks.
- Hyperparameter Tuning: Applied grid search or random search to optimize model parameters.
- Evaluation Metrics: Models were evaluated based on Accuracy, ROC-AUC, and F1-score.
  
## Results
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
Project made by:
-  Lorenzo Guetta
-  Claudia Jurado
-  Ivan Isaenko

- [Kaggle Datasets](https://www.kaggle.com/datasets/hiraddolatzadeh/bitcoin-tweets-2021-2022)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Gensim for Topic Modeling](https://radimrehurek.com/gensim/)

