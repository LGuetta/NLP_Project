# Bitcoin Sentiment Analysis

## Overview

This project performs sentiment analysis on Bitcoin-related tweets from 2022 and examines their influence on Bitcoin (BTC) price movements. By analyzing millions of tweets and corresponding minute-level BTC price data, the project aims to uncover patterns and insights that can inform trading strategies and understand market sentiment dynamics.

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

## Data

### 1. Bitcoin Price Data
- **Source:** Binance BTCUSDT 2022 Minute Data
- **Description:** Contains minute-level BTC price information for the year 2022.
- **Columns:**
  - `unix`: Unix timestamp in milliseconds
  - `date`: Date and time
  - `symbol`: Trading pair symbol (e.g., BTCUSDT)
  - `open`: Opening price
  - `high`: Highest price
  - `low`: Lowest price
  - `close`: Closing price
  - `volume`: Trading volume
  - `volume_from`: Volume from the base asset
  - `tradecount`: Number of trades

### 2. Bitcoin Tweets Data
- **Source:** [Kaggle - Bitcoin Tweets 2021-2022](https://www.kaggle.com/datasets/hiraddolatzadeh/bitcoin-tweets-2021-2022)
- **Description:** Contains millions of tweets related to Bitcoin from 2022.
- **Columns:**
  - `datetime`: Timestamp of the tweet
  - `date`: Date of the tweet
  - `username`: Twitter handle of the user
  - `text`: Content of the tweet

## Project Structure

```
bitcoin_sentiment_analysis/
│
├── data/
│   ├── Binance_BTCUSDT_2022_minute.csv
│   └── bitcoin-tweets-2022.csv
│
├── notebooks/
│   └── Exploratory_Data_Analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── sentiment_analysis.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
│
├── models/
│   └── lda_model.pkl
│
├── outputs/
│   ├── cleaned_data.csv
│   ├── sentiment_scores.csv
│   └── final_model.pkl
│
├── requirements.txt
├── README.md
└── main.py
```

### Descriptions

- **data/**: Contains raw datasets.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and visualizations.
- **src/**: Python modules for different stages of the project.
  - `data_preprocessing.py`: Functions to load and clean data.
  - `sentiment_analysis.py`: Functions to perform sentiment analysis.
  - `feature_engineering.py`: Functions to create features for modeling.
  - `model_training.py`: Functions to train and evaluate predictive models.
  - `utils.py`: Utility functions.
- **models/**: Serialized machine learning models.
- **outputs/**: Processed data and final model outputs.
- **requirements.txt**: Python dependencies.
- **README.md**: Project documentation.
- **main.py**: Orchestrates the workflow by calling functions from `src/` modules.

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
   - Navigate to the `notebooks/` directory and open `Exploratory_Data_Analysis.ipynb` to perform EDA and visualize data insights.

## Analysis

- **Sentiment Analysis:** Uses VADER to assign sentiment scores to each tweet, indicating positive, negative, or neutral sentiments.
- **Topic Modeling:** Implements LDA to identify prevalent topics within the Bitcoin-related tweets.
- **Feature Engineering:** Combines sentiment scores, tweet counts, and topic distributions with BTC price data to create features for modeling.

## Model Training

- **Algorithms Used:** Random Forest Regressor, Linear Regression, etc.
- **Evaluation Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared (R²).
- **Hyperparameter Tuning:** Utilizes Grid Search or Random Search to optimize model parameters.

## Results

- **Sentiment vs. BTC Price:** Analyzes the correlation between tweet sentiments and BTC price movements.
- **Topic Insights:** Identifies key topics driving market sentiment.
- **Predictive Performance:** Evaluates how well the model can predict BTC prices based on tweet data.

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

- [Kaggle Datasets](https://www.kaggle.com/datasets/hiraddolatzadeh/bitcoin-tweets-2021-2022)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Gensim for Topic Modeling](https://radimrehurek.com/gensim/)

