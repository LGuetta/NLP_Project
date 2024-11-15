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
│   └── visualization/                 # Visualization scripts
│       └── visualization_utils.py
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
├── tests/                             # Unit tests for functionality
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_model_training.py
│
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── main.py                            # Main script for orchestrating the workflow



```

### Descriptions

- **data/**: Contains all data files, divided into subdirectories:
  - **raw/**: Holds original datasets in their unmodified state, including:
    - `Binance_BTCUSDT_2022_minute.csv`: Minute-level BTC price data for 2022.
    - `bitcoin-tweets-2022.csv`: Bitcoin-related tweets from 2022.
  - **processed/**: Stores cleaned and processed data ready for modeling, including:
    - `btc_data_processed.csv`: Cleaned Bitcoin price data.
    - `tweets_data_december.csv`: Filtered tweets for December 2022.
    - `merged_data.csv`: Merged dataset of sentiment and returns.

- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis, sentiment analysis, and model training:
  - `EDA.ipynb`: For initial data exploration and visualization.
  - `Sentiment_Analysis.ipynb`: For applying sentiment analysis on tweets.
  - `Model_Training.ipynb`: For model training and evaluation.

- **src/**: Source code for the project's pipeline, divided by functionality:
  - **data/**: Scripts for data loading and preprocessing:
    - `data_preprocessing.py`: Functions to load and preprocess raw data.
  - **features/**: Scripts for feature extraction and engineering:
    - `feature_engineering.py`: Functions to create features from datasets.
  - **models/**: Scripts for training, evaluating, and tuning models:
    - `model_training.py`: Logic for training machine learning models.
    - `evaluate_model.py`: Functions for model evaluation.
  - **sentiment/**: Scripts for various sentiment analysis models:
    - `vader_sentiment.py`: Script for VADER sentiment analysis.
    - `cryptobert_sentiment.py`: Script for CryptoBERT sentiment analysis.
  - **utils/**: General utility functions for common tasks:
    - `helpers.py`: Helper functions used throughout the pipeline.
  - **visualization/**: Functions for generating plots and figures:
    - `plot_functions.py`: Plotting functions for data visualization.

- **models/**: Directory for storing trained machine learning models:
  - `random_forest.pkl`: Serialized Random Forest model.
  - `xgboost_model.pkl`: Serialized XGBoost model.

- **reports/**: Directory for project documentation and reports:
  - `final_report.pdf`: Final compiled report in PDF.
  - `latex_sources/`: LaTeX source files for the report:
    - `main.tex`: Main LaTeX document file.
    - `bibliography.bib`: References used in the report.

- **tests/**: Directory containing unit tests for scripts in the project:
  - `test_data_processing.py`: Tests for data preprocessing functions.
  - `test_sentiment_analysis.py`: Tests for sentiment analysis functions.
  - `test_model_training.py`: Tests for model training and evaluation scripts.

- **outputs/**: Stores generated output files from the workflow:
  - `cleaned_data.csv`: Final cleaned data.
  - `sentiment_scores.csv`: Sentiment scores obtained from analysis.
  - `model_predictions.csv`: Predictions from the trained models.

- **requirements.txt**: Contains all Python dependencies required to run the project.

- **README.md**: Project documentation and setup instructions.

- **main.py**: Main script for executing the complete workflow, including data processing, sentiment extraction, and model training.




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

- [Kaggle Datasets](https://www.kaggle.com/datasets/hiraddolatzadeh/bitcoin-tweets-2021-2022)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Gensim for Topic Modeling](https://radimrehurek.com/gensim/)

