from src.data_preprocessing import load_and_clean_data
from src.sentiment_analysis import perform_sentiment_analysis
from src.feature_engineering import create_features
from src.model_training import train_model

def main():
    # Load and preprocess data
    price_df, tweets_df = load_and_clean_data('data/Binance_BTCUSDT_2022_minute.csv', 'data/bitcoin-tweets-2022.csv')
    
    # Perform sentiment analysis
    sentiment_df = perform_sentiment_analysis(tweets_df)
    
    # Feature engineering
    features_df = create_features(price_df, sentiment_df)
    
    # Train predictive model
    model = train_model(features_df)
    
    # Save outputs
    features_df.to_csv('outputs/cleaned_data.csv')
    model.save('outputs/final_model.pkl')

if __name__ == "__main__":
    main()
