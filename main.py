# main.py

from src.data_preprocessing import load_price_data, load_and_clean_tweets
from src.sentiment_analysis import perform_sentiment_analysis
from src.feature_engineering import create_features
from src.model_training import train_model

def main():
    # Definisci i percorsi dei file
    price_path = 'data/Binance_BTCUSDT_2022_minute.csv'
    tweets_path = 'data/NLP_Bitcoin_tweets.csv'
    
    # Carica e pre-processa i dati dei prezzi
    price_df = load_price_data(price_path)
    
    # Carica e pre-processa i dati dei tweet
    tweets_df = load_and_clean_tweets(tweets_path)
    
    # Esegui l'analisi del sentiment
    tweets_df = perform_sentiment_analysis(tweets_df)
    print("Analisi del sentiment completata.")
    
    # Salva i tweet con i punteggi di sentiment (opzionale)
    tweets_df.to_csv('outputs/sentiment_scores.csv')
    print("Punteggi di sentiment salvati in 'outputs/sentiment_scores.csv'.")
    
    # Feature Engineering
    features_df = create_features(price_df, tweets_df)
    print("Feature engineering completato.")
    
    # Salva i dati con le feature (opzionale)
    features_df.to_csv('outputs/cleaned_data_with_features.csv')
    print("Dati con le feature salvati in 'outputs/cleaned_data_with_features.csv'.")
    
    # Addestra il modello predittivo
    model = train_model(features_df)
    print("Modello predittivo addestrato.")
    
    # Salva il modello finale
    model.save('outputs/final_model.pkl')
    print("Modello finale salvato in 'outputs/final_model.pkl'.")

if __name__ == "__main__":
    main()
