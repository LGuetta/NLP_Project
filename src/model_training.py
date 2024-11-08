# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model(features_df):
    """
    Addestra un modello di regressione per prevedere il prezzo di BTC basato sulle feature.
    
    Parameters:
    - features_df (pd.DataFrame): DataFrame con feature ingegnerizzate e target.
    
    Returns:
    - RandomForestRegressor: Modello addestrato.
    """
    # Definisci le feature e il target
    X = features_df[['avg_sentiment', 'tweet_count', 'sentiment_ma_15', 'price_change', 'price_ma_15']]
    y = features_df['close']
    
    # Split in training e testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inizializza il modello
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Addestra il modello
    rf_model.fit(X_train, y_train)
    
    # Previsioni
    y_pred = rf_model.predict(X_test)
    
    # Valutazione
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest MSE: {mse}")
    print(f"Random Forest R²: {r2}")
    
    # Salva il modello
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    print("Modello Random Forest salvato in 'models/random_forest_model.pkl'.")
    
    return rf_model
