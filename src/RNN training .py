import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow CPU instruction warnings

import pandas as pd
import numpy as np

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import Callback
from tqdm import tqdm
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm.keras import TqdmCallback


# Upload the data: 
df = pd.read_csv('Hourly_merged_data.csv')
print(df.columns)
print(df['Return'][740])
length = len(df)
print(f"Length of the data: {length}")

model = Sequential([
    Dense(32, activation='relu', input_shape=(1,)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X  = np.array(df['Positive_score']).reshape(-1, 1)
Y  = np.array(df['Return_sign'])

# Split the data into training, test and validation sets:
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

print(X_train.shape)

# Check for data leakage
if len(set(X_train.flatten()).intersection(set(X_val.flatten()))) == 0 : 
    print("Data leakage detected!")

# Check class distribution
print("Training set class distribution:")
print(pd.Series(Y_train).value_counts())
print("Validation set class distribution:")
print(pd.Series(Y_val).value_counts())

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

history = model.fit(
    X_train, Y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping, model_checkpoint, TqdmCallback(verbose=1)],
    verbose=0
)

# Evaluar en el conjunto de validación o de prueba
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

print("The run is complete")

#print(df.shape)
#print(train_data.shape)