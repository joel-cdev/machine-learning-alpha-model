import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor


FEATURE_COLUMNS = [
    "Return",
    "MA_5",
    "MA_20",
    "Volatility_5",
    "Volatility_20",
    "Volume_Change",
    "Lag_Return_1",
    "Lag_Return_2",
    "Lag_Return_3",
]


def time_series_train_test_split(df, test_size=0.2):
    split_index = int(len(df) * (1 - test_size))

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = train[FEATURE_COLUMNS]
    y_train = train["Target"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["Target"]

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "mse": mean_squared_error(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }

    return predictions, metrics


def create_lstm_sequences(df, feature_columns, target_column="Target", sequence_length=20):
    scaler = MinMaxScaler()

    features = df[feature_columns].values
    target = df[target_column].values

    scaled_features = scaler.fit_transform(features)

    X = []
    y = []

    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i - sequence_length:i])
        y.append(target[i])

    return np.array(X), np.array(y), scaler


def train_lstm_model(X_train, y_train, epochs=10, batch_size=32):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model


def evaluate_lstm_model(model, X_test, y_test):
    predictions = model.predict(X_test).flatten()

    metrics = {
        "mse": mean_squared_error(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }

    return predictions, metrics