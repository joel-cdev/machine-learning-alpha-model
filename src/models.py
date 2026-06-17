from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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