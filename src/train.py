import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

from utils import (
    load_data,
    remove_high_missing,
    fill_missing,
    encode_and_split
)


def main():
    print("Loading and preprocessing data...")
    df = load_data("../data/train.csv")
    df = remove_high_missing(df, threshold=0.40)
    df = fill_missing(df)
    X_train, X_valid, y_train, y_valid = encode_and_split(
        df,
        target_col="SalePrice",
        test_size=0.2,
        random_state=42,
    )
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    print("Training and evaluating models...")
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        results[name] = rmse
        print(f"{name}: RMSE = {rmse:.2f}")

    best_name = min(results, key=results.get)
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (RMSE = {results[best_name]:.2f})")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "../models/house_price_model.pkl")
    print(f"Best model Saved: models/house_price_model.pkl")


if __name__ == "__main__":
    main()
