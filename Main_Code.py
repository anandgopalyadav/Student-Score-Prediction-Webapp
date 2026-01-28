"""
Student Score Prediction using Linear Regression

This project predicts a student's score percentage based on hours studied
using Simple Linear Regression.

Features:
- Auto loads dataset from data/student_scores.csv
- Data exploration and visualization
- Train/test split and model training
- Prediction and evaluation metrics
- Saves predictions to outputs/predictions.csv
- Manual prediction mode with validations
- Ensures predicted scores are always between 0 and 100 (percentage rule)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

warnings.filterwarnings("ignore")


DEFAULT_DATA_PATH = "data/student_scores.csv"
DEFAULT_OUTPUT_PATH = "outputs/predictions.csv"


def load_dataset(file_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\nDataset not found: {file_path}\n"
            "✅ Please keep your dataset file inside this folder:\n"
            "   data/student_scores.csv\n"
        )
    return pd.read_csv(file_path)


def explore_dataset(df: pd.DataFrame) -> None:
    """Print basic dataset information."""
    print("\n[INFO] Dataset Shape:", df.shape)
    print("\n[INFO] First 10 rows:\n", df.head(10))
    print("\n[INFO] Dataset Summary:\n", df.describe())
    print("\n[INFO] Missing Values:\n", df.isnull().sum())


def visualize_data(df: pd.DataFrame) -> None:
    """Plot Hours vs Scores scatter plot."""
    df.plot(x="Hours", y="Scores", style="o")
    plt.title("Hours vs Scores")
    plt.xlabel("Hours Studied")
    plt.ylabel("Score Percentage")
    plt.show()


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features (X) and target (y)."""
    X = df[["Hours"]]
    y = df["Scores"]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def plot_regression_line(model: LinearRegression, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Plot regression line on training data."""
    plt.scatter(X_train.values, y_train.values)
    plt.plot(X_train.values, model.predict(X_train), linewidth=2)
    plt.title("Regression Line (Training Data)")
    plt.xlabel("Hours Studied")
    plt.ylabel("Score Percentage")
    plt.show()


def predict(model: LinearRegression, X_test: pd.DataFrame) -> np.ndarray:
    """Predict scores for test dataset."""
    return model.predict(X_test)


def evaluate_model(y_test: pd.Series, y_pred: np.ndarray) -> dict:
    """Calculate evaluation metrics and return them."""
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)

    n = len(y_test)
    k = 1
    adj_r2 = 1 - (((1 - r2) * (n - 1)) / (n - k - 1))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2
    }


def create_result_df(X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Create a DataFrame for Actual vs Predicted results."""
    return pd.DataFrame({
        "Hours": X_test["Hours"].values,
        "Actual Score": y_test.values,
        "Predicted Score": y_pred
    })


def save_predictions(result_df: pd.DataFrame, output_path: str = DEFAULT_OUTPUT_PATH) -> None:
    """Save predictions to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\n[SAVED] Predictions saved at: {output_path}")


def print_summary(metrics_dict: dict) -> None:
    """Print evaluation summary and explain performance."""
    print("\nModel Performance Summary")
    print("-" * 30)
    print(f"MAE        : {metrics_dict['MAE']:.4f}")
    print(f"MSE        : {metrics_dict['MSE']:.4f}")
    print(f"RMSE       : {metrics_dict['RMSE']:.4f}")
    print(f"R2 Score   : {metrics_dict['R2']:.4f}")
    print(f"Adj R2     : {metrics_dict['Adjusted_R2']:.4f}")

    r2 = metrics_dict["R2"]
    if r2 >= 0.80:
        print("\n✅ Model performance is GOOD")
        print("➡️ Reason: R² score is high, meaning the model explains most of the score variation.")
        print("➡️ Predictions are close to actual values for most test cases.")
    elif r2 >= 0.60:
        print("\n✅ Model performance is AVERAGE")
        print("➡️ Reason: R² score is moderate, so predictions are okay but not highly accurate.")
    else:
        print("\n⚠️ Model performance is LOW (needs improvement)")
        print("➡️ Reason: R² score is low, meaning model is not learning the pattern properly.")
        print("➡️ Try adding more features or using a better model.")


def predict_manual_input(model: LinearRegression) -> None:
    """
    Manual prediction mode.
    Validations:
    - Hours cannot be negative
    - Hours cannot be more than 20
    - Predicted score always between 0 and 100 (percentage)
    """
    print("\nManual Prediction Mode")
    print("-" * 30)
    print("Type hours like: 5 or 7.5")
    print("Type 'exit' to stop")

    while True:
        user_input = input("\nEnter Hours Studied: ").strip().lower()

        if user_input == "exit":
            print("\n✅ Exiting Manual Prediction Mode.")
            break

        try:
            hours = float(user_input)

            if hours < 0:
                print("[ERROR] Hours cannot be negative.")
                continue

            if hours > 20:
                print("❌ No one can study more than 20 hours in a day, please enter the right study hours.")
                continue

            predicted_score = model.predict([[hours]])[0]
            predicted_score = max(0, min(predicted_score, 100))

            print(f"✅ Predicted Score for {hours} hours = {predicted_score:.2f}%")

        except ValueError:
            print("[ERROR] Please enter a valid number (example: 4, 6.5) or type 'exit'.")


def main() -> None:
    """Run complete pipeline."""
    print("\nStudent Score Prediction (Linear Regression)")
    print("=" * 45)

    df = load_dataset()
    explore_dataset(df)
    visualize_data(df)

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    print("\n[MODEL] Training completed")
    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coef_)

    plot_regression_line(model, X_train, y_train)

    y_pred = predict(model, X_test)

    # ✅ Fix: Keep predictions between 0 and 100 (percentage)
    y_pred = np.clip(y_pred, 0, 100)

    result_df = create_result_df(X_test, y_test, y_pred)

    print("\n[RESULT] Actual vs Predicted:\n")
    print(result_df)

    model_metrics = evaluate_model(y_test, y_pred)
    print_summary(model_metrics)

    save_predictions(result_df)

    predict_manual_input(model)


if __name__ == "__main__":
    main()
