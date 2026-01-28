from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use("Agg")  # for server-side rendering
import matplotlib.pyplot as plt

import io
import base64

app = Flask(__name__)

DATA_PATH = "data/student_scores.csv"

# Load dataset and train model once
df = pd.read_csv(DATA_PATH)
X = df[["Hours"]]
y = df["Scores"]

model = LinearRegression()
model.fit(X, y)


def generate_plot(hours_input=None, predicted_score=None):
    """
    Creates a scatter plot of Hours vs Scores
    + Regression line
    + Predicted point (if user enters input)

    Returns:
        base64 image string to show in HTML
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter plot of dataset
    ax.scatter(df["Hours"], df["Scores"], alpha=0.6, label="Actual Data")

    # Regression line
    x_line = np.linspace(df["Hours"].min(), df["Hours"].max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, linewidth=2, label="Regression Line")

    # Predicted point
    if hours_input is not None and predicted_score is not None:
        ax.scatter([hours_input], [predicted_score], s=120, label="Your Prediction")

    ax.set_title("Study Hours vs Score Prediction")
    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Convert plot to PNG and then base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    return img_base64


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    hours_value = ""

    # Default graph (only dataset + regression line)
    graph_img = generate_plot()

    if request.method == "POST":
        hours_value = request.form.get("hours", "").strip()

        try:
            hours = float(hours_value)

            if hours < 0:
                error = "Hours cannot be negative."
            elif hours > 20:
                error = "No one can study more than 20 hours in a day. Please enter correct study hours."
            else:
                predicted_score = model.predict([[hours]])[0]
                predicted_score = float(np.clip(predicted_score, 0, 100))

                prediction = round(predicted_score, 2)

                # Graph with predicted point
                graph_img = generate_plot(hours_input=hours, predicted_score=predicted_score)

        except ValueError:
            error = "Please enter a valid number (example: 5 or 7.5)."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        hours_value=hours_value,
        graph_img=graph_img
    )


if __name__ == "__main__":
    app.run(debug=True)
