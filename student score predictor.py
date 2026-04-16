"""
============================================================
 Student Score Predictor — Simple Linear Regression
 Author : Maryam Khalid
 Purpose: Predict exam marks from study hours using ML + GUI
============================================================
"""

# ── Imports ───────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import tkinter as tk
from tkinter import messagebox

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ── 1. Dataset ────────────────────────────────────────────
def load_data():
    data = {
        "Hours": [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9],
        "Marks": [28, 33, 38, 50, 52, 55, 60, 65, 66, 70, 74, 80, 83, 90, 95],
    }

    df = pd.DataFrame(data)
    return df


# ── 2. Preprocessing ──────────────────────────────────────
def preprocess(df):
    X = df[["Hours"]].values
    y = df["Marks"].values

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ── 3. Train Model ────────────────────────────────────────
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ── 4. Evaluate (simple) ───────────────────────────────────
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nPredictions:")
    for a, p in zip(y_test, y_pred):
        print(f"Actual: {a}, Predicted: {p:.2f}")

    return y_pred


# ── 5. Plot ───────────────────────────────────────────────
def plot_results(model, df):
    plt.scatter(df["Hours"], df["Marks"])
    plt.plot(df["Hours"], model.predict(df[["Hours"]]))
    plt.xlabel("Hours Studied")
    plt.ylabel("Marks")
    plt.title("Student Score Prediction")
    plt.show()


# ── 6. TKINTER GUI ────────────────────────────────────────
class ScoreApp:
    def __init__(self, root, model):
        self.model = model
        self.root = root

        self.root.title("Student Score Predictor")
        self.root.geometry("350x300")

        tk.Label(root, text="Enter Study Hours", font=("Arial", 14)).pack(pady=10)

        self.entry = tk.Entry(root, font=("Arial", 12))
        self.entry.pack(pady=5)

        tk.Button(root, text="Predict", command=self.predict,
                  bg="blue", fg="white").pack(pady=10)

        self.result = tk.Label(root, text="", font=("Arial", 12))
        self.result.pack(pady=10)

    def predict(self):
        try:
            hours = float(self.entry.get())
            pred = self.model.predict([[hours]])[0]

            pred = max(0, min(100, pred))

            self.result.config(text=f"Predicted Marks: {pred:.2f}")

        except:
            messagebox.showerror("Error", "Please enter a valid number")


# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":

    # Load data
    df = load_data()

    # Split data
    X_train, X_test, y_train, y_test = preprocess(df)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Plot
    plot_results(model, df)

    # GUI
    root = tk.Tk()
    app = ScoreApp(root, model)
    root.mainloop()