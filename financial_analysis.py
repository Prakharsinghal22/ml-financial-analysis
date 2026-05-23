"""
Machine Learning for Financial Analysis
Authors: Prakhar Singhal, Priyanshu Saxena
Institution: Jai Narain College of Technology, Bhopal
Session: 2022-2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import os

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load and sort stock CSV by date."""
    df = pd.read_csv(filepath)
    df = df.sort_values(by=["Date"]).reset_index(drop=True)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nTotal stocks: {len(df.columns[1:])}")
    print("Stocks:", ", ".join(df.columns[1:].tolist()))
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

def eda(df: pd.DataFrame):
    """Print key EDA statistics."""
    print("\n── DataFrame Info ──")
    df.info()
    print("\n── Null Values ──")
    print(df.isnull().sum())
    print("\n── Descriptive Statistics ──")
    print(df.describe())
    print("\n── Mean Values ──")
    print(df.drop(columns=["Date"]).mean())
    print("\n── Standard Deviation (Volatility) ──")
    print(df.drop(columns=["Date"]).std())


# ─────────────────────────────────────────────
# 3. VISUALIZATIONS
# ─────────────────────────────────────────────

def show_plot(df: pd.DataFrame, title: str, save_path: str = None):
    """Static line plot for all stocks."""
    df.plot(x="Date", figsize=(15, 7), linewidth=2, title=title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize stock prices relative to the first trading day."""
    x = df.copy()
    for col in x.columns[1:]:
        x[col] = x[col] / x[col].iloc[0]
    return x


def interactive_plot(df: pd.DataFrame, title: str):
    """Interactive Plotly line chart."""
    fig = px.line(title=title)
    for col in df.columns[1:]:
        fig.add_scatter(x=df["Date"], y=df[col], name=col)
    fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    fig.show()


# ─────────────────────────────────────────────
# 4. DAILY RETURNS
# ─────────────────────────────────────────────

def daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage daily returns for all stocks."""
    dr = df.copy()
    for col in df.columns[1:]:
        for j in range(1, len(df)):
            dr[col].iloc[j] = ((df[col].iloc[j] - df[col].iloc[j - 1]) / df[col].iloc[j - 1]) * 100
        dr[col].iloc[0] = 0
    return dr


def plot_daily_returns(dr: pd.DataFrame, save_path: str = None):
    """Plot daily returns for all stocks."""
    dr.plot(x="Date", figsize=(15, 7), linewidth=1, title="STOCKS DAILY RETURNS")
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_daily_return_histograms(dr: pd.DataFrame, save_path: str = None):
    """Histogram of daily returns per stock."""
    dr.drop(columns=["Date"]).hist(figsize=(12, 10), bins=40)
    plt.suptitle("Daily Return Distributions", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_correlation_heatmap(dr: pd.DataFrame, save_path: str = None):
    """Correlation heatmap of daily returns."""
    cm = dr.drop(columns=["Date"]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdPu", linewidths=0.5)
    plt.title("Correlation Heatmap of Daily Returns")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    return cm


# ─────────────────────────────────────────────
# 5. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error


def create_features(df: pd.DataFrame, target: str) -> tuple:
    """Create lag-1 feature for predicting next-day closing price."""
    data = df[["Date", target]].copy()
    data["prev_close"] = data[target].shift(1)
    data.dropna(inplace=True)
    X = data[["prev_close"]]
    y = data[target]
    return X, y


def train_and_evaluate(df: pd.DataFrame, target: str = "AAPL") -> dict:
    """Train four ML models and return their R² and MAE scores."""
    X, y = create_features(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Support Vector Machine": SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
    }

    results = {}
    print(f"\n── Model Evaluation for {target} ──")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {"R2": round(r2, 4), "MAE": round(mae, 4), "model": model}
        print(f"  {name:30s} R²: {r2:.4f}   MAE: {mae:.4f}")

    return results, X_test, y_test


def plot_model_comparison(results: dict, save_path: str = None):
    """Bar chart comparing R² scores across models."""
    names = list(results.keys())
    scores = [results[n]["R2"] for n in names]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, scores, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    plt.ylim(0.99, 1.0005)
    plt.title("Model R² Score Comparison")
    plt.ylabel("R² Score")
    plt.xticks(rotation=15, ha="right")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00005,
                 f"{score:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def main(data_path: str = "data/stock.csv"):
    os.makedirs("outputs", exist_ok=True)

    # Load
    df = load_data(data_path)

    # EDA
    eda(df)

    # Raw price plot
    show_plot(df, "RAW STOCK PRICES (WITHOUT NORMALIZATION)", "outputs/raw_prices.png")

    # Normalized price plot
    show_plot(normalize(df), "NORMALIZED STOCK PRICES", "outputs/normalized_prices.png")

    # Daily returns
    dr = daily_return(df)
    plot_daily_returns(dr, "outputs/daily_returns.png")
    plot_daily_return_histograms(dr, "outputs/return_histograms.png")
    corr = plot_correlation_heatmap(dr, "outputs/correlation_heatmap.png")
    print("\n── Correlation Matrix ──\n", corr)

    # Model training (using AAPL as default target)
    results, X_test, y_test = train_and_evaluate(df, target="AAPL")
    plot_model_comparison(results, "outputs/model_comparison.png")

    print("\n✅ Analysis complete. Outputs saved to /outputs/")


if __name__ == "__main__":
    main()
