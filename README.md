# 📈 Machine Learning for Financial Analysis

> **Major Project-II** | B.Tech in AI & ML | Jai Narain College of Technology, Bhopal  
> **Authors:** Prakhar Singhal [0131CL221071] · Priyanshu Saxena [0131CL221078]  
> **Guide:** Ms. Usha Patel | **HOD:** Dr. Ayonija Pathre  
> **Session:** 2022–2026

---

## 🧠 Overview

This project applies machine learning techniques to financial datasets to:
- Uncover hidden patterns in stock price data
- Forecast future prices with regression models
- Assess risk via volatility and correlation analysis
- Visualize market trends interactively

---

## 📁 Project Structure

```
ml-financial-analysis/
├── data/
│   └── stock.csv              # Historical stock prices (AAPL, BA, AMZN, TSLA, GOOG, etc.)
├── notebooks/
│   └── ML_Financial_Analysis.ipynb   # Step-by-step Jupyter notebook
├── src/
│   └── financial_analysis.py  # Modular Python implementation
├── outputs/                   # Generated plots and figures
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Historical daily closing prices (2012–2020) for 9 instruments:

| Ticker | Company         |
|--------|-----------------|
| AAPL   | Apple           |
| BA     | Boeing          |
| T      | AT&T            |
| MGM    | MGM Resorts     |
| AMZN   | Amazon          |
| IBM    | IBM             |
| TSLA   | Tesla           |
| GOOG   | Google          |
| sp500  | S&P 500 Index   |

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ml-financial-analysis.git
cd ml-financial-analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook notebooks/ML_Financial_Analysis.ipynb
```

---

## 🔬 Features & Methodology

### 1. Exploratory Data Analysis (EDA)
- Dataset info, null-value checks, descriptive statistics
- Mean & standard deviation (volatility) per stock

### 2. Visualizations
| Plot | Description |
|------|-------------|
| Raw Prices | Absolute stock prices over time |
| Normalized Prices | Price growth relative to start date |
| Daily Returns | Percentage change each trading day |
| Return Histograms | Distribution of daily returns per stock |
| Correlation Heatmap | Pairwise return correlations |

### 3. Machine Learning Models
All models trained on a lag-1 feature (previous day's closing price → next day's price):

| Model | R² Score |
|-------|----------|
| Linear Regression | 0.9987 |
| Decision Tree | 0.9999 |
| Random Forest | 0.9999 |
| Support Vector Machine | 0.9999 |

Evaluation metrics: **R² Score**, **Mean Absolute Error (MAE)**

---

## 🛠️ Tech Stack

- **Language:** Python 3.6+
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **ML:** Scikit-learn (Linear Regression, Decision Tree, Random Forest, SVR)
- **Environment:** Jupyter Notebook / Anaconda

---

## ⚠️ Limitations

- Models rely on a single lag feature; real-world models use richer feature sets
- High R² scores may indicate overfitting on simple lag-1 input
- Does not account for external market events or news sentiment
- No real-time data integration

---

## 🚀 Future Scope

- Real-time data via Yahoo Finance / Alpha Vantage APIs
- LSTM / Transformer models for time-series forecasting
- NLP-based sentiment analysis from financial news
- Portfolio optimization (Markowitz / Sharpe ratio)
- Explainable AI (SHAP values) for model interpretability
- Deployment via Flask / FastAPI REST API

---

## 📜 License

This project is submitted for academic purposes under RGPV, Bhopal.  
Free to use for educational and research purposes.
