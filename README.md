# 📈 AI Financial Analyst Dashboard

An end-to-end AI-powered financial analysis dashboard that simulates the workflow of a real-world financial analyst using machine learning, technical indicators, and interactive visualizations.

This project ingests historical stock market data, performs feature engineering on time-series data, computes technical indicators, and applies machine learning models to analyze and predict market behavior through an interactive dashboard.

## Features
- Stock price data ingestion using Yahoo Finance
- Technical indicator computation (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- Time-series feature engineering on OHLCV data
- Machine learning models for trend and price-movement analysis
- Interactive dashboard built with Streamlit
- Clean, modular, production-style project structure

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- yfinance
- Matplotlib, Plotly
- Streamlit
- Git & GitHub

## Project Structure
ai_financial_analyst_dashboard/
├── data/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── indicators.py
│   ├── model.py
│   └── evaluation.py
├── app/
│   └── dashboard.py
├── requirements.txt
└── README.md

## How to Run
1. Clone the repository  
2. Create and activate a virtual environment  
3. Install dependencies  
4. Run the Streamlit app

## Use Cases
- AI-driven stock market analysis
- Financial data science experimentation
- Machine learning portfolio project
- Decision-support system for trading insights

## Author
Siddhesh Patil  
Master’s in Artificial Intelligence, DePaul University
