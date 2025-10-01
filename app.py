import os
import time
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from data import (
    fetch_historic_bars,
    fetch_latest_trade,
)
from features import (
    make_supervised,
    add_technical_indicators,
    train_test_split_time,
)
from models_ml import (
    train_linear_regression,
    train_random_forest,
    predict_model,
)
from models_dl import (
    train_lstm,
    predict_lstm,
)
from utils import (
    plot_price_with_preds,
    plot_residuals,
    infer_frequency,
    human_dt,
)

st.set_page_config(
    page_title="Real-time Stocks: Viz + Predictions",
    page_icon="📈",
    layout="wide"
)

# Sidebar: configuration
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
timespan = st.sidebar.selectbox("Timespan", ["minute", "hour", "day"], index=0)
multiplier = st.sidebar.number_input("Bar multiplier", min_value=1, max_value=60, value=1, step=1)
lookback_days = st.sidebar.number_input("Historical lookback (days)", min_value=1, max_value=365, value=60, step=1)
forecast_horizon = st.sidebar.number_input("Forecast horizon (steps)", min_value=1, max_value=60, value=10, step=1)
model_name = st.sidebar.selectbox("Model", ["Baseline (Last Value)", "Linear Regression", "Random Forest", "LSTM"], index=2)
use_indicators = st.sidebar.checkbox("Add technical indicators (SMA, EMA, RSI, MACD)", value=True)
scale_features = st.sidebar.checkbox("Scale features (StandardScaler)", value=True)
polygon_api_key = st.sidebar.text_input("Polygon API Key (optional if using .env)", value="", type="password")
refresh_interval = st.sidebar.slider("Real-time refresh (seconds)", min_value=2, max_value=30, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: Use fewer features and shorter horizons for faster training.")

# Resolve API key
if polygon_api_key.strip():
    os.environ["JVNPy2_YzkZ27MPp4Zgdu9OqjLvo9ztp"] = polygon_api_key.strip()

st.title("📈 Real-time Stock Data Visualization + Predictions")
st.caption("Stream historical bars from Polygon.io, watch live quotes, and forecast short-horizon prices with ML/DL models.")

# Fetch historical
with st.spinner("Fetching historical bars..."):
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=int(lookback_days))
    df = fetch_historic_bars(
        ticker=ticker.upper(),
        multiplier=int(multiplier),
        timespan=timespan,
        start=start,
        end=end,
    )

if df is None or df.empty:
    st.error("No historical data received. Check ticker/timespan/multiplier/API key.")
    st.stop()

# Enrich features
base_col = "close"
freq = infer_frequency(timespan, multiplier)
df_feat = df.copy()

if use_indicators:
    df_feat = add_technical_indicators(df_feat)

# Supervised dataset
X, y, feature_names = make_supervised(df_feat, target_col=base_col, lags=20, horizon=int(forecast_horizon), scale=scale_features)

# Split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split_time(
    X, y, test_size=max(40, int(0.2 * len(X))), return_index=True
)

# Train/predict
model_obj = None
preds = None

if model_name == "Baseline (Last Value)":
    # Baseline: predict the last observed close (naive)
    preds = np.array([df_feat[base_col].iloc[idx - 1] if idx - 1 >= 0 else df_feat[base_col].iloc[0] for idx in idx_test])
else:
    with st.spinner(f"Training {model_name}..."):
        if model_name == "Linear Regression":
            model_obj = train_linear_regression(X_train, y_train)
            preds = predict_model(model_obj, X_test)
        elif model_name == "Random Forest":
            model_obj = train_random_forest(X_train, y_train, n_estimators=300, max_depth=None, n_jobs=-1)
            preds = predict_model(model_obj, X_test)
        elif model_name == "LSTM":
            model_obj, scaler_y = train_lstm(X_train, y_train, epochs=20, batch_size=64)
            preds = predict_lstm(model_obj, X_test, scaler_y)

# Align predictions and actuals
df_plot = df_feat.iloc[idx_test].copy()
df_plot["y_true"] = y_test
df_plot["y_pred"] = preds

# Layout: charts
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Historical prices and predictions")
    fig_price = plot_price_with_preds(df_feat, df_plot, price_col=base_col, freq=freq)
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    st.subheader("Evaluation (test set)")
    mae = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    st.metric(label="MAE", value=f"{mae:,.4f}")
    st.metric(label="RMSE", value=f"{rmse:,.4f}")
    fig_resid = plot_residuals(y_test, preds)
    st.plotly_chart(fig_resid, use_container_width=True)

st.markdown("---")
st.subheader("Live quote stream")
placeholder = st.empty()
run_stream = st.checkbox("Stream live quotes", value=False, help="Poll Polygon for latest trades and update in real-time.")

if run_stream:
    # Simple polling loop; Streamlit reruns on each iteration
    for _ in range(120):  # ~10 minutes at default 5s
        trade = fetch_latest_trade(ticker.upper())
        if trade is not None:
            lp = trade.get("price")
            ts = trade.get("sip_timestamp")
            placeholder.info(f"Last trade: {ticker.upper()} | Price: {lp} | Time: {human_dt(ts)}")
        else:
            placeholder.warning("No live trade data received.")
        time.sleep(refresh_interval)
        # Respect Streamlit rerun model: break if user unchecks
        if not st.session_state.get("run_stream", True) and not run_stream:
            break

st.caption("Note: Streaming uses REST polling to keep the setup simple. For ultra-low-latency, migrate to Polygon WebSocket.")
