import datetime as dt
import pandas as pd
import plotly.graph_objects as go

def infer_frequency(timespan: str, multiplier: int) -> str:
    if timespan == "minute":
        return f"{multiplier} min"
    if timespan == "hour":
        return f"{multiplier} h"
    return f"{multiplier} day"

def plot_price_with_preds(df_all: pd.DataFrame, df_test: pd.DataFrame, price_col: str = "close", freq: str = "1 min"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_all.index, y=df_all[price_col],
        name="Price", mode="lines", line=dict(color="#1f77b4")
    ))
    if "y_true" in df_test.columns:
        fig.add_trace(go.Scatter(
            x=df_test.index, y=df_test["y_true"],
            name="Future True", mode="lines", line=dict(color="#ff7f0e")
        ))
    if "y_pred" in df_test.columns:
        fig.add_trace(go.Scatter(
            x=df_test.index, y=df_test["y_pred"],
            name="Future Pred", mode="lines", line=dict(color="#2ca02c", dash="dash")
        ))
    fig.update_layout(
        title=f"Price and Predictions ({freq})",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_residuals(y_true, y_pred):
    import numpy as np
    import plotly.express as px
    resid = y_true - y_pred
    df = pd.DataFrame({"Residuals": resid})
    fig = px.histogram(df, x="Residuals", nbins=30, title="Residuals Distribution")
    fig.update_layout(template="plotly_white")
    return fig

def human_dt(ts_ms: int) -> str:
    if ts_ms is None:
        return "n/a"
    return dt.datetime.utcfromtimestamp(ts_ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S UTC")
