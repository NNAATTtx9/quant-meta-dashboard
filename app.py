import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(layout="wide")
st.title("🧠 FINAL BOSS Hedge Fund AI System")

# -----------------------
# INPUTS
# -----------------------
tickers_input = st.sidebar.text_input("Tickers", "AAPL,MSFT,TSLA,NVDA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start = st.sidebar.date_input("Start", pd.to_datetime("2022-01-01"))
end = st.sidebar.date_input("End", pd.to_datetime("today"))

risk_limit = st.sidebar.slider("Risk Limit (Vol Cap)", 0.01, 0.5, 0.15)
mc_sims = st.sidebar.slider("Monte Carlo Sims", 100, 1500, 500)

if len(tickers) == 0:
    st.stop()

# -----------------------
# DATA ENGINE
# -----------------------
@st.cache_data
def load_data():
    df = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    return df.dropna()

data = load_data()

if data is None or data.empty:
    st.error("No data loaded")
    st.stop()

st.subheader("Market Data")
st.dataframe(data.tail())

# -----------------------
# RETURNS
# -----------------------
returns = data.pct_change().dropna()

# -----------------------
# REGIME DETECTION (HMM proxy)
# -----------------------
vol = returns.rolling(20).std().mean(axis=1)

regime = pd.qcut(vol, 3, labels=[0, 1, 2])  # low, mid, high

# amplify risk in high regime
regime_weight = regime.astype(float).fillna(1)

# -----------------------
# EWMA VOLATILITY (GARCH proxy)
# -----------------------
ewma_vol = returns.ewm(span=20).std().mean(axis=1)

# -----------------------
# FEATURE ENGINEERING (AI layer)
# -----------------------
X = pd.DataFrame(index=returns.index)

X["ret"] = returns.mean(axis=1)
X["vol"] = ewma_vol
X["momentum"] = returns.rolling(5).mean().mean(axis=1)
X["regime"] = regime_weight.values

y = (returns.mean(axis=1).shift(-1) > 0).astype(int)

df_ai = pd.concat([X, y], axis=1).dropna()

X = df_ai.iloc[:, :-1]
y = df_ai.iloc[:, -1]

# -----------------------
# AI MODEL (ensemble base)
# -----------------------
model = LogisticRegression(max_iter=2000)

tscv = TimeSeriesSplit(n_splits=5)

pred_list = []

for train, test in tscv.split(X):
    model.fit(X.iloc[train], y.iloc[train])
    pred_list.append(model.predict_proba(X.iloc[test])[:, 1])

latest_prob = model.predict_proba(X.iloc[-1:])[0][1]

st.subheader("AI Signal Probability")
st.write(round(latest_prob, 3))

# -----------------------
# RISK PARITY ENGINE
# -----------------------
inv_vol = 1 / (returns.std() + 1e-9)
risk_weights = inv_vol / inv_vol.sum()

# -----------------------
# SHARPE OPTIMIZER (random search)
# -----------------------
mean_vec = returns.mean().values
cov = returns.cov().values

best_w = None
best_score = -np.inf

for _ in range(4000):
    w = np.random.random(len(tickers))
    w /= w.sum()

    port_ret = np.dot(w, mean_vec)
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))

    score = port_ret / (port_vol + 1e-9)

    # risk penalty (regime-aware)
    score *= (1 - regime_weight.iloc[-1] * 0.3)

    if port_vol > risk_limit:
        score *= 0.5

    if score > best_score:
        best_score = score
        best_w = w

# -----------------------
# FINAL WEIGHT ENGINE (multi-intelligence blend)
# -----------------------
ai_weight = latest_prob
ai_weight = np.clip(ai_weight, 0.3, 0.7)

weights = (
    0.45 * best_w +
    0.35 * risk_weights.values +
    0.20 * ai_weight
)

weights /= np.sum(weights)

st.subheader("Portfolio Weights")

for t, w in zip(tickers, weights):
    st.write(f"{t}: {round(w,3)}")

# -----------------------
# PORTFOLIO BACKTEST
# -----------------------
portfolio_ret = returns.dot(weights)

cum = (1 + portfolio_ret).cumprod()

sharpe = portfolio_ret.mean() / (portfolio_ret.std() + 1e-9) * np.sqrt(252)

drawdown = cum / cum.cummax() - 1

col1, col2, col3 = st.columns(3)
col1.metric("Sharpe", round(sharpe, 2))
col2.metric("Max DD", round(drawdown.min(), 3))
col3.metric("Regime", int(regime.iloc[-1]))

st.line_chart(cum)

# -----------------------
# FINAL BOSS MONTE CARLO STRESS TEST
# -----------------------
st.subheader("Stress Test Engine")

mean_vec = returns.mean().values
cov = returns.cov().values

def stress_mc():
    results = []

    for _ in range(mc_sims):
        value = 1
        path = [value]

        for _ in range(30):

            shock = np.random.multivariate_normal(mean_vec, cov)

            port_r = np.dot(weights, shock)

            # regime amplification
            if regime_weight.iloc[-1] == 2:
                port_r *= 1.6

            # volatility scaling
            port_r *= (1 + ewma_vol.iloc[-1].mean())

            value *= np.exp(port_r)
            path.append(value)

        results.append(path)

    return np.array(results)

if st.button("RUN FINAL BOSS STRESS TEST"):

    sims = stress_mc()
    final = sims[:, -1]

    st.metric("Expected Return", round(np.mean(final), 3))
    st.metric("VaR (5%)", round(np.percentile(final, 5), 3))
    st.metric("Worst Case", round(np.min(final), 3))

    st.line_chart(sims.mean(axis=0))