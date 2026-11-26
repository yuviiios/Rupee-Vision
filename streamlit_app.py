"""Streamlit client for the RupeeVision forecast API."""
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

from rupeevision.core.constants import CURRENCY_CODES

API = os.environ.get("RUPEEVISION_API_URL", "http://127.0.0.1:5000").rstrip("/")
UNITS = {code: f"INR per {100 if code == 'JPY' else 1} {code}" for code in CURRENCY_CODES.values()}


def request_forecast(sessions, latest):
    params = {"days": sessions}
    if latest:
        params["latest"] = 1
    response = requests.get(f"{API}/predict", params=params, timeout=(5, 120))
    response.raise_for_status()
    payload = response.json()
    dates = pd.to_datetime(payload["dates"], errors="raise")
    rates = pd.DataFrame(
        {code: payload["predictions"][name]["values"] for name, code in CURRENCY_CODES.items()},
        index=pd.Index(dates, name="Date"),
    )
    if len(rates) != sessions or not np.isfinite(rates.to_numpy()).all() or (rates <= 0).any().any():
        raise ValueError("Incomplete forecast")
    return payload, rates


st.set_page_config(page_title="RupeeVision · Exchange forecasts", layout="wide")
st.title("RupeeVision")
st.caption("Indian rupee exchange-rate forecasts")
with st.sidebar:
    st.header("Forecast settings")
    with st.form("forecast"):
        sessions = st.slider("Sessions ahead", 1, 30, 7)
        latest = st.checkbox("Check the latest RBI/FBIL observations")
        submit = st.form_submit_button("Run forecast", type="primary")

if submit or "attempted" not in st.session_state:
    st.session_state.attempted = True
    try:
        with st.spinner("Loading exchange rates…"):
            st.session_state.result = request_forecast(sessions, latest)
        st.session_state.error = False
    except (requests.RequestException, ValueError, KeyError, TypeError):
        st.session_state.error = True
        st.session_state.pop("result", None)

if st.session_state.get("error"):
    st.error("Forecast unavailable. Check that the Flask API is running, then select Run forecast.")
    st.stop()

payload, rates = st.session_state.result
if payload.get("archive_status") == "unavailable":
    st.warning("The archive check failed. These forecasts use the last available observations.")
st.caption(f"Observations through {payload['as_of']} · {len(rates)} forecast sessions")
for column, code in zip(st.columns(4), rates.columns):
    column.metric(f"{code} · session {len(rates)}", f"₹{rates[code].iloc[-1]:,.4f}", help=UNITS[code])

code = st.selectbox("Currency", list(UNITS))
model = payload.get("models", {}).get(code, "Legacy LSTM")
st.subheader(f"{code} forecast")
st.caption(f"{UNITS[code]} · Model: {model}")
if len(rates) > 1:
    st.line_chart(rates[[code]], color="#197356", y_label=UNITS[code], height=310)
if model == "Last rate":
    st.caption("This model carries the latest observed rate through each session.")
if payload.get("date_basis") == "estimated_weekdays":
    st.caption("Dates are estimated weekdays. Market holidays can shift the actual fixing.")
with st.expander("All forecast values", expanded=True):
    display = rates.rename(columns=UNITS).copy()
    display.index = display.index.strftime("%Y-%m-%d")
    st.dataframe(display, width="stretch")
    st.download_button("Download forecast CSV", display.to_csv(), "rupeevision-forecast.csv", "text/csv")
