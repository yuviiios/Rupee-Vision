# RupeeVision: Currency Exchange Rate Predictor

RupeeVision is a web application that predicts future currency exchange rates for several major currencies against the Indian Rupee (INR). It forecasts the next 1 to 30 fixing sessions using the model selected for each currency during walk-forward evaluation.

## Features

*   Predicts exchange rates for:
    *   US Dollar (USD)
    *   Euro (EUR)
    *   Japanese Yen (JPY)
    *   Pound Sterling (GBP)
*   User-friendly interface to select the number of sessions for prediction.
*   Visualizes the predicted rates using interactive line charts.
*   RESTful API backend built with Flask.
*   Streamlit frontend for a responsive user experience.

## Technologies Used

### Backend

*   **Flask**: A micro web framework for Python.
*   **TensorFlow/Keras**: For building and training the LSTM model.
*   **Pandas**: For data manipulation and analysis.
*   **scikit-learn**: For data preprocessing (scaling).
*   **Joblib**: For saving and loading the scaler.

### Frontend

*   **Streamlit** renders the forecast controls, charts and downloadable table.

## Setup and Usage

### Backend (Flask)

From the cloned repository root, activate a Python 3.11 virtual environment and run:

```bash
python -m pip install -e .
rupeevision-serve
```

The API listens on `http://127.0.0.1:5000`. Open `/currencies` to see the latest observation date, or `/predict?days=7` to request a forecast. Forecast dates start after the last observation in the dataset. The `/ready` endpoint reports stale data when it exceeds the configured age limit.

### Streamlit dashboard

Start the API with `rupeevision-serve`. From a second terminal in the same
Python environment and repository root, run:

```bash
python -m streamlit run streamlit_app.py
```

Open `http://localhost:8501`, select a horizon, and choose **Run forecast**.
Enable **Check the latest RBI/FBIL observations** to refresh the data.
The dashboard shows the model and quote unit for each currency and offers a
CSV download. Set `RUPEEVISION_API_URL` to point it at a deployed Flask API.

## Disclaimer

This project is for educational purposes only. The predicted exchange rates are not guaranteed to be accurate and should not be used for making financial decisions.

## Forecast with the latest observations

Enable **Check the latest RBI/FBIL observations** in Streamlit and select **Run forecast**, or request
`GET /predict?days=7&latest=1`. The API checks the [RBI/FBIL historical archive](https://github.com/the-solipsist/rbi-forex-reference-rates),
validates its dates, rates and units, and uses the existing trained models with
the refreshed observations. Yen forecasts are INR per 100 JPY.

The response includes the dataset date in `as_of`. An `archive_status` of
`verified` means the archive was checked successfully. If the archive is
unavailable or invalid, `unavailable` accompanies the forecast from the last
usable snapshot. Each server process keeps its snapshot in memory and reuses a
successful check for five minutes. After a restart it starts with the configured
local CSV. Ordinary forecasts require no network access.
## Retraining and release selection

The November 2026 release uses 6,631 jointly observed sessions through November
25. The [pinned input](data/rbi_fbil_13f8f41.csv) is the long-format CSV from
[archive revision 13f8f41](https://github.com/the-solipsist/rbi-forex-reference-rates/tree/13f8f41326c3677c8ebd715fc458a8b7dfb21f0c).
Use the installed project environment from the repository root:

```bash
python benchmark_sessions.py
python retrain_release.py --output reports/new-session-release
```

The first command compares monthly Ridge and boosting refits, the legacy frozen
LSTM and last rate at 1, 7 and 30 sessions. The second fits fresh 30-output LSTMs
and two Ridge regularization strengths for each currency. It creates a new
output directory and leaves the active release usable during fitting. To serve
a reviewed release, point `RUPEEVISION_MODEL_DIR` at that directory and restart
the API. The active model names are included in forecast responses.

For the release experiment, each fold trains on labels fully observed before
the evaluation period. LSTM epoch selection uses a chronological inner split,
a 30-session gap, early stopping and L2 regularization. Final fits start from
scratch. The evaluation uses four consecutive six-month validation periods
starting June 2024, followed by the previously examined June–November 2026
period. Only origins with all 30 outcomes observed inside each block are scored.
This yields 91 paired origins in the last block. The monthly comparison uses
all available origins separately at each horizon.

Candidates must beat last rate by at least 2% on average across validation
folds and improve in at least three folds. Confirmation requires positive
average gain and no horizon worse by more than 2% in the final block. Scores
weight the 1-, 7- and 30-session relative MAE gains equally.

| Currency | Selected model | 1-session MAE | 7-session MAE | 30-session MAE |
|---|---|---:|---:|---:|
| USD | Ridge, alpha 10000 | 0.177 | 0.672 | 0.921 |
| EUR | Last rate | 0.303 | 1.095 | 2.375 |
| GBP | Last rate | 0.390 | 1.409 | 2.001 |
| JPY (100 yen) | Last rate | 0.205 | 0.730 | 1.222 |

All errors are in INR. EUR's fresh LSTM gained 3.76% in-sample and lost 2.28%
on the final evaluation. JPY's fresh LSTM stayed within 0.1% of last rate on
both samples. The latter is consistent with weak signal or underfitting.
The more flexible candidate did not establish an advantage. USD Ridge gained
1.33% overall on the final block, including a 0.46% loss at seven sessions.
Subsequent published observations provide the next prospective evaluation.

[Fit diagnostics](artifacts/session_20261125/diagnostics.csv) ·
[Active release](artifacts/session_20261125/release.json) ·
[Monthly comparison](artifacts/session_20261125/monthly_scorecard.csv)
