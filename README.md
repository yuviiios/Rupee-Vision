# RupeeVision: Currency Exchange Rate Predictor

RupeeVision is a web application that predicts future currency exchange rates for several major currencies against the Indian Rupee (INR). It uses a Long Short-Term Memory (LSTM) neural network to forecast the rates for the next 1 to 30 days.

## Features

*   Predicts exchange rates for:
    *   US Dollar (USD)
    *   Euro (EUR)
    *   Japanese Yen (JPY)
    *   Pound Sterling (GBP)
*   User-friendly interface to select the number of days for prediction.
*   Visualizes the predicted rates using interactive line charts.
*   RESTful API backend built with Flask.
*   React-based frontend for a responsive user experience.

## Technologies Used

### Backend

*   **Flask**: A micro web framework for Python.
*   **TensorFlow/Keras**: For building and training the LSTM model.
*   **Pandas**: For data manipulation and analysis.
*   **scikit-learn**: For data preprocessing (scaling).
*   **Joblib**: For saving and loading the scaler.

### Frontend

*   **React**: A JavaScript library for building user interfaces.
*   **Recharts**: A composable charting library built on React components.
*   **Flask-CORS**: for Cross Origin Resource Sharing.

## Setup and Usage

### Backend (Flask)

From the cloned repository root, activate a Python 3.11 virtual environment and run:

```bash
python -m pip install -e .
rupeevision-serve
```

The API listens on `http://127.0.0.1:5000`. Open `/currencies` to see the latest observation date, or `/predict?days=7` to request a forecast. Forecast dates start after the last observation in the dataset. The `/ready` endpoint reports stale data when it exceeds the configured age limit.

### Frontend (React)

1.  **Navigate to the frontend directory:**
    ```bash
    cd rupeevision-frontend
    ```

2.  **Install the required Node.js packages:**
    ```bash
    npm ci
    ```

3.  **Run the React application:**
    ```bash
    npm start
    ```
    The frontend development server will start on `http://localhost:3000`. You can view the application in your browser at this address.

Set `REACT_APP_API_BASE_URL` before starting or building React to use a different API host. Configure `RUPEEVISION_ALLOWED_ORIGINS` on the backend to allow the frontend origin.

## Disclaimer

This project is for educational purposes only. The predicted exchange rates are not guaranteed to be accurate and should not be used for making financial decisions.

## Forecast with the latest observations

Select **Predict with latest data** in either interface, or request
`GET /predict?days=7&latest=1`. The API checks the [RBI/FBIL historical archive](https://github.com/the-solipsist/rbi-forex-reference-rates),
validates its dates, rates and units, and uses the existing trained models with
the refreshed observations. Yen forecasts are INR per 100 JPY.

The response includes the dataset date in `as_of`. An `archive_status` of
`verified` means the archive was checked successfully. If the archive is
unavailable or invalid, `unavailable` accompanies the forecast from the last
usable snapshot. Each server process keeps its snapshot in memory and reuses a
successful check for five minutes. After a restart it starts with the configured
local CSV. Ordinary forecasts require no network access.
