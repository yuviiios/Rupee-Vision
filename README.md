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

1.  **Navigate to the project directory:**
    ```bash
    cd RupeeVision
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file first. See below.)*

3.  **Run the Flask application:**
    ```bash
    python flaskapp-1.py
    ```
    The backend server will start on `http://127.0.0.1:5000`.

### Frontend (React)

1.  **Navigate to the frontend directory:**
    ```bash
    cd rupeevision-frontend
    ```

2.  **Install the required Node.js packages:**
    ```bash
    npm install
    ```

3.  **Run the React application:**
    ```bash
    npm start
    ```
    The frontend development server will start on `http://localhost:3000`. You can view the application in your browser at this address.

### Creating `requirements.txt`

If you don't have a `requirements.txt` file, you can create one with the following content:

```
flask
pandas
numpy
joblib
tensorflow
flask-cors
```

## Disclaimer

This project is for educational purposes only. The predicted exchange rates are not guaranteed to be accurate and should not be used for making financial decisions.
