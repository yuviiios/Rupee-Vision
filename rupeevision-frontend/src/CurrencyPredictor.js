import React, { useState } from "react";
import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";

function CurrencyPredictor() {
  const [days, setDays] = useState(7);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");
  const horizon = Number(days);
  const valid = Number.isInteger(horizon) && horizon >= 1 && horizon <= 30;

  const handlePredict = async (latest) => {
    setLoading(true);
    setError("");
    setPredictions(null);
    try {
      const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || "http://127.0.0.1:5000"}/predict${latest ? "?latest=1" : ""}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ days: horizon })
      });
      if (!response.ok) throw new Error("Forecast request failed");
      const { dates, predictions, as_of, archive_status } = await response.json();
      const currencies = ["Euro", "Japanese Yen", "Pound Sterling", "US Dollar"];
      if (dates.length !== horizon || currencies.some(name => {
        const values = predictions[name]?.values;
        return !Array.isArray(values) || values.length !== horizon || !values.every(Number.isFinite);
      })) throw new Error("Incomplete forecast");
      const rows = dates.map((date, i) => ({
        date,
        ...Object.fromEntries(currencies.map(name => [name, predictions[name].values[i]]))
      }));
      setPredictions({ rows, asOf: as_of, archiveStatus: archive_status });
    } catch {
      setError("Forecast unavailable. Check the API connection and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>📈 RupeeVision Predictor</h1>

      <input
        aria-label="Forecast days (1-30)"
        type="number"
        value={days}
        onChange={(e) => setDays(e.target.value)}
        min="1"
        max="30"
        style={{ margin: "10px", padding: "5px" }}
      />
      <button disabled={loading || !valid} onClick={() => handlePredict(false)}>Predict</button>
      <button disabled={loading || !valid} onClick={() => handlePredict(true)}>Predict with latest data</button>

      {error && <p role="alert">{error}</p>}
      {loading && <p>Loading...</p>}

      {predictions && (
        <div style={{ marginTop: "30px" }}>
          <h2>Currency Predictions ({predictions.rows.length} days from {predictions.asOf})</h2>
          {predictions.archiveStatus === "unavailable" && <p role="status">Archive unavailable. Showing a forecast from the last available dataset.</p>}
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={predictions.rows}>
              <CartesianGrid stroke="#ccc" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="Euro" stroke="#8884d8" />
              <Line type="monotone" dataKey="Japanese Yen" name="Japanese Yen (100 JPY)" stroke="#82ca9d" />
              <Line type="monotone" dataKey="Pound Sterling" stroke="#ff7300" />
              <Line type="monotone" dataKey="US Dollar" stroke="#387908" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export default CurrencyPredictor;
