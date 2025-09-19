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

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ days })
      });
      const data = await response.json();

      const formattedData = Array.from({ length: days }, (_, i) => ({
        day: i + 1,
        "Euro": data["Euro"]?.[i],
        "Japanese Yen": data["Japanese Yen"]?.[i],
        "Pound Sterling": data["Pound Sterling"]?.[i],
        "US Dollar": data["US Dollar"]?.[i]
      }));

      setPredictions(formattedData);
    } catch (error) {
      console.error("Error fetching predictions:", error);
    }
    setLoading(false);
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>📈 RupeeVision Predictor</h1>

      <input
        type="number"
        value={days}
        onChange={(e) => setDays(e.target.value)}
        min="1"
        max="30"
        style={{ margin: "10px", padding: "5px" }}
      />
      <button onClick={handlePredict}>Predict</button>

      {loading && <p>Loading...</p>}

      {predictions && (
        <div style={{ marginTop: "30px" }}>
          <h2>Currency Predictions (Next {days} Days)</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={predictions}>
              <CartesianGrid stroke="#ccc" />
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="Euro" stroke="#8884d8" />
              <Line type="monotone" dataKey="Japanese Yen" stroke="#82ca9d" />
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
