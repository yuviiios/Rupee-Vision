let chart;

// Function to predict for specific number of days
function predictDays(days) {
    document.getElementById("days").value = days;
    predictRates();
}

// Main prediction function
async function predictRates(latest = false) {
    const daysInput = document.getElementById("days");
    const days = Number(daysInput.value);
    
    if (!Number.isInteger(days) || days < 1 || days > 30) {
        alert("Please enter a valid number of days (1-30)");
        return;
    }
    
    const loader = document.getElementById("loader");
    if (loader.style.display === "block") return;
    loader.style.display = "block";
    document.getElementById("prediction-results").style.display = "none";
    
    try {
        const response = await fetch(`http://127.0.0.1:5000/predict${latest ? "?latest=1" : ""}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ days }),
        });

        if (!response.ok) {
            throw new Error("Failed to fetch data");
        }

        const data = await response.json();
        displayResults(data, days);
        document.getElementById("prediction-results").style.display = "block";
    } catch (error) {
        alert("Forecast unavailable. Check the API connection and try again.");
    } finally {
        loader.style.display = "none";
    }
}

// Function to display prediction results
function displayResults(data, days) {
    const tbody = document.querySelector("#resultTable tbody");
    tbody.innerHTML = "";
    const currencies = ["US Dollar", "Euro", "Japanese Yen", "Pound Sterling"];
    if (data.dates.length !== days || currencies.some(currency => {
        const values = data.predictions[currency]?.values;
        return !Array.isArray(values) || values.length !== days || !values.every(Number.isFinite);
    })) throw new Error("Incomplete forecast");

    const labels = data.dates;
    const datasets = currencies.map(currency => ({
        label: currency === "Japanese Yen" ? "Japanese Yen (100 JPY)" : currency,
        data: data.predictions[currency].values,
        borderColor: currencyColors[currency],
        fill: false,
        tension: 0.4
    }));
    document.getElementById("forecast-as-of").textContent = (data.archive_status === "unavailable" ? "Archive unavailable. Using observations through " : "Based on observations through ") + data.as_of;
    labels.forEach((date, index) => {
        const row = tbody.insertRow();
        row.insertCell().textContent = date;
        currencies.forEach(currency => {
            row.insertCell().textContent = data.predictions[currency].values[index].toFixed(3);
        });
    });

    updateChart(labels, datasets);
}

// Function to update the chart
function updateChart(labels, datasets) {
    const ctx = document.getElementById('rateChart').getContext('2d');
    if (chart) {
        chart.destroy();
    }
    
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#a0a0a0'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#a0a0a0'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#e0e0e0',
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(31, 27, 36, 0.9)',
                    titleColor: '#e0e0e0',
                    bodyColor: '#e0e0e0',
                    borderColor: '#4CAF50',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            },
            elements: {
                line: {
                    tension: 0.4
                },
                point: {
                    radius: 4,
                    hoverRadius: 6
                }
            }
        }
    });
}

// Function to fetch current exchange rates from API
async function fetchCurrentRates() {
    const ratesLoader = document.getElementById('rates-loader');
    ratesLoader.style.display = 'block';
    
    try {
        // Use ExchangeRate-API to get real rates
        const response = await fetch('https://open.er-api.com/v6/latest/INR');
        const data = await response.json();
        
        if (data && data.rates) {
            const currencies = ['USD', 'EUR', 'JPY', 'GBP'];
            const currentRatesContainer = document.getElementById('current-rates');
            const cards = document.createDocumentFragment();
            currencies.forEach(currency => {
                // Calculate inverse rate (currency to INR)
                const quote = data.rates[currency];
                if (!Number.isFinite(quote) || quote <= 0) throw new Error("Invalid rate");
                const card = document.createElement("div");
                card.className = "currency-card";
                card.textContent = currency + "/INR " + (1 / quote).toFixed(2);
                cards.append(card);
            });
            currentRatesContainer.replaceChildren(cards);
        } else {
            throw new Error('Invalid API response');
        }
    } catch (error) {
        console.error('Error fetching current rates:', error);
        document.getElementById("current-rates").textContent = "Current rates unavailable.";
    } finally {
        ratesLoader.style.display = 'none';
    }
}

async function fetchMarketSentiment() {
    const list = document.getElementById("market-insights");
    try {
        const response = await fetch("http://127.0.0.1:5000/sentiment");
        if (!response.ok) throw new Error("Sentiment unavailable");
        const { sentiments } = await response.json();
        const items = sentiments
            .filter(entry => entry.data_points > 0 && Number.isFinite(entry.sentiment_score))
            .map(entry => {
                const item = document.createElement("li");
                const name = entry.currency.replaceAll("_", " ");
                item.textContent = name + ": " + entry.sentiment_score + " (" + entry.data_points + " articles)";
                return item;
            });
        if (items.length) list.replaceChildren(...items);
        else list.textContent = "No scored articles available.";
    } catch {
        list.textContent = "Market sentiment unavailable.";
    }
}

const currencyColors = {
    "US Dollar": "#4CAF50",
    "Pound Sterling": "#2196F3",
    "Euro": "#FFC107",
    "Japanese Yen": "#E91E63"
};

// Function to update the live clock
function updateClock() {
    const now = new Date();
    const clockElement = document.getElementById('live-clock');
    
    const timeString = now.toLocaleTimeString('en-IN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
    });
    
    if (clockElement) {
        clockElement.textContent = timeString;
    }
}

// Initialize the app
window.onload = function() {
    // Start the clock
    updateClock();
    setInterval(updateClock, 1000);
    
    // Fetch current rates on page load
    fetchCurrentRates();
    
    // Set up interval to refresh rates every 5 minutes
    setInterval(fetchCurrentRates, 300000);
    fetchMarketSentiment();
    setInterval(fetchMarketSentiment, 120000);
    
};
