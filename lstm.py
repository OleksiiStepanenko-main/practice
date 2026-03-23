import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random
import os

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set seeds to stabilize the neural network's random initialization as much as possible
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 1. Load and prep data
df = pd.read_csv("co2-data.csv")
years = [str(year) for year in range(1970, 2025)]
country_data = {}

china_mask = (df['Country'] == 'China') & (df['Sub-country'] == 'Mainland')
country_data['China'] = df.loc[china_mask, years].values.flatten()

target_countries = ['Ukraine', 'Germany', 'United States', 'Japan']
for country in target_countries:
    mask = (df['Country'] == country)
    country_data[country] = df.loc[mask, years].sum(axis=0).values

data = pd.DataFrame(country_data)
data.index = np.arange(1970, 2025)
data.index.name = 'Year'

lstm_results = {}
plot_data = {}
look_back = 3  # Use the last 3 years to predict the next

print("--- Training LSTMs (This will take a minute or two) ---")
print("--- 2025 Forecasts ---")

# 2. Main Loop for each country
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# 2. Main Loop for each country
for country in data.columns:
    df_series = pd.DataFrame(data[country].rename('CO2'))

    # 1. Prepare the Base Training Data (1970-2019)
    base_train_raw = df_series.loc[df_series.index < 2020].values

    # Fit the scaler ONCE on the historical baseline and freeze it
    scaler = MinMaxScaler(feature_range=(0, 1))
    base_train_scaled = scaler.fit_transform(base_train_raw)

    # Use Keras built-in generator to instantly create 3D sequences (No manual loops)
    generator = TimeseriesGenerator(base_train_scaled, base_train_scaled, length=look_back, batch_size=1)

    # Build and compile the model ONCE per country
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train heavily on the historical baseline
    model.fit(generator, epochs=50, verbose=0)

    # --- WALK-FORWARD VALIDATION (2020-2024) ---
    years_to_predict = [2020, 2021, 2022, 2023, 2024]
    wf_predictions = []
    wf_actuals = []

    for target_year in years_to_predict:
        # Grab the 3 years immediately preceding the target year
        history_raw = df_series.loc[target_year-look_back : target_year-1].values

        # Transform using the frozen baseline scaler
        history_scaled = scaler.transform(history_raw)
        history_reshaped = np.reshape(history_scaled, (1, look_back, 1))

        # Predict the target year
        pred_scaled = model.predict(history_reshaped, verbose=0)
        pred_absolute = scaler.inverse_transform(pred_scaled)[0][0]
        actual_absolute = df_series.loc[target_year, 'CO2'].item()

        wf_predictions.append(pred_absolute)
        wf_actuals.append(actual_absolute)

        # ONLINE LEARNING: Update the model with the newly revealed actual data
        # We package the history and the new actual target into a single step
        actual_scaled = scaler.transform([[actual_absolute]])
        model.fit(history_reshaped, actual_scaled, epochs=5, verbose=0)

    # Calculate metrics
    mae = mean_absolute_error(wf_actuals, wf_predictions)
    mape = mean_absolute_percentage_error(wf_actuals, wf_predictions) * 100
    rmse = np.sqrt(mean_squared_error(wf_actuals, wf_predictions))
    lstm_results[country] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    # --- FORECAST 2025 ---
    # The model has already been updated incrementally through 2024.
    # We just need to feed it the final 3 known years (2022, 2023, 2024).
    final_lags_raw = df_series.loc[2022:2024].values
    final_lags_scaled = scaler.transform(final_lags_raw)
    final_lags_reshaped = np.reshape(final_lags_scaled, (1, look_back, 1))

    forecast_2025_scaled = model.predict(final_lags_reshaped, verbose=0)
    forecast_2025 = scaler.inverse_transform(forecast_2025_scaled)[0][0]

    print(f"{country}: {forecast_2025:.2f}")

    # Save data for the charts
    plot_data[country] = {
        'recent_years': np.arange(2015, 2025),
        'recent_actuals': df_series.loc[2015:2024, 'CO2'].values,
        'wf_years': years_to_predict,
        'wf_preds': wf_predictions,
        'forecast_2025': forecast_2025
    }

# 3. Output the performance metrics
print("\n--- LSTM Walk-Forward Validation Errors (2020-2024) ---")
for country, metrics in lstm_results.items():
    print(f"{country} - RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f} | MAPE: {metrics['MAPE']:.2f}%")

# 4. Visualization
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axes = axes.flatten()

for idx, country in enumerate(data.columns):
    ax = axes[idx]
    p_data = plot_data[country]

    ax.plot(p_data['recent_years'], p_data['recent_actuals'], marker='o', color='blue', label='Actual CO2')
    ax.plot(p_data['wf_years'], p_data['wf_preds'], marker='x', linestyle='--', color='red', label='Predicted (LSTM)')
    ax.plot([2025], [p_data['forecast_2025']], marker='*', markersize=10, color='green', label='2025 Forecast')

    ax.set_title(f"{country} CO2 Emissions Forecast (LSTM)")
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2 Levels")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()

fig.delaxes(axes[5])
plt.tight_layout()
plt.show()