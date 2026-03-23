import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

# 1. Load and prep data
df = pd.read_csv("co2-data.csv")
years = [str(year) for year in range(1970, 2025)]
country_data = {}

# Extract China Mainland specifically
china_mask = (df['Country'] == 'China') & (df['Sub-country'] == 'Mainland')
country_data['China'] = df.loc[china_mask, years].values.flatten()

# Extract the other countries
target_countries = ['Ukraine', 'Germany', 'United States', 'Japan']
for country in target_countries:
    mask = (df['Country'] == country)
    country_data[country] = df.loc[mask, years].sum(axis=0).values

# Build the final clean DataFrame
data = pd.DataFrame(country_data)
data.index = np.arange(1970, 2025)
data.index.name = 'Year'

# 2. Setup storage for metrics and plotting
tree_results = {}
plot_data = {}

print("--- 2025 Forecasts ---")

# 3. Main Loop for each country
for country in data.columns:

    df_series = pd.DataFrame(data[country].rename('CO2'))

    # THE FIX: Apply differencing to find the year-over-year change
    df_series['CO2_Diff'] = df_series['CO2'].diff()

    # Create lagged features based purely on the DIFFERENCES
    for i in range(1, 4):
        df_series[f'Lag_{i}'] = df_series['CO2_Diff'].shift(i)

    # Drop the rows with NaNs caused by diff() and shift()
    df_series.dropna(inplace=True)

    # --- WALK-FORWARD VALIDATION (2020-2024) ---
    years_to_predict = [2020, 2021, 2022, 2023, 2024]
    wf_predictions = []
    wf_actuals = []

    for target_year in years_to_predict:
        # Train on history up to the target year
        train_data = df_series.loc[df_series.index < target_year]
        test_data = df_series.loc[[target_year]]

        # Features are the lagged differences, target is the current difference
        X_train = train_data[['Lag_1', 'Lag_2', 'Lag_3']]
        y_train = train_data['CO2_Diff']
        X_test = test_data[['Lag_1', 'Lag_2', 'Lag_3']]

        model = DecisionTreeRegressor(max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # Predict the DIFFERENCE for the target year
        pred_diff = model.predict(X_test)[0]

        # Reconstruct the absolute value: Actual Previous Year + Predicted Difference
        actual_previous_year = data.loc[target_year - 1, country]
        pred_absolute = actual_previous_year + pred_diff

        # Get the actual absolute value for scoring
        actual_absolute = data.loc[target_year, country]

        wf_predictions.append(pred_absolute)
        wf_actuals.append(actual_absolute)

    # Calculate metrics on the reconstructed absolute values
    mae = mean_absolute_error(wf_actuals, wf_predictions)
    mape = mean_absolute_percentage_error(wf_actuals, wf_predictions) * 100
    rmse = np.sqrt(mean_squared_error(wf_actuals, wf_predictions))
    tree_results[country] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    # --- FORECAST 2025 ---
    X_full = df_series[['Lag_1', 'Lag_2', 'Lag_3']]
    y_full = df_series['CO2_Diff']

    final_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    final_model.fit(X_full, y_full)

    # Lags for 2025 are the actual differences from 2024, 2023, and 2022
    lags_for_2025 = pd.DataFrame({
        'Lag_1': [y_full.loc[2024]],
        'Lag_2': [y_full.loc[2023]],
        'Lag_3': [y_full.loc[2022]]
    })

    # Predict the 2025 difference and add it to the 2024 absolute actual
    pred_diff_2025 = final_model.predict(lags_for_2025)[0]
    actual_2024_absolute = data.loc[2024, country]
    forecast_2025 = actual_2024_absolute + pred_diff_2025

    print(f"{country}: {forecast_2025:.2f}")

    # Save data for the charts
    plot_data[country] = {
        'recent_years': np.arange(2015, 2025),
        'recent_actuals': data.loc[2015:2024, country].values,
        'wf_years': years_to_predict,
        'wf_preds': wf_predictions,
        'forecast_2025': forecast_2025
    }

# 4. Output the performance metrics
print("\n--- Walk-Forward Validation Errors (2020-2024) ---")
for country, metrics in tree_results.items():
    print(f"{country} - RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f} | MAPE: {metrics['MAPE']:.2f}%")

# 5. Visualization
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axes = axes.flatten()

for idx, country in enumerate(data.columns):
    ax = axes[idx]
    p_data = plot_data[country]

    ax.plot(p_data['recent_years'], p_data['recent_actuals'], marker='o', color='blue', label='Actual CO2')
    ax.plot(p_data['wf_years'], p_data['wf_preds'], marker='x', linestyle='--', color='red', label='Predicted (Tree)')
    ax.plot([2025], [p_data['forecast_2025']], marker='*', markersize=10, color='green', label='2025 Forecast')

    ax.set_title(f"{country} CO2 Emissions Forecast (Differenced Tree)")
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2 Levels")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()

fig.delaxes(axes[5])
plt.tight_layout()
plt.show()