
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Step 1: Load and Prepare the Data
# Load real estate dataset
real_estate_data = pd.read_csv("real_estate_data.csv")

# Data Preprocessing
real_estate_data["date"] = pd.to_datetime(real_estate_data["date"])
real_estate_data = real_estate_data.sort_values("date")
real_estate_data["price_per_sqft"] = real_estate_data["price_usd"] / \
    real_estate_data["size_sqft"]

# Feature Engineering
real_estate_data["month"] = real_estate_data["date"].dt.month
real_estate_data["lag_price"] = real_estate_data["price_usd"].shift(1)
real_estate_data.dropna(inplace=True)

# Step 2: Train a Demand Forecasting Model
X = real_estate_data[["month", "size_sqft", "demand_factor", "lag_price"]]
y = real_estate_data["price_usd"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Prices")
plt.show()

# Step 3: Inventory Optimization
# Define optimization parameters
# Example holding cost
holding_cost = real_estate_data["price_usd"].mean() * 0.01
ordering_cost = 5000  # Fixed ordering cost per property
demand_forecast = model.predict(X)  # Forecasted prices

# Optimization problem: Minimize cost while meeting demand
costs = [holding_cost, ordering_cost]
constraints = [[1, 1]]  # Example constraint coefficients
bounds = [(0, None), (0, None)]  # No negative stock levels

result = linprog(c=costs, A_eq=constraints, b_eq=[
                 demand_forecast.sum()], bounds=bounds, method="highs")
print("Optimal Inventory Levels:", result.x)

# Step 4: Export Data
real_estate_data["forecasted_price"] = demand_forecast
real_estate_data.to_csv("forecasted_real_estate_data.csv", index=False)
print("Data exported for visualization!")
