import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
data = pd.read_csv('dataset/urban_mobility_data_past_year.csv')
data = data.dropna()

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, drop=None) 
weather_encoded = encoder.fit_transform(data[['weather_conditions']])
weather_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['weather_conditions']))

# Include humidity and population_density in the feature set
temperature_df = data[['temperature']].reset_index(drop=True)
humidity_df = data[['humidity']].reset_index(drop=True)
population_density_df = data[['population_density']].reset_index(drop=True)

X = pd.concat([weather_df, temperature_df, humidity_df, population_density_df], axis=1)
y = data['congestion_level']

# Scale the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.ravel())  # Use ravel() to flatten y_train

# Evaluate the model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Rescale predictions
train_predictions_rescaled = scaler_y.inverse_transform(train_predictions.reshape(-1, 1))
test_predictions_rescaled = scaler_y.inverse_transform(test_predictions.reshape(-1, 1))
y_test_rescaled = scaler_y.inverse_transform(y_test)

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, color='blue', label='Actual')
plt.plot(test_predictions_rescaled, color='red', label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Index')
plt.ylabel('Congestion Level')
plt.legend()
plt.show()

# Calculate metrics
mae = mean_absolute_error(y_test_rescaled, test_predictions_rescaled)
mse = mean_squared_error(y_test_rescaled, test_predictions_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_rescaled, test_predictions_rescaled)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')

# Save the trained model
import joblib
joblib.dump(model, 'traffic_congestion_model_with_rf.pkl')
