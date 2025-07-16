import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Step 1: Load your spending dataset (make sure file exists)
csv_path = "C:/Users/User/Desktop/Project/budget_predictor.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ CSV file not found at {csv_path}. Please check the path or filename.")

data = pd.read_csv(csv_path, parse_dates=['Date'])

# Step 2: Feature Engineering - aggregate monthly spending
data['Month'] = data['Date'].dt.to_period('M')
monthly_data = data.groupby('Month')['Amount'].sum().reset_index()
monthly_data['Month'] = monthly_data['Month'].astype(str)

# Step 3: Prepare data for prediction
monthly_data['MonthIndex'] = range(len(monthly_data))
X = monthly_data[['MonthIndex']]
y = monthly_data['Amount']

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)
print("🔍 Evaluation Metrics:")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Step 7: Predict future months
future_months = pd.DataFrame({'MonthIndex': range(len(monthly_data), len(monthly_data) + 6)})
predicted_spending = model.predict(future_months)
future_months['PredictedAmount'] = predicted_spending

# Step 8: Save Predictions to CSV
output_csv = "C:/Users/User/Desktop/Project/future_predictions.csv"
future_months.to_csv(output_csv, index=False)
print(f"✅ Future predictions saved to: {output_csv}")

# Step 9: Visualize the prediction
plt.figure(figsize=(10, 6))
plt.plot(monthly_data['MonthIndex'], y, label='Actual Spending', marker='o')
plt.plot(future_months['MonthIndex'], predicted_spending, label='Predicted Spending', linestyle='--', marker='x')
plt.xlabel('Month Index')
plt.ylabel('Total Spending')
plt.title('Personal Budget Predictor')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()