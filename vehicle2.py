import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('C:/Users/soham/Downloads/Projects-20240722T093004Z-001/Projects/vehicle_price_prediction/Vehicle Price Prediction/dataset.csv')

# Data Cleaning
data.dropna(inplace=True)  # Remove missing values

# Inspect the 'engine' column
print(data['engine'].unique())  # Check unique values in the engine column

# Feature Engineering for the 'engine' column
# Example: Extracting number of cylinders from the engine description
data['cylinders'] = data['engine'].str.extract('(\d+)').astype(float)  # Extract digits and convert to float

# Drop the original 'engine' column as it's no longer needed
data.drop('engine', axis=1, inplace=True)

# Convert categorical variables into numerical format
data = pd.get_dummies(data, columns=['make', 'model', 'fuel', 'transmission', 'body', 'drivetrain', 'trim', 'exterior_color', 'interior_color'], drop_first=True)

# Feature Engineering: Create 'age' feature
data['age'] = 2024 - data['year']  # Assuming current year is 2024

# Define features and target variable
features = data.drop(['name', 'description', 'price', 'year'], axis=1)
target = data['price']

# Check the shape of features and target
print(f'Features shape: {features.shape}')
print(f'Target shape: {target.shape}')

# Check for NaN values in features
print(f'NaN values in features: {features.isnull().sum().sum()}')

# Drop any remaining NaN values after conversion
features.dropna(inplace=True)
target = target[features.index]  # Align target with features

# Check the shape after dropping NaN values
print(f'Features shape after dropping NaN: {features.shape}')
print(f'Target shape after aligning: {target.shape}')

# Ensure all features are numeric
print(features.dtypes)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}, MSE: {mse}, R^2: {r2}')