"""
Simplified script to train and save the CO2 emission prediction model (faster version)
"""
import pandas as pd
import numpy as np
import numpy.random as nr
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import feature_selection as fs
import joblib
import os
import json

# Set random seed for reproducibility
random_state_num = 0
nr.seed(1)

# Load the cleaned dataset
data_path = 'Dataset/data_cleaned.csv'
data = pd.read_csv(data_path)

print("Dataset loaded. Shape:", data.shape)

# Remove the ARE outliers (as done in the notebook)
data = data[data['country'] != 'ARE']
print("Shape after removing ARE outliers:", data.shape)

# Choose features and label columns
feature_cols = ['cereal_yield', 'fdi_perc_gdp', 'gni_per_cap', 'en_per_cap', 
                'pop_urb_aggl_perc', 'prot_area_perc', 'gdp', 'pop_growth_perc', 
                'urb_pop_growth_perc']
label_col = ['co2_per_cap']

# Convert into numpy arrays
features = np.array(data[feature_cols])
labels = np.array(data[label_col])

# Split into training and testing subsets
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=random_state_num
)

print(f"Training set size: {features_train.shape[0]}")
print(f"Test set size: {features_test.shape[0]}")

# Feature selection using RFECV
print("\nPerforming feature selection...")
feature_folds = KFold(n_splits=4, shuffle=True, random_state=random_state_num)
rf_selector = RandomForestRegressor(n_estimators=100, random_state=random_state_num)
selector = fs.RFECV(estimator=rf_selector, cv=feature_folds, scoring='r2', n_jobs=-1)
selector = selector.fit(features_train, np.ravel(labels_train))

# Get selected features
ranks_transform = list(np.transpose(selector.ranking_))
chosen_features = [i for i, j in zip(feature_cols, ranks_transform) if j == 1]
print(f"Selected features: {chosen_features}")

# Transform features
features_train_reduced = selector.transform(features_train)
features_test_reduced = selector.transform(features_test)

# Train model with good default parameters (faster than hyperparameter tuning)
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=random_state_num,
    n_jobs=-1
)

rf_model.fit(features_train_reduced, np.ravel(labels_train))

# Evaluate on test set
from sklearn.metrics import r2_score, mean_squared_error
predictions = rf_model.predict(features_test_reduced)
r2 = r2_score(y_true=labels_test, y_pred=predictions)
mse = mean_squared_error(y_true=labels_test, y_pred=predictions)
rmse = np.sqrt(mse)

print(f"\nTest set performance:")
print(f"  R2 Score: {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")

# Save the model and feature selector
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/co2_model.pkl')
joblib.dump(selector, 'models/feature_selector.pkl')

# Calculate feature ranges for UI (before filtering)
feature_ranges = {}
for feature in feature_cols:
    if feature in data.columns:
        feature_ranges[feature] = {
            'min': float(data[feature].min()),
            'max': float(data[feature].max()),
            'mean': float(data[feature].mean())
        }

# Save feature information including ranges
model_info = {
    'selected_features': chosen_features,
    'all_features': feature_cols,
    'r2_score': float(r2),
    'rmse': float(rmse),
    'feature_ranges': feature_ranges
}

with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\nModel saved to 'models/co2_model.pkl'")
print(f"Feature selector saved to 'models/feature_selector.pkl'")
print(f"Model info saved to 'models/model_info.json'")

