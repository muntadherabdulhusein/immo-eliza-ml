import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt  # Correct import
import xgboost as xgb



m_l_df=pd.read_csv(r"properties.csv")

m_l_df = m_l_df.drop(columns=['latitude', 'longitude'])

m_l_df = m_l_df.fillna(m_l_df.mean(numeric_only=True))

non_numeric_columns = m_l_df.select_dtypes(exclude='number').columns
m_l_df = m_l_df[~m_l_df[non_numeric_columns].isin(['MISSING']).any(axis=1)]

# Defining the target (price) and features by dropping irrelevant columns
target = m_l_df['price']

# Dropping the 'id' and 'price' columns to keep only relevant features
features = m_l_df.drop(columns=['id', 'price'])




from sklearn.preprocessing import OneHotEncoder

# One-hot encoding function
def one_hot_encode(features, drop_first=True):
    # Identify categorical features
    categorical_features = features.select_dtypes(include=['object']).columns
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(drop='first' if drop_first else None, sparse_output=False)
    # Fit and transform categorical columns
    encoded_categorical_data = encoder.fit_transform(features[categorical_features])

    # Create DataFrame for the encoded features
    encoded_categorical_df = pd.DataFrame(
        encoded_categorical_data, 
        columns=encoder.get_feature_names_out(categorical_features),
        index=features.index
    )

    # Drop original categorical columns and add the encoded columns
    features_encoded = features.drop(columns=categorical_features).join(encoded_categorical_df)
    return features_encoded

# Apply encoding to the features
features_encoded = one_hot_encode(features)




def Scaling(features, target, test_size=0.2, random_state=42):

# Scaling features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)
    
# Splitting data
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

# Example 
X_train, X_test, y_train, y_test = Scaling(features_encoded, target)

# Displaying shapes to confirm preprocessing steps
print("Feature shapes:", X_train.shape, X_test.shape)
print("Target shapes:", y_train.shape, y_test.shape)



def plot_single_feature_regression(X_train, X_test, y_train, y_test, target_name="Price"):
    
# Calculating correlations and selecting the feature with the highest correlation
    correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    feature_name = correlations.index[0]
    
# Preparing the feature data for the model
    X_train_feature = X_train[feature_name].values.reshape(-1, 1)
    X_test_feature = X_test[feature_name].values.reshape(-1, 1)
    
# Fitting the model
    single_feature_model = LinearRegression()
    single_feature_model.fit(X_train_feature, y_train)
    
# Predicting on the test data
    y_pred_single = single_feature_model.predict(X_test_feature)
    
# Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[feature_name], y_test, color="blue", label="Actual Prices", alpha=0.5)
    plt.plot(X_test[feature_name], y_pred_single, color="red", label="Predicted Prices (Linear Fit)")
    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.title(f"Linear Regression with {feature_name} vs. {target_name}")
    plt.legend()
    plt.show()

# Example
plot_single_feature_regression(X_train, X_test, y_train, y_test, target_name="Price")





from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_evaluate_xgboost(X_train, y_train, X_test, y_test, random_state=42):
    # Initialize and train the XGBoost Regressor
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
    xgboost_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred_xgboost = xgboost_model.predict(X_test)
    
    # Calculate evaluation metrics
    mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)
    rmse_xgboost = np.sqrt(mse_xgboost)  # Root Mean Squared Error
    r2_xgboost = r2_score(y_test, y_pred_xgboost)
    
    # Print and return the evaluation metrics
    print(f"XGBoost Mean Squared Error: {mse_xgboost}")
    print(f"XGBoost Root Mean Squared Error: {rmse_xgboost}")
    print(f"XGBoost R-squared: {r2_xgboost}")
    
    return {"MSE": mse_xgboost, "RMSE": rmse_xgboost, "R2": r2_xgboost}

# Example evaluation
metrics = train_evaluate_xgboost(X_train, y_train, X_test, y_test)
print("Evaluation Metrics:", metrics)

import xgboost as xgb
import pickle

# Initialize and train the XGBoost model
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgboost_model.fit(X_train, y_train)


import joblib
import os

# Define the save path
save_path = r"C:\Users\munta\OneDrive\Desktop\The Projects\immo-eliza-deployment\streamlit"
os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

# Full file path for the model
model_file_path = os.path.join(save_path, "xgboost_model.pkl")

# Save the trained XGBoost model
joblib.dump(xgboost_model, model_file_path)

print(f"Model saved successfully at {model_file_path}")













