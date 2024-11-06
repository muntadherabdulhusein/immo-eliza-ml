
def one_hot_encode( df, drop_first=True, sparse_output=False):

# Using OneHotEncoder to transform categorical variables in features
    categorical_features = features.select_dtypes(include=['object']).columns

    encoder = OneHotEncoder(drop='first' if drop_first else None, sparse_output=False)

    encoded_categorical_data = encoder.fit_transform(features[categorical_features])
    
    encoded_categorical_df = pd.DataFrame(
    encoded_categorical_data, 
    columns=encoder.get_feature_names_out(categorical_features),
    index=features.index)

# Removing original categorical columns and adding encoded ones
    features_encoded = features.drop(columns=categorical_features).join(encoded_categorical_df)
    return features_encoded
# Example
one_hot_encode(features)




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





def train_evaluate_xgboost(X_train, y_train, X_test, y_test, random_state=42):
    
# Initialize and train the XGBoost Regressor
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
    xgboost_model.fit(X_train, y_train)
    
# Make predictions on the test set
    y_pred_xgboost = xgboost_model.predict(X_test)
    
# Calculate evaluation metrics
    mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)
    r2_xgboost = r2_score(y_test, y_pred_xgboost)
    
# Print and return the evaluation metrics
    print(f"XGBoost Mean Squared Error: {mse_xgboost}")
    print(f"XGBoost R-squared: {r2_xgboost}")
    
    return {"MSE": mse_xgboost, "R2": r2_xgboost}

# Example 
metrics = train_evaluate_xgboost(X_train, y_train, X_test, y_test)
print("Evaluation Metrics:", metrics)











