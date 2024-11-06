# immo-eliza-ml





This project builds a machine learning model to predict property prices based on various features from a real estate dataset. The project includes data preprocessing, feature encoding, scaling, and model training with evaluation metrics to assess the model's accuracy.

## Project Overview

The dataset used contains various features of properties, including location, size, and other property-specific details. The goal is to predict property prices by training a machine learning model and evaluating its performance.

## Prerequisites

- Python
- Libraries:
  - 'pandas'
  - 'numpy'
  - 'matplotlib'
  - 'seaborn'
  - 'scikit-learn'
  - 'xgboost'

## Data Preprocessing


Data Loading: Load the dataset from a CSV file.
Missing Values: Check for and handle missing values:
Drop columns latitude and longitude.
Fill missing numeric values with the mean.
Remove rows with non-numeric missing values labeled as 'MISSING'.
Feature Selection: Drop irrelevant columns, including id and price (target variable).
One-Hot Encoding: Encode categorical features to prepare them for model training.
Feature Scaling: Standardize numerical features using StandardScaler.



## Exploratory Data Analysis 

Correlation Analysis: Determine features most correlated with the target variable (price).
Top 10 Correlated Features: Identify and select the 10 features with the highest correlation to price.


## Model Training

Train-Test Split: Split the dataset into training and testing sets (80%-20%).




## Linear Regression:

Train a simple linear regression model with the feature most correlated with price.
Plot the regression line against actual values.



## Multiple Linear Regression:

Train a multiple linear regression model using the top 10 correlated features.
Evaluate using Mean Squared Error (MSE) and R-squared metrics.

## using xgboost as another models 
I used xgboost to check my result as a secound ML model.  

## Model Evaluation

Mean Squared Error (MSE): Evaluates the average squared difference between predicted and actual prices.
R-squared (R²): Indicates the proportion of variance in the target variable explained by the model.