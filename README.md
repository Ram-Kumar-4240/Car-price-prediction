# Car Price Prediction Project

## Overview

This project involves building a machine learning model to predict car prices based on various features like mileage, engine size, transmission type, and more. The model is implemented using regression techniques, and the app allows users to interactively predict prices by inputting car details.

Additionally, a Streamlit app is developed to provide an easy-to-use interface for users to enter car features and get predictions instantly. The app also offers insights through data visualizations, like mileage distribution and correlation heatmaps.

## Features
#### 1. Data Cleaning & Preprocessing:
- Handling missing values and outliers.
- Encoding categorical data using One-Hot Encoding and Label Encoding.
- Scaling numerical features for model performance.
#### 2. Feature Engineering:
- Extracting useful features such as mileage, torque, car dimensions, and engine specifications.
- Applying statistical tests like:
  - Chi-Square Test: To evaluate relationships between categorical features and the target variable.
  - ANOVA Test: To compare means between numerical features and the target.
  - Correlation Matrix & Heatmap: To identify strong relationships between numerical features.
  - Low Variance Threshold: To remove features with very little variance, as they provide minimal information.
#### 3. Exploratory Data Analysis (EDA):
  - Univariate, bivariate, and multivariate analysis to uncover important patterns.
  - Plotting relationships between features and the car price.
#### 4. Model Training:
- Using multiple regression algorithms to predict car prices:
  - XGBoost Regressor
  - Random Forest Regressor
  - Linear Regression
  - Decision Tree Regressor
  - Support Vector Regression
- Selecting the best model using performance metrics such as:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)

## App Features
- Home Page: A brief introduction explaining the purpose of the app.
- Predict Page: Allows users to input car features such as fuel type, mileage, engine, and more to get an estimated car price.
- Insights Page: Offers visual insights like mileage distribution and a correlation heatmap to help users understand trends in the dataset.

## Installation

To set up and run this project on your local machine, follow these steps:
- Clone the repository:
```bash
  git clone https://github.com/Ram-Kumar-4240/Car_price_predictiom_Guvi.git
```
- Install the required libraries: Use pip to install all necessary dependencies.
```bash
  pip install -r requirements.txt
```
- Run the Streamlit app:
```bash
  streamlit run app.py
```
## Data Preprocessing
1. Missing Value Handling: Missing values were imputed using the mode for categorical variables and mean for numerical ones.
2. Encoding Categorical Data:
- One-Hot Encoding for non-ordinal categorical features like fuel type, body type, transmission, etc.
- Label Encoding for ordinal categorical features such as engine type, number of cylinders, seating capacity, etc.

## Feature Selection
- Chi-Square Test: Used to determine whether there is a significant association between categorical features.
  - For example, the relationship between fuel type and body type can be identified through this test.

- ANOVA (Analysis of Variance): This test was applied to identify relationships between the categorical features and the numerical target variable (car price). This helped in assessing which categorical features (e.g., transmission type) had a significant impact on car prices.

- Correlation Matrix & Heatmap: The relationships between numerical features like mileage, torque, length, and price were visualized using a correlation matrix to identify strong correlations and multicollinearity.

## Model Training
Several regression models were trained and tested:
- XGBoost Regressor: Known for its performance on tabular data, this model was one of the top-performing models for car price prediction.
- Random Forest Regressor: An ensemble method that combines predictions from multiple decision trees to produce more accurate results.
- Linear Regression: A basic approach that assumes a linear relationship between features and the target variable.
- Decision Tree Regressor: A model that splits the data into different branches based on feature values.
- Support Vector Regression: A model that finds the best fit line within a margin to predict the car price.

Hyperparameter tuning was performed for all models to optimize performance, and the models were evaluated using cross-validation to avoid overfitting.

## Evaluation Metrics
The models were compared using the following metrics:
- Mean Squared Error (MSE): Measures the average squared difference between the predicted and actual prices.
- Mean Absolute Error (MAE): Captures the average absolute difference between predicted and actual prices.
- R-squared (R²): Indicates the proportion of the variance in the target variable that is explained by the input features.

## Model Performance
- XGBoost Regressor
  - RMSE: 434,041.70
  - R-squared: 0.9369

## Streamlit App

The Streamlit app was developed to make the model accessible to users for predicting car prices interactively. The app has the following features:

- Home Page: Introduces the app and guides users on how to interact with the prediction feature.
#
![Screenshot 2024-10-05 231352](https://github.com/user-attachments/assets/d8b3a117-313b-409e-8a01-09995df98d1b)
#
- Predict Page:
  - Users can input details like fuel type, transmission, mileage, engine size, and more.
  - Based on the inputs, the model predicts the car price.
#
![Screenshot 2024-10-07 143421](https://github.com/user-attachments/assets/08c1b556-c102-4ad0-aba2-4d05f6338971)
#
- Insights Page: Provides users with data visualizations such as:
  - Mileage Distribution: A histogram to visualize how car mileage is distributed.
  - Correlation Heatmap: Displays the correlations between numerical features like mileage, torque, and car dimensions.
#
![Screenshot 2024-10-22 092526](https://github.com/user-attachments/assets/555ca452-6fbc-4950-ad6c-79bc3e51d965)
# 
## File Descriptions
- app.py: The main file containing the Streamlit app logic.
- models folder: Contains the trained machine learning models, scalers, and encoders.
- Dataset folder: Contains the training data used for predictions and insights.
- requirements.txt: Lists the Python libraries needed to run the project.
## Contributors
- Name : Ramkumar
- Email : infogramrk@gmail.com

