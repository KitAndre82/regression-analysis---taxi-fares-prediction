# regression-analysis---taxi-fares-prediction

## Taxi Fare Prediction

### Overview

This project aims to build a machine learning model to predict taxi fare amounts for yellow taxis in New York City based on various features such as trip duration, distance, pickup and dropoff locations, etc. The model is built using a dataset of yellow taxi trip records from 2017.

### Project Structure

The repository contains the following files:

Python code file: Containing the entire data analysis, preprocessing, modeling, and evaluation process.

2017_Yellow_Taxi_Trip_Data.csv: Dataset used for analysis and modeling. This dataset contains information about yellow taxi trips in NYC, including trip duration, distance, fare amount, pickup/dropoff locations, and other relevant features.

README.md: README file containing project overview, installation instructions, and other relevant information.

### Project Workflow

The project workflow includes the following steps:

Data Loading: The dataset containing yellow taxi trip records is loaded into a Pandas DataFrame.

Data Cleaning and Preprocessing: Various data cleaning and preprocessing steps are performed, including handling missing values, removing duplicates, converting data types, and dealing with outliers.

Feature Engineering: New features such as mean trip distance and mean trip duration are created to capture additional information about trips based on pickup and dropoff locations. Day, month, and rush hour features are also engineered from the pickup datetime.

Exploratory Data Analysis (EDA): Exploratory data analysis is conducted to understand the distribution of variables, identify patterns, and explore relationships between features and the target variable (fare amount).

Modeling: A linear regression model is built using the processed data to predict taxi fare amounts. Features are scaled using StandardScaler, and the model is trained and evaluated using training and test datasets.

Model Evaluation: The trained model is evaluated based on various metrics such as R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). Residual analysis and visualization are also performed to assess model performance.

Prediction on Full Dataset: The trained model is used to predict taxi fare amounts for the entire dataset, including imputing fares for trips with RatecodeID 2 (JFK airport trips).

Results: Model predictions and evaluation results are summarized and presented.

### Conclusion

The project demonstrates the process of building a machine learning model for taxi fare prediction using NYC yellow taxi trip data. The trained model shows good performance in predicting fare amounts based on trip features. Further optimization and refinement of the model could be explored to improve performance and address potential issues such as data leakage and feature engineering.
