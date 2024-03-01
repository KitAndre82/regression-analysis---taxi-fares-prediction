#!/usr/bin/env python
# coding: utf-8

# Imports
# Packages for numerics + dataframes
import pandas as pd
import numpy as np

# Packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Packages for date conversions for calculating trip durations
from datetime import datetime
from datetime import date
from datetime import timedelta

# Packages for OLS, MLR, confusion matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics # For confusion matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

df0 = pd.read_csv("/Users/andrew/Downloads/2017_Yellow_Taxi_Trip_Data.csv")

df0.head()

df0.info()

# Keep `df0` as the original dataframe and create a copy (df) where changes will go
# Can revert `df` to `df0` if needed down the line
df = df0.copy()

# Display the dataset's shape
print(df.shape)

# Display basic info about the dataset
df.info()

# Check for duplicates
print('Shape of dataframe:', df.shape)
print('Shape of dataframe with duplicates dropped:', df.drop_duplicates().shape)

# Check for missing values in dataframe
print('Total count of missing values:', df.isna().sum().sum())

# Display missing values per column in dataframe
print('Missing values per column:')
df.isna().sum()

# Display descriptive stats about the data
df.describe()

# Some things stand out from this table of summary statistics. For instance, there are clearly some outliers in several variables,
# like tip_amount -$200 & total_amount -$1,200. Also, a number of the variables, such as mta_tax, seem to be almost constant
# throughout the data, which would imply that they would not be expected to be very predictive.

# Check the format of the data
df['tpep_dropoff_datetime'][0]

# Convert datetime columns to datetime
# Display data types of `tpep_pickup_datetime`, `tpep_dropoff_datetime`
print('Data type of tpep_pickup_datetime:', df['tpep_pickup_datetime'].dtype)
print('Data type of tpep_dropoff_datetime:', df['tpep_dropoff_datetime'].dtype)

# Convert `tpep_pickup_datetime` to datetime format
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')

# Convert `tpep_dropoff_datetime` to datetime format
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')

# Display data types of `tpep_pickup_datetime`, `tpep_dropoff_datetime`
print('Data type of tpep_pickup_datetime:', df['tpep_pickup_datetime'].dtype)
print('Data type of tpep_dropoff_datetime:', df['tpep_dropoff_datetime'].dtype)

df.head(3)

# Create `duration` column
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'])/np.timedelta64(1,'m')

# Inspect the columns and decide which ones to check for outliers.
df.info()

# The most important columns to check for outliers are likely to be:
# - trip_distance
# - fare_amount
# - duration

# Plot a box plot for each feature: trip_distance, fare_amount, duration.

fig, axes = plt.subplots(1, 3, figsize=(15, 2))
fig.suptitle('Boxplots for outlier detection')
sns.boxplot(ax=axes[0], x=df['trip_distance'])
sns.boxplot(ax=axes[1], x=df['fare_amount'])
sns.boxplot(ax=axes[2], x=df['duration'])
plt.show();


# All three variables contain outliers. Some are extreme, but others not so much.
# It's 30 miles from the southern tip of Staten Island to the northern end of Manhattan and that's in a straight line.
# With this knowledge and the distribution of the values in this column, it's reasonable to leave these values alone and not alter them.
#However, the values for fare_amount and duration definitely seem to have problematic outliers on the higher end.

# Are trip distances of 0 bad data or very short trips rounded down?
sorted(set(df['trip_distance']))[:10]

# Number of rides with a trip distance of zero
sum(df['trip_distance']==0)

# 148 out of  over 22,000 rides is relatively insignificant. These values are unlikely to have much of an effect on the model.
#Therefore, the trip_distance column will remain untouched with regard to outliers.

# #### fare_amount outliers

df['fare_amount'].describe()


# The range of values in the fare_amount column is large and the extremes don't make much sense.
# Low values -  Negative values are problematic. Values of zero could be legitimate if the taxi logged a trip that was immediately canceled.
# High values - The maximum fare amount in this dataset is close to $1,000, which seems very unlikely.
# High values for this feature can be capped based on intuition and statistics. The interquartile range (IQR) is $8.
# The standard formula of Q3 + (1.5 * IQR) yields $26.50. That doesn't seem appropriate for the maximum fare cap.
# In this case, we'll use a factor of 6, which results in a cap of $62.50.
# Impute values less than $0 with 0.

# Impute values less than $0 with 0
df.loc[df['fare_amount'] < 0, 'fare_amount'] = 0
df['fare_amount'].min()

# Now impute the maximum value as Q3 + (6 * IQR).

def outlier_imputer(column_list, iqr_factor):
    '''
    Impute upper-limit values in specified columns based on their interquartile range.

    Arguments:
        column_list: A list of columns to iterate over
        iqr_factor: A number representing x in the formula:
                    Q3 + (x * IQR). Used to determine maximum threshold,
                    beyond which a point is considered an outlier.

    The IQR is computed for each column in column_list and values exceeding
    the upper threshold for each column are imputed with the upper threshold value.
    '''
    for col in column_list:
        # Reassign minimum to zero
        df.loc[df[col] < 0, col] = 0

        # Calculate upper threshold
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_threshold = q3 + (iqr_factor * iqr)
        print(col)
        print('q3:', q3)
        print('upper_threshold:', upper_threshold)

        # Reassign values > threshold to threshold
        df.loc[df[col] > upper_threshold, col] = upper_threshold
        print(df[col].describe())
        print()

outlier_imputer(['fare_amount'], 6)

# #### duration outliers

df['duration'].describe()

# The duration column has problematic values at both the lower and upper extremities.
# 
# Low values: There should be no values that represent negative time. Impute all negative durations with 0.
# 
# High values: Impute high values the same way you imputed the high-end outliers for fares: Q3 + (6 * IQR).

# Impute a 0 for any negative values
df.loc[df['duration'] < 0, 'duration'] = 0
df['duration'].min()

# Impute the high outliers
outlier_imputer(['duration'], 6)

# #### Feature engineering

# Create mean_distance column
# 
# When deployed, the model will not know the duration of a trip until after the trip occurs,
# so we cannot train a model that uses this feature.
# However, we can use the statistics of trips we do know to generalize about ones we do not know.
# 
# In this step, I create a column called mean_distance that captures the mean distance for each group of trips that share pickup and dropoff points.
# 

# Create `pickup_dropoff` column
df['pickup_dropoff'] = df['PULocationID'].astype(str) + ' ' + df['DOLocationID'].astype(str)
df['pickup_dropoff'].head(2)


# Now, use a groupby() statement to group each row by the new pickup_dropoff column, compute the mean,
# and capture the values only in the trip_distance column. Assign the results to a variable named grouped.

grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[['trip_distance']]
grouped[:5]

# 1. Convert `grouped` to a dictionary
grouped_dict = grouped.to_dict()

# 2. Reassign to only contain the inner dictionary
grouped_dict = grouped_dict['trip_distance']

# 1. Create a mean_distance column that is a copy of the pickup_dropoff helper column
df['mean_distance'] = df['pickup_dropoff']

# 2. Map `grouped_dict` to the `mean_distance` column
df['mean_distance'] = df['mean_distance'].map(grouped_dict)

# Confirm that it worked
df[(df['PULocationID']==100) & (df['DOLocationID']==231)][['mean_distance']]

# Create mean_duration column
grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[['duration']]
grouped

# Create a dictionary where keys are unique pickup_dropoffs and values are
# mean trip duration for all trips with those pickup_dropoff combos
grouped_dict = grouped.to_dict()
grouped_dict = grouped_dict['duration']

df['mean_duration'] = df['pickup_dropoff']
df['mean_duration'] = df['mean_duration'].map(grouped_dict)

# Confirm that it worked
df[(df['PULocationID']==100) & (df['DOLocationID']==231)][['mean_duration']]


# ##### Create day and month columns
# 
# Create two new columns, day (name of day) and month (name of month) by extracting the relevant information from the tpep_pickup_datetime column.

# Create 'day' col
df['day'] = df['tpep_pickup_datetime'].dt.day_name().str.lower()

# Create 'month' col
df['month'] = df['tpep_pickup_datetime'].dt.strftime('%b').str.lower()

# Create 'rush_hour' col
df['rush_hour'] = df['tpep_pickup_datetime'].dt.hour

# If day is Saturday or Sunday, impute 0 in `rush_hour` column
df.loc[df['day'].isin(['saturday', 'sunday']), 'rush_hour'] = 0

def rush_hourizer(hour):
    if 6 <= hour['rush_hour'] < 10:
        val = 1
    elif 16 <= hour['rush_hour'] < 20:
        val = 1
    else:
        val = 0
    return val

# Apply the `rush_hourizer()` function to the new column
df.loc[(df.day != 'saturday') & (df.day != 'sunday'), 'rush_hour'] = df.apply(rush_hourizer, axis=1)
df.head()

# Create a scatter plot of duration and trip_distance, with a line of best fit
sns.set(style='whitegrid')
f = plt.figure()
f.set_figwidth(5)
f.set_figheight(5)
sns.regplot(x=df['mean_duration'], y=df['fare_amount'],
            scatter_kws={'alpha':0.5, 's':5},
            line_kws={'color':'red'})
plt.ylim(0, 70)
plt.xlim(0, 70)
plt.title('Mean duration x fare amount')
plt.show()

# The mean_duration variable correlates with the target variable. But what are the horizontal lines around fare amounts
# of 52 dollars and 63 dollars? What are the values and how many are there?
# You know what one of the lines represents. 62 dollars and 50 cents is the maximum that was imputed for outliers,
# so all former outliers will now have fare amounts of $62.50. What is the other line?
# Check the value of the rides in the second horizontal line in the scatter plot.

df[df['fare_amount'] > 50]['fare_amount'].value_counts().head()


# There are 514 trips whose fares were $52.

# Examine the first 30 of these trips.

# Set pandas to display all columns
pd.set_option('display.max_columns', None)
df[df['fare_amount']==52].head(30)


# It seems that almost all of the trips in the first 30 rows where the fare amount was $52 either begin or end at
# location 132, and all of them have a RatecodeID of 2.
# There is no readily apparent reason why PULocation 132 should have so many fares of 52 dollars.
# They seem to occur on all different days, at different times, with both vendors, in all months.
# However, there are many toll amounts of $5.76 and $5.54. This would seem to indicate that location 132
# is in an area that frequently requires tolls to get to and from. It's likely this is an airport.
# The data dictionary says that RatecodeID of 2 indicates trips for JFK, which is John F. Kennedy International Airport.
# A quick Google search for "new york city taxi flat rate $52" indicates that in 2017 (the year that this data was collected)
# there was indeed a flat fare for taxi trips between JFK airport (in Queens) and Manhattan.
# Because RatecodeID is known from the data dictionary, the values for this rate code can be imputed back into the
# data after the model makes its predictions. This way you know that those data points will always be correct.

# Drop features that are irrelevant
df2 = df.copy()

df2 = df2.drop(['Unnamed: 0', 'tpep_dropoff_datetime', 'tpep_pickup_datetime',
               'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID',
               'payment_type', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
               'total_amount', 'tpep_dropoff_datetime', 'tpep_pickup_datetime', 'duration',
               'pickup_dropoff', 'day', 'month'
               ], axis=1)

df2.info()

# Create a pairplot to visualize pairwise relationships between variables in the data
### YOUR CODE HERE ###

sns.pairplot(df2[['fare_amount', 'mean_duration', 'mean_distance']],
             plot_kws={'alpha':0.4, 'size':5},
             );

# Create correlation matrix containing pairwise correlation of columns, using pearson correlation coefficient
df2.corr(method='pearson')

# Create correlation heatmap

plt.figure(figsize=(6,4))
sns.heatmap(df2.corr(method='pearson'), annot=True, cmap='Reds')
plt.title('Correlation heatmap',
          fontsize=10)
plt.show()


# mean_duration and mean_distance are both highly correlated with the target variable of fare_amount.
# They're also both correlated with each other, with a Pearson correlation of 0.87.
# Recall that highly correlated predictor variables can be bad for linear regression models when you want
# to be able to draw statistical inferences about the data from the model.
# However, correlated predictor variables can still be used to create an accurate predictor
# if the prediction itself is more important than using the model as a tool to learn about your data.
# This model will predict fare_amount, which will be used as a predictor variable in machine learning models.
# Therefore, try modeling with both variables even though they are correlated.

# #### Constructing the model

# Remove the target column from the features
X = df2.drop(columns=['fare_amount'])

# Set y variable
y = df2[['fare_amount']]

# Display first few rows
X.head()

# Convert VendorID to string
X['VendorID'] = X['VendorID'].astype(str)

# Get dummies
X = pd.get_dummies(X, drop_first=True)
X.head()

# Create training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the X variables
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print('X_train scaled:', X_train_scaled)

# Fit model to the training data
lr=LinearRegression()
lr.fit(X_train_scaled, y_train)

# Evaluate the model performance on the training data
r_sq = lr.score(X_train_scaled, y_train)
print('Coefficient of determination:', r_sq)
y_pred_train = lr.predict(X_train_scaled)
print('R^2:', r2_score(y_train, y_pred_train))
print('MAE:', mean_absolute_error(y_train, y_pred_train))
print('MSE:', mean_squared_error(y_train, y_pred_train))
print('RMSE:',np.sqrt(mean_squared_error(y_train, y_pred_train)))

# Scale the X_test data
X_test_scaled = scaler.transform(X_test)

# Evaluate the model performance on the testing data
r_sq_test = lr.score(X_test_scaled, y_test)
print('Coefficient of determination:', r_sq_test)
y_pred_test = lr.predict(X_test_scaled)
print('R^2:', r2_score(y_test, y_pred_test))
print('MAE:', mean_absolute_error(y_test,y_pred_test))
print('MSE:', mean_squared_error(y_test, y_pred_test))
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred_test)))


# The model performance is high on both training and test sets, suggesting that there islittle bias in the
# model and that the model is not overfit. In fact, the test scores were even better than the training scores.
# For the test data, an R2 of 0.868 means that 86.8% of the variance in the fare_amount variable is described by the model.
# The mean absolute error is informative here because, for the purposes of the model, an error of two is not more than twice as bad as an error of one.

# Create a `results` dataframe
results = pd.DataFrame(data={'actual': y_test['fare_amount'],
                             'predicted': y_pred_test.ravel()})
results['residual'] = results['actual'] - results['predicted']
results.head()

# Create a scatterplot to visualize `predicted` over `actual`
fig, ax = plt.subplots(figsize=(6, 6))
sns.set(style='whitegrid')
sns.scatterplot(x='actual',
                y='predicted',
                data=results,
                s=20,
                alpha=0.5,
                ax=ax
)
# Draw an x=y line to show what the results would be if the model were perfect
plt.plot([0,60], [0,60], c='red', linewidth=2)
plt.title('Actual vs. predicted');

# Visualize the distribution of the `residuals`
sns.histplot(results['residual'], bins=np.arange(-15,15.5,0.5))
plt.title('Distribution of the residuals')
plt.xlabel('residual value')
plt.ylabel('count');

results['residual'].mean()


# The distribution of the residuals is approximately normal and has a mean of -0.015.
# The residuals represent the variance in the outcome variable that is not explained by the model.
# A normal distribution around zero is good, as it demonstrates that the model's errors are evenly distributed and unbiased.

# Create a scatterplot of `residuals` over `predicted`

sns.scatterplot(x='predicted', y='residual', data=results)
plt.axhline(0, c='red')
plt.title('Scatterplot of residuals over predicted values')
plt.xlabel('predicted value')
plt.ylabel('residual value')
plt.show()

# The model's residuals are evenly distributed above and below zero, with the exception of the sloping lines from the upper-left corner
# to the lower-right corner, which you know are the imputed maximum of $62.50 and the flat rate of $52 for JFK airport trips.

# Get model coefficients
coefficients = pd.DataFrame(lr.coef_, columns=X.columns)
coefficients

# The coefficients reveal that mean_distance was the feature with the greatest weight in the model's final prediction.
# Be careful here! A common misinterpretation is that for every mile traveled, the fare amount increases by a mean of $7.13.
# This is incorrect. Remember, the data used to train the model was standardized with StandardScaler().
# As such, the units are no longer miles. In other words, you cannot say "for every mile traveled...", as stated above.
# The correct interpretation of this coefficient is: controlling for other variables, for every +1 change in standard deviation,
# the fare amount increases by a mean of $7.13.
# Note also that because some highly correlated features were not removed, the confidence interval of this assessment is wider.
# So, translate this back to miles instead of standard deviation (i.e., unscale the data).
# Calculate the standard deviation of mean_distance in the X_train data.
# Divide the coefficient (7.133867) by the result to yield a more intuitive interpretation.

# 1. Calculate SD of `mean_distance` in X_train data
print(X_train['mean_distance'].std())

# 2. Divide the model coefficient by the standard deviation
print(7.133867 / X_train['mean_distance'].std())


# Now you can make a more intuitive interpretation: for every 3.57 miles traveled, the fare increased by a mean of $7.13.
# Or, reduced: for every 1 mile traveled, the fare increased by a mean of $2.00.

# #### Predict on full dataset

X_scaled = scaler.transform(X)
y_preds_full = lr.predict(X_scaled)

# Impute ratecode 2 fare
# The data dictionary says that the RatecodeID column captures the following information:
# 1 = standard rate
# 2 = JFK (airport)
# 3 = Newark (airport)
# 4 = Nassau or Westchester
# 5 = Negotiated fare
# 6 = Group ride
# This means that some fares don't need to be predicted. They can simply be imputed based on their rate code.
# Specifically, all rate codes of 2 can be imputed with $52, as this is a flat rate for JFK airport.
# The other rate codes have some variation (not shown here, but feel free to check for yourself).
# They are not a fixed rate, so these fares will remain untouched.
# Impute 52 at all predictions where RatecodeID is 2.

# Create a new df containing just the RatecodeID col from the whole dataset
final_preds = df[['RatecodeID']].copy()

# Add a column containing all the predictions
final_preds['y_preds_full'] = y_preds_full

# Impute a prediction of 52 at all rows where RatecodeID == 2
final_preds.loc[final_preds['RatecodeID']==2, 'y_preds_full'] = 52

# Check that it worked
final_preds[final_preds['RatecodeID']==2].head()

# Check performance on full dataset
final_preds = final_preds['y_preds_full']
print('R^2:', r2_score(y, final_preds))
print('MAE:', mean_absolute_error(y, final_preds))
print('MSE:', mean_squared_error(y, final_preds))
print('RMSE:',np.sqrt(mean_squared_error(y, final_preds)))

# Combine means columns with predictions column
nyc_preds_means = df[['mean_duration', 'mean_distance']].copy()
nyc_preds_means['predicted_fare'] = final_preds

nyc_preds_means.head()

# Save as a csv file
# 
# NOTES
# 
# Some things to note that differ from best practice or from how tasks are typically performed:
# When the mean_distance and mean_duration columns were computed, the means were calculated from the entire dataset.
# These same columns were then used to train a model that was used to predict on a test set.
# A test set is supposed to represent entirely new data that the model has not seen before,
# but in this case, some of its predictor variables were derived using data that was in the test set.
# 
# This is known as data leakage. Data leakage is when information from your training data contaminates the test data.
# If your model has unexpectedly high scores, there is a good chance that there was some data leakage. 
# 
# To avoid data leakage in this modeling process, it would be best to compute the means using only the training set
# and then copy those into the test set, thus preventing values from the test set from being included in the computation of the means.
# This would have created some problems because it's very likely that some combinations of pickup-dropoff locations would only appear in
# the test data (not the train data).
# This means that there would be NaNs in the test data, and further steps would be required to address this. 
# 
# In this case, the data leakage improved the R2 score by ~0.03. 
# 
# Imputing the fare amount for RatecodeID 2 after training the model and then calculating model performance metrics on the
# post-imputed data is not best practice. It would be better to separate the rides that did not have rate codes of 2,
# train the model on that data specifically, and then add the RatecodeID 2 data (and its imputed rates) after.
# This would prevent training the model on data that you don't need a model for, and would likely result in a better final model.
# However, the steps were combined for simplicity. 
# 
# Models that predict values to be used in another downstream model are common in data science workflows.
# When models are deployed, the data cleaning, imputations, splits, predictions, etc. are done using modeling pipelines.
# Pandas was used here to granularize and explain the concepts of certain steps, but this process would be streamlined by
# machine learning engineers. The ideas are the same, but the implementation would differ.
# Once a modeling workflow has been validated, the entire process can be automated, often with no need for pandas
# and no need to examine outputs at each step. This entire process would be reduced to a page of code.





