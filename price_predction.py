import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("housing_price_dataset.csv")
df.columns = df.columns.str.lower()

# Now let's change how we measure the years in the company by changing the joining year by the years in the company.


df["house_age"] = 2024 - df["yearbuilt"]

del df["yearbuilt"]

# The Bedrooms ans bathrooms are categorical and non contineuous values


df["bathrooms"].astype(object)
df["bedrooms"].astype(object)

categorical_columns = df.dtypes[df.dtypes==object].index
for column in categorical_columns:
    df[column] = df[column].str.lower().str.replace(" ","_")

  

 
# #### Data Preprocessing
# 
# In this step, we'll handle any missing values and convert categorical variables into a numerical format. We'll use DictVectorizer to create one-hot encoding for the 'Neighborhood' variable.


df.isnull().sum()
df = pd.get_dummies(df, columns=['neighborhood'], drop_first=False).astype(int)
df = pd.get_dummies(df, columns=['bathrooms'], drop_first=False).astype(int)
df = pd.get_dummies(df, columns=['bedrooms'], drop_first=False).astype(int)

numerical_features = ['squarefeet', 'house_age']
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=feature, y='price', data=df)
    plt.title(f'{feature} vs Price')
    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.show()


from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df,test_size=0.2,random_state=1)
df_train, df_val = train_test_split(df_full_train,test_size=0.25,random_state=1)
print(len(df_train),len(df_test),len(df_val))

 
# ### Splitting into Training, Validation and Testing Sets
# 
# Now, we'll split the data into training, validation and testing sets to train our models on one subset and evaluate them on another.


df_full_train =df_full_train.reset_index(drop=True)
df_train =df_train.reset_index(drop=True)
df_val =df_val.reset_index(drop=True)
df_test =df_test.reset_index(drop=True)

 
# ###  Splitting the Data
# 
# Now, we'll split the data into features (X) and the target variable (y). The target variable is 'Price', and the rest of the columns are considered features
# 


y_train = df_train["price"]
y_val = df_val["price"]
y_test = df_test["price"]
y_full_train = df_full_train["price"]

del df_train["price"]
del df_val["price"]
del df_test["price"]
del df_full_train["price"]


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

linear_model = LinearRegression()

linear_model.fit(df_train, y_train)

# Predictions on the training set
training_predictions = linear_model.predict(df_train)

# Evaluate the model on the training set
print("Training set performance:")
print("Mean Squared Error:", mean_squared_error(y_train, training_predictions))
print("R-squared Score:", r2_score(y_train, training_predictions))

# Predictions on the validation set
validation_predictions = linear_model.predict(df_val)

# Evaluate the model on the validation set
print("Validation set performance:")
print("Mean Squared Error:", mean_squared_error(y_val, validation_predictions))
print("R-squared Score:", r2_score(y_val, validation_predictions))

 
# ### Feature Scaling
# 
# We'll use standardization to scale the numerical features. This involves transforming the data such that it has a mean of 0 and a standard deviation of 1.


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train)
X_val_scaled = scaler.transform(df_val)



# Now, let's retrain the Linear Regression model on the scaled data
linear_model_scaled = LinearRegression()
linear_model_scaled.fit(X_train_scaled, y_train)

# Predictions on the training set
train_predictions_scaled = linear_model_scaled.predict(X_train_scaled)

# Predictions on the testing set
val_predictions_scaled = linear_model_scaled.predict(X_val_scaled)

# Evaluate the model on both sets
print("Scaled Training set performance:")
print("Mean Squared Error:", mean_squared_error(y_train, train_predictions_scaled))
print("R-squared Score:", r2_score(y_train, train_predictions_scaled))

print("\nScaled Validation set performance:")
print("Mean Squared Error:", mean_squared_error(y_val, val_predictions_scaled))
print("R-squared Score:", r2_score(y_val, val_predictions_scaled))

 
# It seems that feature scaling did not significantly improve the model's performance. The results are similar to the unscaled model.
# 
# Next, we can explore feature engineering to see if creating new features or transforming existing ones can enhance the model's predictive power. We'll start by adding polynomial features.


from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly_features.fit_transform(df_train)

X_validation_poly = poly_features.transform(df_val)

# Retrain the Linear Regression model on the polynomial features
linear_model_poly = LinearRegression()
linear_model_poly.fit(X_train_poly, y_train)

train_predictions_poly = linear_model_poly.predict(X_train_poly)

validation_predictions_poly = linear_model_poly.predict(X_validation_poly)

# Evaluate the model on both sets
print("Polynomial Training set performance:")
print("Mean Squared Error:", mean_squared_error(y_train, train_predictions_poly))
print("R-squared Score:", r2_score(y_train, train_predictions_poly))

print("\nPolynomial validation set performance:")
print("Mean Squared Error:", mean_squared_error(y_val, validation_predictions_poly))
print("R-squared Score:", r2_score(y_val, validation_predictions_poly))


 
# The addition of polynomial features does not seem to have a substantial impact on the model's performance. The results are still comparable to the previous models.
# 
# Now, let's explore another option - trying a more complex model. We'll use the Random Forest Regressor, which is an ensemble method that can capture non-linear relationships in the data.


from sklearn.ensemble import RandomForestRegressor


# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=6)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Predictions on the training set
train_predictions_rf = rf_model.predict(X_train_scaled)

# Predictions on the validation set
validation_predictions_rf = rf_model.predict(X_val_scaled)

# Evaluate the model on both sets
print("Random Forest Training set performance:")
print("Mean Squared Error:", mean_squared_error(y_train, train_predictions_rf))
print("R-squared Score:", r2_score(y_train, train_predictions_rf))

print("\nRandom Forest Validation set performance:")
print("Mean Squared Error:", mean_squared_error(y_val, validation_predictions_rf))
print("R-squared Score:", r2_score(y_val, validation_predictions_rf))
       

 
# The Random Forest Regressor shows promising results on the training set, capturing a significant amount of variance (R-squared Score of 0.93). However, there is a notable drop in performance on the validation set, which suggests potential overfitting.
# 
# To address overfitting, we can explore tuning the hyperparameters of the Random Forest model or consider other ensemble models. Alternatively, we can try a different algorithm like Gradient Boosting, which is another powerful ensemble method.


from sklearn.ensemble import GradientBoostingRegressor

# Initialize the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(random_state=42)

# Train the model
gb_model.fit(X_train_scaled, y_train)

# Predictions on the training set
train_predictions_gb = gb_model.predict(X_train_scaled)

# Predictions on the validation set
val_predictions_gb = gb_model.predict(X_val_scaled)

# Evaluate the model on both sets
print("Gradient Boosting Training set performance:")
print("Mean Squared Error:", mean_squared_error(y_train, train_predictions_gb))
print("R-squared Score:", r2_score(y_train, train_predictions_gb))

print("\nGradient Boosting Validation set performance:")
print("Mean Squared Error:", mean_squared_error(y_val, val_predictions_gb))
print("R-squared Score:", r2_score(y_val, val_predictions_gb))


 
# ### Hyperparameter Tuning - Grid Search for Gradient Boosting


from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 3, 4]
}

# Initialize the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(random_state=42)

# Initialize Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Extract feature importances from the best estimator
feature_importances = grid_search.best_estimator_.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': df_train.columns, 'Importance': feature_importances})

# Sort features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.show()



# Display the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)



# Initialize the Gradient Boosting Regressor
gb_model_tuned = GradientBoostingRegressor(random_state=42,learning_rate=0.1,
                                           max_depth=3,min_samples_split=4,
                                           n_estimators=50)

# Train the model
gb_model.fit(X_train_scaled, y_train)

# Predictions on the training set using the best model
train_predictions_gb_tuned = grid_search.best_estimator_.predict(X_train_scaled)

# Predictions on the testing set using the best model
val_predictions_gb_tuned = grid_search.best_estimator_.predict(X_val_scaled)

# Evaluate the model on both sets
print("\nTuned Gradient Boosting Training set performance:")
print("Mean Squared Error:", mean_squared_error(y_train, train_predictions_gb_tuned))
print("R-squared Score:", r2_score(y_train, train_predictions_gb_tuned))

print("\nTuned Gradient Boosting Validation set performance:")
print("Mean Squared Error:", mean_squared_error(y_val, val_predictions_gb_tuned))
print("R-squared Score:", r2_score(y_val, val_predictions_gb_tuned))

 
# It seems that the best model is the last model so let's train it to the full training data set


# Initialize the Gradient Boosting Regressor
gb_model_tuned_full = GradientBoostingRegressor(random_state=42,learning_rate=0.1,
                                           max_depth=3,min_samples_split=4,
                                           n_estimators=50)

scaler = StandardScaler()
X_full_train_scaled = scaler.fit_transform(df_full_train)
X_test_scaled = scaler.transform(df_test)
# Train the model
gb_model.fit(X_full_train_scaled, y_full_train)

# Predictions on the training set using the best model
full_train_predictions_gb_tuned = grid_search.best_estimator_.predict(X_full_train_scaled)

# Predictions on the testing set using the best model
test_gb_tuned = grid_search.best_estimator_.predict(X_test_scaled)

# Evaluate the model on both sets
print("\nTuned Gradient Boosting Training set performance:")
print("Mean Squared Error:", mean_squared_error(y_full_train, full_train_predictions_gb_tuned))
print("R-squared Score:", r2_score(y_full_train, full_train_predictions_gb_tuned))

print("\nTuned Gradient Boosting Test set performance:")
print("Mean Squared Error:", mean_squared_error(y_test, test_gb_tuned))
print("R-squared Score:", r2_score(y_test, test_gb_tuned))



pickle.dump(gb_model_tuned_full,open("house_price_prediction.pkl","wb"))








