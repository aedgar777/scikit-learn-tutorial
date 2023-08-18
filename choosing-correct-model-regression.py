import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Things to note:

# *There are different names in the field for different things, e.g. Scikit learns calls models estimators and
# algorithms

# Classification - predicts a category (recognizing an object) based on attributes
# uses classifiers (clf)

# Regression - predicting a number (guessing price of a car) based on attributes


# 2.1 Picking a machine learning model for a regression problem

# Get California Housing dataset

from sklearn.datasets import fetch_california_housing  ## sklearn has build in datasets for practicing

housing = fetch_california_housing()

# organizing data set into DataFrame and assigning names to columns programatically given their labels in the dataset

housing_df = pd.DataFrame(housing["data"], columns=housing["feature_names"])

housing_df["Target (MedHouseVal)"] = housing["target"]

print(housing_df)

np.random.seed(42)

X = housing_df.drop("Target (MedHouseVal)", axis=1)
y = housing_df["Target (MedHouseVal)"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = Ridge()
model.fit(X_train, y_train)

model.score(X_test, y_test)

# What if Ridge didn't work or the score didn't fit our needs? Try a different model

np.random.seed(42)

X = housing_df.drop("Target (MedHouseVal)", axis=1)
y = housing_df["Target (MedHouseVal)"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestRegressor()  # Uses a bunch of different trees to determine prediction
model.fit(X_train, y_train)

print(model.score(X_test, y_test))


