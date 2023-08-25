import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

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

# 2.2 Choosing correct model for a classification problem

heart_disease = pd.read_csv("data/heart-disease.csv")
np.random.seed(42)

X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
clf = LinearSVC(max_iter=10000)

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = RandomForestClassifier()

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# Notes:
# 1. If you have structured data, use ensemble methods
# 2. If you have unstructured data, use deep learning or transfer learning
