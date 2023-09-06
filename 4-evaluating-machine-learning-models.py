from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing  ## sk

# Three ways to evaluate SkiKit-learn models/estimators
# 1. Built-in score() method
# 2. The "scoring" parameter
# 3. Problem-specific metric functions

heart_disease = pd.read_csv("data/heart-disease.csv")
# Using score() method
np.random.seed(42)
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
model.score(X_test,y_test)

housing = fetch_california_housing()

# organizing data set into DataFrame and assigning names to columns programatically given their labels in the dataset

housing_df = pd.DataFrame(housing["data"], columns=housing["feature_names"])

np.random.seed(42)

housing_df["Target (MedHouseVal)"] = housing["target"]

X = housing_df.drop("Target (MedHouseVal)", axis=1)
y = housing_df["Target (MedHouseVal)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

model = RandomForestRegressor(n_estimators=1000)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))
