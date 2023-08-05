import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Getting data ready to be used with ML
# 1. Split the data into features (X) and labels (y)
# 2. Filling (imputing) or disregarding missing values
# 3. Converting non-numerical values to numerical values (feature coding)


heart_disease = pd.read_csv("data/heart-disease.csv")
X = heart_disease.drop("target", axis=1)
X.head()

y = heart_disease["target"]
y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

shapeArray = X_train.shape, X_test.shape, y_train.shape, y_test.shape  # provides amount of rows, columns in split sets

print(shapeArray)

# 1.1 Converting all data to numerical values

car_sales = pd.read_csv("data/car-sales-extended.csv")

print(car_sales.head())

# Split into X/y

X = car_sales.drop("Price", axis=1)
y = car_sales["Price"]

# Identify Categorical/Qualitative features and convert them to numbers that the machine learning model can digest
# and work with

categorical_features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

transformed_X = transformer.fit_transform(X)

print(transformed_X)

pd.DataFrame(transformed_X)

# Shorter way of reformatting data set to make categorical attributes numerical
dummies = pd.get_dummies(car_sales[["Make", "Colour", "Doors"]])
print(dummies)

# Refit model with clean, numerical data
# Split into training and test sets using clean, numerical data


np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=.20)

# Build machine learning model

model = RandomForestRegressor()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# 1.2 How to handle missing values
# A. Fill them with a value (imputation)
# B. Remove flawed sample altogether


# Import car sales csv with missing data
car_sales_missing = pd.read_csv("data/car-sales-extended-missing-data.csv")

print(car_sales_missing.isna().sum())  # tells us how many times an entry is missing each attribute

# Fill missing data with Pandas

# Fill the Make column with "missing" for entries in which it is empty
car_sales_missing["Make"].fillna("missing", inplace=True)

# Fill the Colour column the same way
car_sales_missing["Colour"].fillna("missing", inplace=True)

# Fill the Odometer (KM) column by filling empty cells with the mean of the entire column
car_sales_missing["Odometer (KM)"].fillna(car_sales_missing["Odometer (KM)"].mean(), inplace=True)

# Fill the Doors column by filling cells with what we estimate is the number of doors on a typical car

car_sales_missing["Doors"].fillna(4, inplace=True)

# Remove rows missing price since that is the attribute we are trying to predict and entries without it are useless

car_sales_missing.dropna(inplace=True)

print(car_sales_missing.isna().sum())

# Reassign X and y
X = car_sales_missing.drop("Price", axis=1)
y = car_sales_missing["Price"]

# Make data numerical again

categorical_features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

transformed_X = transformer.fit_transform(X)

pd.DataFrame(transformed_X)

X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=.20)
model = RandomForestRegressor()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# Filling missing values using scikit-learn

car_sales_missing = pd.read_csv("data/car-sales-extended-missing-data.csv")
car_sales_missing.dropna(subset=["Price"], inplace=True)

X = car_sales_missing.drop("Price", axis=1)
y = car_sales_missing["Price"]

# Fill categorical values with 'missing' & numerical values with mean using skikit library
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
door_imputer = SimpleImputer(strategy="constant", fill_value=4)
num_imputer = SimpleImputer(strategy="mean")

# Define columns
cat_features = ["Make", "Colour"]
door_feature = ["Doors"]
num_features = ["Odometer (KM)"]

# Create an imputer (something that fills missing data)

imputer = ColumnTransformer([
    ("cat_imputer", cat_imputer, cat_features),
    ("door_imputer", door_imputer, door_feature),
    ("num_imputer", num_imputer, num_features)
])

# Transform the data by using the column transformer we passed our imputers into

filled_X = imputer.fit_transform(X)
print(filled_X)

car_sales_filled = pd.DataFrame(filled_X, columns=["Make", "Colour", "Doors", "Odometer (KM)"])

categorical_features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

transformed_X = transformer.fit_transform(car_sales_filled)

print(transformed_X)

# Data is now numerical and has no missing values, can be used to fit a model

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score()
