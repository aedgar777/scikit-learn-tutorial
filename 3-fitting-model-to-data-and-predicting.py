import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

heart_disease = pd.read_csv("data/heart-disease.csv")

# Setup random seed
np.random.seed(42)

# Specify the data to be compared
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# Instantiate the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# 3.2 Making a predictions using a created model, 2 ways:
# A. predict() function
# B. predict.proba() function


clf.predict(X_test)  # uses model to predict target based on test features
np.array(y_test)  # puts target data in same format as the array returned from predict()

# Now we can compare predictions to the real values to see how good our model is at predicting
# This is what the .score() method does

y_preds = clf.predict(X_test)
print(np.mean(y_preds == y_test))  # These two methods do the same thing
print(accuracy_score(y_test, y_preds))


# Making predictions with predict.proba()
# predict.proba() returns probability that each specified data point is one of the outcomes of our possible target
# Example: If our possible outcomes are coded as 0 and 1, and predict.proba() returnes [.89, .11] for
# an item, that means the model thinks it has an 89% of being 0 and an 11% of being 1. predict() will return 0 for that
# same item because it is the more likely outcome according to the model
print(clf.predict_proba(X_test[:5]))

# Using predict() on the same data

print(clf.predict(X_test[:5]))



