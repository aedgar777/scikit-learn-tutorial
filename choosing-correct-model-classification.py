import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

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
