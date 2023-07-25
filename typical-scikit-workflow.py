# End to end Scikit-learn workflow

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Get data ready
heart_disease = pd.read_csv("data/heart-disease.csv")
print(heart_disease)

# Create X (features matrix)
X = heart_disease.drop("target", axis=1)  # axis 1 is the row "target" is in, i.e. the header 0 is the column

# Create Y (labels)
y = heart_disease["target"]

# 2. Choose the right model and hyperparameters
clf = RandomForestClassifier()

# Keep the default parameters
clf.get_params()

# 3. Fit the model to the data
# test_size is the portion of the data used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# tells our RandomForestClassifier to find patterns between our features and our target
clf.fit(X_train, y_train)

# Make a prediction

y_preds = clf.predict(X_test)
print(y_preds)
print(y_test)

# 4. Evaluate the model

training_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print(f"Evaluation when model can check itself against training data: {training_score}")
print(f"Evaluation when model can check itself against using only test data: {test_score}")

print(classification_report(y_test, y_preds))

print(confusion_matrix(y_test, y_preds))
print(accuracy_score(y_test, y_preds))

# 5. Improve/Tweak a model
# Try different amount of n_estimators

np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators...")
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(X_train, y_train)
    print(f"Model accuracy on test set: {clf.score(X_test, y_test) * 100}%")
    print(" ")

# 6. Save a model and load it

pickle.dump(clf, open("random_forest_model_1.pkl", "wb"))
loaded_model = pickle.load(open("random_forest_model_1.pkl", "rb"))

loaded_model.score(X_test, y_test)
