from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing  ## sk
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Three ways to evaluate SkiKit-learn models/estimators
# 1. Built-in score() method
# 2. The "scoring" parameter
# 3. Problem-specific metric functions

heart_disease = pd.read_csv("data/heart-disease.csv")

# 4.1  Using score() method
np.random.seed(42)
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
model.score(X_test, y_test)

housing = fetch_california_housing()

# organizing data set into DataFrame and assigning names to columns programatically given their labels in the dataset

housing_df = pd.DataFrame(housing["data"], columns=housing["feature_names"])

np.random.seed(42)

housing_df["Target (MedHouseVal)"] = housing["target"]

X = housing_df.drop("Target (MedHouseVal)", axis=1)
y = housing_df["Target (MedHouseVal)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

model = RandomForestRegressor(n_estimators=10)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# 4.2 Evaluating a model using the scoring parameter

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
clf_single_score = clf.score(X_test, y_test)

# Instead of making one test group of the passed test_size, we run train_test_split
# multiple times to get different sets of training and test data, then returns a score for each

clf_cross_val_score = np.mean(cross_val_score(clf, X, y, cv=5))

print(f"single score {clf_single_score} cross val score mean {clf_cross_val_score}")

# 4.2.1 Classification model evaluation metrics

# Accuracy

np.mean(clf_cross_val_score)

# Area under the Receiver operation characteristic curve (AUC/ROC)
# ROC curves are a comparison of a model's true positive rate (tpr) versus a model's false positive rate (fpr)

# Make predictions with probabilities

y_probs = clf.predict_proba(X_test)
print(y_probs[:10], len(y_probs))

y_probs_positive = y_probs[:, 1]  # splices out just the firsts column of any array, which is the chance of a positive
# result in the case of y_probs
print(y_probs_positive[:10])

# Calculate fpr, tpr, and thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)
# tests our predictions against the corresponding items we
# already know the outcome for to see the false positive/negative rate of our model

# Check FPR

print(fpr)

# Confusion Matrix (Quick way to compare the labels a model predicts and the actual labels it was supposed to predict.)
# In essence, giving you an idea of where a model is getting confused

y_preds = clf.predict(X_test)
print(confusion_matrix(y_test, y_preds))

# Visualize confusion matrix with pd crosstab()

# provides chart that shows how many instances there are of the predicted and actual labels (y) matched or
# didn't match
pd.crosstab(y_test, y_preds, rownames=["Actual Labels"], colnames=["Predicted Labels"])

# Make our confusion matrix more visual with Seaborn's heatmap()

# Set the font scale
sns.set(font_scale=1.5)
# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_preds)

sns.heatmap(conf_mat)
plt.show()

