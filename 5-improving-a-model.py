from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing  ## sk
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns

# First predictions = baseline predictions
# First model = baseline model

# Improvements that can be made on the data side:
# -Could we collect more data
# -Could we improve hygiene and robustness of current data

# Improvements on the model side:
# -Is there a better model we can use
# -Could we improve the current model by tuning it

# Parameters = tell the model what patterns to look for in data
# Hyperparameters = settings on a model you can adjust to (potentially) improve its ability to find patterns

# 3 ways to tune hyperparameters
# -By hand
# -Randomly with RandomSearchCV
# -Exhaustively with GridSearchCV

clf = RandomForestClassifier()

print(clf.get_params())

# Tuning hyperparameters by hand


