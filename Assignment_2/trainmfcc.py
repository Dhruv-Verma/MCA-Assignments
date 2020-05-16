# Code credits: https://www.kaggle.com/anmour/svm-using-mfcc-features

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC

os.chdir('C:/Users/dhruv/Desktop/MCA Assignment-2/Clean/')
X = np.asarray(pickle.load(open('Training\mfccs_vector.pkl', 'rb')), dtype=float)
y = np.asarray(pickle.load(open('Training\labels.pkl', 'rb')), dtype=int)

X_test = np.asarray(pickle.load(open('Testing\mfccs_vector.pkl', 'rb')), dtype=float)
y_test = np.asarray(pickle.load(open('Testing\labels.pkl', 'rb')), dtype=int)

# Apply scaling for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Apply PCA for dimension reduction
pca = PCA(n_components=512).fit(X_scaled)
X_pca = pca.transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(sum(pca.explained_variance_ratio_)) 

# Fit an SVM model
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size = 0.2, random_state = 0, shuffle = True)
clf = SVC(kernel = 'rbf', probability=True, verbose=True)
clf.fit(X_train, y_train)

print(accuracy_score(clf.predict(X_val), y_val))

# Define the paramter grid for C from 0.001 to 10, gamma from 0.001 to 10
C_grid = [0.001, 0.01, 0.1, 1, 10]
gamma_grid = [0.001, 0.01, 0.1, 1, 10]
param_grid = {'C': C_grid, 'gamma' : gamma_grid}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv = 3, scoring = "accuracy")
grid.fit(X_train, y_train)

# Find the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

# Optimal model
model = SVC(C=10, kernel='rbf', degree=3, gamma=0.001, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
model.fit(X_pca, y)
pickle.dump(model, open('model_mfcc_clean.pkl','wb'))
y_pred = model.predict(X_test_pca)    
print(classification_report(y_test, y_pred))

