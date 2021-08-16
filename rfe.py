# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:31:52 2021

BMI3002 HW3 Q8 Recursive Feature Elimination (RFE)

@author: Jonathan Chien
"""

#%% Import dataset, set seed, and randomly split data into train and test sets

import numpy as np
import random

datafile = open("ionosphere.data", "r")

# nObs = 351 and nAttributes = 34
ionosphere = np.empty((0,34), float)
classLabels = np.full(351, np.nan)
for iLine, line in enumerate(datafile.readlines()):
    if line[-2] == "g": # indexing with -1 will yield '\n'
        classLabels[iLine] = 1
    elif line[-2] == "b":
        classLabels[iLine] = 0
    newLine = np.fromstring(line[:-1], dtype=float, sep=',') # 5/6/21 deprecation warning, this line will raise ValueError in future
    ionosphere = np.append(ionosphere, newLine[np.newaxis,:], axis = 0)
    
# Generate random seed, seed numpy PRNG. 
seed = random.randint(10000, 99999)
np.random.seed(seed)

# Randomly split data into train and test.
randperm = np.random.permutation(351)
trainSet = ionosphere[randperm[:234],:]
testSet = ionosphere[randperm[234:],:]
trainLabels = classLabels[randperm[:234]]
testLabels = classLabels[randperm[234:]]

# Initialize dictionary to store selected features from each run.
allSelectedFeatures = {}

 
#%% Set core classifier and run feature selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

# Set core classifier.
classifier = "LR" # "RF", "LSVM", or "LR"

if classifier == "RF":
    estimator = RandomForestClassifier()
elif classifier == "LSVM":
    estimator = LinearSVC()
elif classifier == "LR":
    estimator = LogisticRegression()

# Run feature selection and store results in dictionary.
np.random.seed(seed)
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(trainSet, trainLabels)
selectedFeatures = [iFeature 
                    for iFeature, feature in enumerate(selector.support_) if feature]

if classifier == "RF":
    allSelectedFeatures["RF"] = selectedFeatures
elif classifier == "LSVM":
    allSelectedFeatures["LSVM"] = selectedFeatures
elif classifier == "LR":
    allSelectedFeatures["LR"] = selectedFeatures
    
    
#%% Try all combinations of classifiers and selected feature sets

import sklearn as skl 

# For each classifier, cycle through each of the three feature sets from above.
# Store results in results1 array, where elements 1-3 are accuracy for RF on
# feature sets from RF, LSVM, and LR respectively, elements 4-6 are AUC for 
# LSVM, again on feature sets from RF, LSVM, and LR, and elements 7-9 AUC for 
# LR on feature sets from RF, LSVM, and LR.
iRun = 0
results1 = np.full((9), np.nan)
for currClassifier in ["RF", "LSVM", "LR"]:
    for featureSet in ["RF", "LSVM", "LR"]:
        
        np.random.seed(seed)
        
        if currClassifier == "RF":
            rf = RandomForestClassifier()
            rf.fit(trainSet[:,allSelectedFeatures[featureSet]], trainLabels)
            results1[iRun] \
                = rf.score(testSet[:,allSelectedFeatures[featureSet]], testLabels)
        elif currClassifier == "LSVM":
            lsvm = LinearSVC()
            lsvm.fit(trainSet[:,allSelectedFeatures[featureSet]], trainLabels)
            scores = lsvm.decision_function(testSet[:,allSelectedFeatures[featureSet]])
            results1[iRun] = skl.metrics.roc_auc_score(testLabels, scores)
        elif currClassifier == "LR":
            lr = LogisticRegression()
            lr.fit(trainSet[:,allSelectedFeatures[featureSet]], trainLabels)
            scores = lr.decision_function(testSet[:,allSelectedFeatures[featureSet]])
            results1[iRun] = skl.metrics.roc_auc_score(testLabels, scores)

        iRun += 1    
            
        
#%% Run all three classifiers with the full set of features

# Array to store results.
results2 = np.full(3, np.nan)      

rf = RandomForestClassifier()
rf.fit(trainSet, trainLabels)
results2[0] = rf.score(testSet, testLabels)

lsvm = LinearSVC()
lsvm.fit(trainSet, trainLabels)
scores = lsvm.decision_function(testSet)
results2[1] = skl.metrics.roc_auc_score(testLabels, scores)

lr = LogisticRegression()
lr.fit(trainSet, trainLabels)
scores = lr.decision_function(testSet)
results2[2] = skl.metrics.roc_auc_score(testLabels, scores)


