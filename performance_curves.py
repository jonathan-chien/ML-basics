# -*- coding: utf-8 -*-
"""
BMI 3002 Machine Learning for Biomedical Data Science HW 1 Question 6

Created on Mon Mar  8 19:59:20 2021

@author: Jonathan Chien
"""

#%% Q6: Read from csv file and create input vectors

import numpy as np
import matplotlib.pyplot as plt

# Read from csv file. First row in csv file comprises column labels.
data = np.genfromtxt('assignment1_problem6_scores.csv', delimiter=',')
groundTruth = data[1:,0]
predScores1 = data[1:,1]
predScores2 = data[1:,2]

del data

#%% Q6a,b: Define plotROC function

def plotROC(groundTruth, predScores, thresholdRange=(0,1), nThresholds=1000,
            legend=True):
    """
    Plots Receiver Operating Characteristic (ROC) curve for evaluation of 
    classifier performance. Also calculates and returns Area Under the Curve
    (AUC).
    
    Parameters
    ----------
    groundTruth : 1D numpy.ndarray
        True class labels. Number of elements is the number of observations.
    predScores : 1D numpy.ndarray
        Predicted class scores. Number of elements is the number of 
        observations.
    thresholdRange : size (2,) array-like
        Range over which thresholds will be tested. Must be specified as 
        (min, max). If order of min and max are reversed, sign of AUC will be 
        negative.
    nThresholds : int
        Number of thresholds to be tested over thresholdRange.
    legend : bool
        Option to include legend with plot output. If calling this function 
        repeatedly to compare performance of multiple classifiers in one plot,
        may want to suppress the legend here and create it after generating 
        all desired plots.

    Returns
    -------
    ROC curve : Line2D
        True positive rate (TPR) plotted against false positive rate (FPR) for 
        a single classifier.
    AUC : float64
        Area under the ROC curve.
    """
    # Define threshold values (moving from maximum to minimum threshold).
    thresholds = np.linspace(thresholdRange[1], thresholdRange[0], nThresholds)
    
    # Calculate true and false positive rates.
    tpr = np.full(nThresholds, np.nan)
    fpr = np.full(nThresholds, np.nan)
    for iThresh, thresh in enumerate(thresholds):
        predPos = predScores > thresh 
        condPos = groundTruth == 1 # condition pos = true pos + false neg
        condNeg = groundTruth == 0 # condition neg = false pos + true neg
        tpr[iThresh] = np.sum(np.logical_and(predPos, condPos)) / \
            np.sum(condPos) 
        fpr[iThresh] = np.sum(np.logical_and(predPos, condNeg)) / \
            np.sum(condNeg) 
               
    # Plot ROC curve.
    plt.plot(fpr, tpr, '-', label="Classifier")
    plt.title("ROC Curve")
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    
    # Plot ROC for no-skill classifier.
    plt.plot([0,1], [0,1], '--', label="No-skill")
    
    # Option to include legend with generation of plot.
    if legend:
        plt.legend(loc = 'lower right')
    
    # Calculate AUC by regarding space between threshold values as a trapezoid,
    # for which area is given as 0.5*width(height1 + height2).
    widths = np.diff(fpr) # numel = nThresholds - 1
    heights1 = tpr[:-1]
    heights2 = tpr[1:]
    AUC = 0.5*np.sum(widths*(heights1+heights2))
    
    return AUC


#%% Q6c,d: Define plotPRC function

def plotPRC(groundTruth, predScores, classLabel, nThresholds=None):
    """
    Plots Precision-Recall curve for evaluation of classifier performance. Also
    calculates and returns area under the precision-recall curve (AUPRC). Would
    probably be ideal to have fewer, larger steps over predictor scores that
    vary more, with many, smaller steps over regions where predictors scores
    vary little from score to score. A less ideal solution might be to have even
    steps but to set nThresholds to be very high so that step sizes are small.

    Parameters
    ----------
    groundTruth : 1D numpy.ndarray
        True class labels. Number of elements is the number of observations.
    predScores : 1D numpy.ndarray
        Predicted class scores. Number of elements is the number of 
        observations.
    classLabel : int
        Which class to consider as positive. May have value 0 or 1.
    nThresholds : int
        Number of thresholds to be tested over thresholdRange.

    Returns
    -------
    PR curve : Line2D
        Precision plotted against recall for a single classifier.   
    AUPRC : float64
        Area under the PR curve.
    """
    # Ensure length of true labels and pred scores match. Assign nThresholds if
    # not given.
    assert np.size(groundTruth) == np.size(predScores)
    if nThresholds == None:
        nThresholds = np.size(groundTruth) - 1
                               
    # Define threshold values (stepping from 0 up to the maximum value 
    # (without including it) of the predicted scores. There must be at least 
    # one observation classified as positive, else nTruePos = 0, 
    # and thus the denominator in calculation of precision will be 0, since
    # nFalsePos > 0 when nTruePos = 0 only in the case of very extreme outliers.
    if classLabel == 1:
        thresholds = np.flip(np.linspace(0, max(predScores), nThresholds,
                                         endpoint=False))
    elif classLabel == 0:
        thresholds = np.flip(np.linspace(1, min(predScores), nThresholds,
                                         endpoint=False))
                                           
    # Calculate precision and recall.
    precision = np.full(nThresholds, np.nan)
    recall = np.full(nThresholds, np.nan)
    for iThresh, thresh in enumerate(thresholds):
        if classLabel == 1: # consider class label 1 to be positive
            predPos   = predScores > thresh 
            condPos   = groundTruth == 1 # condition pos = true pos + false neg
            condNeg   = groundTruth == 0 # condition neg = false pos + true neg
        elif classLabel == 0: # consider class label 0 to be positive
            predPos   = predScores < thresh 
            condPos   = groundTruth == 0 # condition pos = true pos + false neg
            condNeg   = groundTruth == 1 # condition neg = false pos + true neg
        nTruePos  = np.sum(np.logical_and(predPos, condPos))
        nFalsePos = np.sum(np.logical_and(predPos, condNeg))
        precision[iThresh] = nTruePos / (nTruePos + nFalsePos)
        recall[iThresh]    = nTruePos / np.sum(condPos)
    
    # Ensure there is a point with 0 recall (assign it precision 1). This 
    # cannot correspond to a valid threshold, except in cases of extreme
    # outliers (see note above threshold calculation).
    recall = np.insert(recall, 0, 0)    
    precision = np.insert(precision, 0, 1)
    
    # Ensure that there is a point with recall = 1, use last valid value of 
    # precision for this value. Nothing changes if last value for recall is 
    # already 0.
    recall = np.append(recall, 1)
    precision = np.append(precision, precision[-1])
    
    # Plot precision-recall curve.
    plt.plot(recall, precision, '-')
    plt.title("Precision-Recall (PR) Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    
    # Calculate AUPRC by regarding space between threshold values as a 
    # trapezoid, for which area is given as 0.5*width(height1 + height2).
    widths = np.diff(recall) # numel=nThresholds+2-1 (+2 due to 0s added to recall)
    heights1 = precision[:-1]
    heights2 = precision[1:]
    AUPRC = 0.5*np.sum(widths*(heights1+heights2))
    
    # import pdb
    # pdb.set_trace()
    
    return AUPRC


#%% Plot ROC curves for classifier 1 and classifier 2

AUC1 = plotROC(groundTruth, predScores1, legend=False)   
AUC2 = plotROC(groundTruth, predScores2, legend=False)
plt.gca().legend(("Classifier 1", "No-Skill",
                  "Classifier 2",))

#%% Plot Precision-Recall curves for classifier 1 and 2 on class 1

AUPRC1 = plotPRC(groundTruth, predScores1, classLabel=1, nThresholds=20000)
AUPRC2 = plotPRC(groundTruth, predScores2, classLabel=1, nThresholds=20000)
plt.gca().legend(("Classifier 1", "Classifier 2"))

#%% Plot Precision-Recall curves for classifier 1 and 2 on class 0

AUPRC1 = plotPRC(groundTruth, predScores1, classLabel=0, nThresholds=200000)
AUPRC2 = plotPRC(groundTruth, predScores2, classLabel=0, nThresholds=200000)
plt.gca().legend(("Classifier 1", "Classifier 2"))
