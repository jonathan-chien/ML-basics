# -*- coding: utf-8 -*-
"""
BMI 3002 Machine Learning for Biomedical Data Science HW 1 Question 5

Created on Sun Mar  7 16:19:23 2021

@author: Jonathan Chien
"""

#%% Q5: Write BatchScreening class

import matplotlib.pyplot as plt 
from scipy import stats
import numpy as np

class BatchScreening:   
    def __init__(self, hypotheses, prior):
        self.iBatch = 0
        self.hypotheses = hypotheses
        self.nHypotheses = np.size(hypotheses)
        self.priors = prior
        self.likelihoods = np.full((1, self.nHypotheses), np.nan)
        self.posteriors = np.full((1, self.nHypotheses), np.nan)
        assert self.nHypotheses == np.size(self.priors)
        
    def addBatch(self, positives, batchSize, barWidth=0.07):         
        # Calculate likelihood function.
        iBatch = self.iBatch
        if iBatch == 0:
            currentPrior = self.priors
        elif iBatch > 0:
            currentPrior = self.posteriors[iBatch-1,:]
            currentPrior = currentPrior[np.newaxis,:]
            
        likelihoodFx = np.full((1, self.nHypotheses), np.nan)
        for iHypoth, hypoth in np.ndenumerate(self.hypotheses):
            likelihoodFx[iHypoth] = stats.binom.pmf(positives,
                                                    batchSize, hypoth)
        
        # Scale likelihood function to a proper PDF, P(D|H).
        propConstant = np.sum(likelihoodFx)  
        likelihoodFx = likelihoodFx/propConstant   
        
        # Calculate posterior using Bayes' Theorem.                                                                                                      
        currentPosterior = currentPrior*likelihoodFx 
        marginal = np.sum(currentPosterior)
        currentPosterior = currentPosterior/marginal
        
        # Store current batch results.
        if iBatch == 0:
            self.likelihoods = likelihoodFx
            self.posteriors = currentPosterior
        elif iBatch > 0:
            self.priors = np.append(self.priors, currentPrior, 0)
            self.likelihoods = np.append(self.likelihoods, likelihoodFx, 0)
            self.posteriors = np.append(self.posteriors, currentPosterior, 0)
        
        # Plot.
        if iBatch == 0:
            fig, axs = plt.subplots(1,3, sharey='row')
            axs[0].bar(np.squeeze(self.hypotheses),
                    np.squeeze(self.priors[iBatch,:]), barWidth)
            axs[1].bar(np.squeeze(self.hypotheses),
                    np.squeeze(self.likelihoods[iBatch,:]), barWidth)
            axs[2].bar(np.squeeze(self.hypotheses),
                    np.squeeze(self.posteriors[iBatch,:]), barWidth)
        elif iBatch > 0:
            fig, axs = plt.subplots(iBatch+1, 3, sharey='row')
            for iiBatch in range(iBatch+1):
                axs[iiBatch, 0].bar(np.squeeze(self.hypotheses),
                                    np.squeeze(self.priors[iiBatch,:]),
                                    barWidth)
                axs[iiBatch, 1].bar(np.squeeze(self.hypotheses),
                                    np.squeeze(self.likelihoods[iiBatch,:]),
                                    barWidth)
                axs[iiBatch, 2].bar(np.squeeze(self.hypotheses),
                                    np.squeeze(self.posteriors[iiBatch,:]),
                                    barWidth)
        
        # Update batch index.
        self.iBatch += 1
        
        # Return all results from first batch up to current batch.
        return self.priors, self.likelihoods, self.posteriors
    
        
#%% Q5: Instantiate a BatchScreening object with uniform prior, set currentBatch to 0

hypotheses = np.arange(0, 1.1, 0.1)
hypotheses = hypotheses[np.newaxis,:]
initialPrior = np.tile(1/np.size(hypotheses), (1, np.size(hypotheses))) 
screening = BatchScreening(hypotheses, initialPrior)
currentBatch = 0


#%% Q5: Run all successive batches manually (uniform prior)

batchResults = np.array([[7, 25],
                         [5, 25],
                         [10,25],
                         [8, 25],
                         [9, 25],
                         [4, 25]])

screening.addBatch(batchResults[currentBatch,0], batchResults[currentBatch,1])
currentBatch += 1


#%% Q5a: Can we go home yet? (uniform prior)

hypotheses = np.arange(0, 1.1, 0.1)
hypotheses = hypotheses[np.newaxis,:]
initialPrior = np.tile(1/np.size(hypotheses), (1, np.size(hypotheses))) 
screening = BatchScreening(hypotheses, initialPrior)

batchResults = np.array([[7, 25],
                         [5, 25],
                         [10,25],
                         [8, 25],
                         [9, 25],
                         [4, 25]])

for iBatch, batch in enumerate(batchResults):
    priors, likelihoods, posteriors = screening.addBatch(batch[0], batch[1])
    if np.sum(posteriors[-1, np.squeeze(hypotheses>0.2)]) / \
       np.sum(posteriors[-1,:]) > 0.8:
        if iBatch == 0:
            print("Group instructed to quarantine after 1 batch.")
        elif iBatch > 0:
            print(f"Group instructed to quarantine after {iBatch+1} batches.")
        break
    if iBatch+1 == 6:
        print("Group cleared to go home.")
        
        
#%% Q5b: Can we go home yet? (skeptical prior)       

hypotheses = np.arange(0, 1.05, 0.05)
hypotheses = hypotheses[np.newaxis,:]
initialPrior = np.concatenate((np.array([[0, 0.5, 0.5]]),
                                np.tile(0, (1, 18))), axis=1)                              
screening = BatchScreening(hypotheses, initialPrior)

batchResults = np.array([[7,25],
                         [5,25],
                         [10,25],
                         [8,25],
                         [9,25],
                         [4,25]])

for iBatch, batch in enumerate(batchResults):
    priors, likelihoods, posteriors = screening.addBatch(batch[0], batch[1],
                                                           barWidth=0.04)
    if np.sum(posteriors[-1, np.squeeze(hypotheses>0.2)]) / \
       np.sum(posteriors[-1,:]) > 0.8:
        if iBatch == 0:
            print("Group instructed to quarantine after 1 batch.")
        elif iBatch > 0:
            print(f"Group instructed to quarantine after {iBatch+1} batches.")
        break
    if iBatch+1 == 6:
        print("Group cleared to go home.")
