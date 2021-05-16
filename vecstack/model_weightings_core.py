from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
import numpy as np

class RegressionModelWeightings():

    def __init__(self, X, y, n_members):
        self.X = X
        self.y = y
        self.n_members = n_members

    def evaluate(self, weights):
        score = evaluate_ensemble(self.X, weights, self.y)
        print('Given Weights Score: %.3f' % score)
        return score

    def fit_getWeights(self):
        # define bounds on each weight
        bound_w = [(0.0, 1.0)  for _ in range(self.n_members)]
        # arguments to the loss function
        search_arg = (self.X, self.y)
        # global optimization of ensemble weights
        result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
        # get the chosen weights
        weights = normalize(result['x'])
        print('Optimized Weights: %s' % weights)
        return weights

def ensemble_predictions(members, weights):
    # weighted sum across ensemble members
    summed = tensordot(members, weights, axes=((0),(0)))
    return summed

# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, testy):
	# make prediction
	yhat = ensemble_predictions(members, weights)
	# calculate accuracy
	return mean_squared_error(testy, yhat)
 
# normalize a vector to have unit norm
def normalize(weights):
	# calculate l1 vector norm
	result = norm(weights, 1)
	# check for a vector of all zeros
	if result == 0.0:
		return weights
	# return normalized vector (unit norm)
	return weights / result

# loss function for optimization process, designed to be minimized
def loss_function(weights, members, testy):
	# normalize weights
	normalized = normalize(weights)
	# calculate error rate
	return evaluate_ensemble(members, normalized, testy)