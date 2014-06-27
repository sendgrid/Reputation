from scipy import sparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVR

#-------------------------------------------------------------------------------\
def getAnovaKernel(gamma, D):
	return lambda x, y: anovaKernel(x, y, gamma, D) 

def anovaKernel(x, y, gamma, D):
	gramMatrix = np.zeros([len(x), len(y)])
	for i in range(len(x)):
		for j in range(len(y)):
			entry = 0
			for k in range(len(x[0])):
				entry += np.power(
					np.exp( 
						-gamma* np.power(x[i][k]-y[j][k], 2) 
					), D)
			gramMatrix[i, j] = entry
	return gramMatrix

#-------------------------------------------------------------------------------

'''
reduce:
Project the stats onto a lower dimensional space which preserves variance up to the specified cutoff.
'''

def reduce(stats,variance_cutoff):
    
    pca = PCA()
    pca.fit(stats)
    evr = pca.explained_variance_ratio_
    variance_sum = 0.0
    n_components = stats.shape[0]
    
    # Select the number of components to keep based on the variance cutoff 
    for index in range(n_components):
        variance_sum += evr[index]
        if variance_sum > variance_cutoff:
            n_components = index+1
            break
            
    # Project the stats onto the subset of components
    pca = PCA(n_components = n_components)
    pca.fit(stats)
    reduced_stats = pca.transform(stats)
        
    return reduced_stats,pca
    
#-------------------------------------------------------------------------------

'''
train:
Use a training set of dimensionaly reduced stats to train the svr predictor of the punative actions. 
'''    

def train(reduced_stats,punative_actions,C,epsilon,kernel,degree,gamma):
    
    svr = SVR(C=C,epsilon=epsilon,kernel=kernel,degree=degree,gamma=gamma)
    
    # Train each svr and append it to a list of trained svrs
    trained_svr = []
    n_punative_actions = punative_actions.shape[0]
    for index in range():
        svr.fit(reduced_stats,punative_actions[index,:])
        trained_svr.append(svr)
        
    return trained_svr,n_punative_actions

#-------------------------------------------------------------------------------

'''
regress:
Use the trained svrs to predict the punative actions for a set of test data.
'''

def regress(trained_svr,pca,test_stats,n_punative_actions):
    
    # Project the test_stats onto the same subset of components as the training stats
    reduced_stats = pca.transform(test_stats)
    
    # Perform the regression using the trained svrs
    predicted_punative_actions = np.zeros(reduced_stats.shape[0],n_punative_actions)
    for index in range(n_punative_actions):
        svr = trained_svr[index]
        predicted_punative_actions[:,index] = svr.predict(test_data)
        
    return predicted_punative_actions

#-------------------------------------------------------------------------------

'''
evaluate:
Take in a training data set and a test data set, for both of which the punative
actions are known. Use the svrs from the training set to predict the punative actions on the test set. Calculate the euclidean error in the predicted and actual punative actions of the test data. 
'''
