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

def normalize(pre_stats,mc_downsample):
    stats = pre_stats.copy()
    n_senders = pre_stats.shape[0]
    requests = np.maximum(pre_stats[:,0],np.ones(n_senders))
    delivered = np.maximum(pre_stats[:,1],np.ones(n_senders))
    opens = np.maximum(pre_stats[:,8],np.ones(n_senders))
    
    stats[:,[1,2,3,4]] = np.divide(stats[:,[1,2,3,4]],np.outer(requests,np.ones(4)))
    stats[:,[5,8,9]] = np.divide(stats[:,[5,8,9]],np.outer(delivered,np.ones(3)))
    stats[:,[6,7,10]] = np.divide(stats[:,[6,7,10]],np.outer(opens,np.ones(3)))
    stats[:,[11,12,13,14]] = np.divide(stats[:,[11,12,13,14]],np.outer(mc_downsample*requests,np.ones(4)))
    stats[:,[15,16,17]] = np.divide(stats[:,[15,16,17]],np.outer(requests,np.ones(3)))
    
    return stats

#-------------------------------------------------------------------------------

'''
reduce:
Project the stats onto a lower dimensional space which preserves variance up to the specified cutoff.
'''

def reduce(stats,variance_cutoff):
    
    pca = PCA(whiten=True)
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
Use a training set of dimensionaly reduced stats to train the svr predictor of the punitive actions. 
'''    

def train(train_stats,train_punitive_actions,C=1.0,epsilon=0.1,kernel='rbf',degree=3,gamma=0.0):
        
    # Train each svr and append it to a list of trained svrs
    trained_svr = []
    n_punitive_actions = train_punitive_actions.shape[1]
    for index in range(n_punitive_actions):
        svr = SVR(C=C,epsilon=epsilon,kernel=kernel,degree=degree,gamma=gamma)
        svr.fit(train_stats,train_punitive_actions[:,index])
        trained_svr.append(svr)
        
    return trained_svr

#-------------------------------------------------------------------------------

'''
regress:
Use the trained svrs to predict the punitive actions for a set of test data.
'''

def regress(trained_svr,test_stats,pca=None,reduced=True):
    
    # Project the test_stats onto the same subset of components as the training stats
    if not reduced:
        test_stats = pca.transform(test_stats)
    
    # Perform the regression using the trained svrs
    n_punitive_actions = len(trained_svr)
    predicted_punitive_actions = np.zeros((test_stats.shape[0],n_punitive_actions))
    for index in range(n_punitive_actions):
        svr = trained_svr[index]
        predicted_punitive_actions[:,index] = svr.predict(test_stats)
        
    return predicted_punitive_actions

#-------------------------------------------------------------------------------

'''
evaluate:
Take in a training data set and a test data set, for both of which the punitive
actions are known. Use the svrs from the training set to predict the punitive actions on the test set. Calculate the euclidean error in the predicted and actual punitive actions of the test data. 
'''

def evaluate(predicted_punitive_actions,test_punitive_actions,metric='weak'):
    
    if metric is "weak":
        return np.sqrt(np.sum(np.mean(predicted_punitive_actions-test_punitive_actions,0)**2.0))
        
    elif metric is "strong":
        return np.mean(np.sqrt(np.sum((predicted_punitive_actions-test_punitive_actions)**2.0,1)))
        
    else:
        return None