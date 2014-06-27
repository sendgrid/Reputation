import numpy as np

def reputation(M, w):
	M = M.transpose()
	return np.maximum(M[:len(w)], M[len(w):]).transpose()*w
	
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