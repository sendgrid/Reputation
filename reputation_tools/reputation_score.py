import numpy as np

def reputation(M, w):
	M = M.transpose()
	return np.maximum(M[:len(w)], M[len(w):]).transpose()*w
