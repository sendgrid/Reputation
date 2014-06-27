import json
import re
import os
from scipy import sparse
import numpy as np
import time

FILTER_MAPPINGS = {
	13: None,
	1: 0,
	4: 1,
	7: 2,
	10: 3,
	16: 4, 
	19: 5,
	25: 6,
}

def decodeFlag(flag):
	features = np.zeros(6)
	for flagIndex, featureIndex in FILTER_MAPPINGS.iteritems():
		flag >> (flagIndex - 1) & 1
		if featureIndex:
			features[featureIndex] = 1
	return features
