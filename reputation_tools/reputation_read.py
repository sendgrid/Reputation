import json
import re
import os
from scipy import sparse
import numpy as np
import time

#-------------------------------------------------------------------------------

# verdict_flag is not treated as a field
fields = [
    "requests",
    "delivered",
    "bounces",
    "blocked",
    "invalid",
    "invalid_domain",
    "spam_report",
    "clicks",
    "unique_clicks",
    "opens",
    "unique_opens",
    "unsubscribes",
]

#-------------------------------------------------------------------------------

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
	features = np.zeros(7)
	for flagIndex, featureIndex in FILTER_MAPPINGS.iteritems():
		flag >> (flagIndex - 1) & 1
		if featureIndex:
			features[featureIndex] = 1
	return features

#-------------------------------------------------------------------------------

'''
read_senders:
Read in data from the senders and average it over the gathered time. Keep a dictionary of rows/senders.
'''

def read_senders(datafile):
    
    row_dict = {}
    uid_dict = {}
    entry_count = []
    sender_data = []
    
    
    n_fields = len(fields)
    
    # Read each line of the data file
    for content in datafile:
        decoded = json.loads(content)
        uid = decoded['user_id']
        
        # Create new entries if needed and increment the count of visits to a particular sender to normalize later
        if not uid_dict.has_key(uid):
            sender_index = len(entry_count) + 1
            row_dict[sender_index] = uid
            uid_dict[uid] = sender_index
            entry_count.append(1)
            sender_data.append([0]*n_fields)
        else:
            sender_index = uid_dict[uid]
            entry_count[sender_index] += 1
            
        # Add counts for each of the fields and mail channels + blacklists (stored in verdict flag)    
        for index in range(n_fields):
            sender_data[sender_index][index] += decoded[fields[index]]
        flag = decoded["verdict_flag"]
        features = decodeFlag(flag)
        n_features = len(features)
        for index in range(n_features):
            sender_data[sender_index][n_fields+index] = features[index]
        
    # Convert to arrays and normalize the sums to form averages
    sender_data = np.array(sender_data)
    entry_count = np.array(entry_count)
    sender_data = (sender_data.T/entry_count).T

    return sender_data
    