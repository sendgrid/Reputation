import json
import re
import os
from scipy import sparse
import numpy as np
import time

#-------------------------------------------------------------------------------

# verdict_flag is not treated as a field
sender_fields = [
    "requests",
    "delivered",
    "bounces",
    "blocked",
    "invalid",
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
    if flag == u'NULL':
        return features
        
    for flagIndex, featureIndex in FILTER_MAPPINGS.iteritems():
        if featureIndex is not None:
            features[featureIndex] = flag >> (flagIndex - 1) & 1
        
    return features

#-------------------------------------------------------------------------------

'''
read_senders:
Read in data from the senders and average it over the gathered time. Keep a dictionary of rows/senders.
'''

def read_senders(datafile):
    
    n_fields = len(sender_fields)
    row_dict = {}
    uid_dict = {}
    sender_stats = np.zeros((0,0))    
    
    # Read each line of the data file
    rawtext = datafile.read().replace("\'","\"").replace("NULL","\"NULL\"") 
    decoded = json.loads(rawtext)
    for content in decoded:
        
        uid = content['user_id']
        flag = content["verdict_flag"]
        if flag == u'NULL':
            continue
        features = decodeFlag(flag)
        n_features = len(features)
        
        # Create new entries if needed and increment the count of visits to a particular sender to normalize later
        if not uid_dict.has_key(uid):
            sender_index = len(uid_dict)
            row_dict[sender_index] = uid
            uid_dict[uid] = sender_index
            temp_sender_stats = sender_stats
            sender_stats = np.zeros((sender_index+1,n_fields+n_features))
            if not sender_index == 0:
                sender_stats[:-1,:] = temp_sender_stats
        else:
            sender_index = uid_dict[uid]
            
        # Add counts for each of the fields and mail channels + blacklists (stored in verdict flag)
        for index in range(n_fields):
            sender_stats[sender_index,index] += content[sender_fields[index]]
        sender_stats[sender_index,n_fields:] += features

    return sender_stats,row_dict,uid_dict
    
#-------------------------------------------------------------------------------

'''
read_punitive_actions:
Read in data on punitive actions taken against senders. Keep a dictionary of col/punitive actions
'''

def read_punitive_actions(datafile,uid_dict):
            
    n_senders = len(uid_dict)
    col_dict = {}
    pa_dict = {}
    punitive_actions = np.zeros((0,0))
    count = 0
    
    # Read each line of the data file
    rawtext = datafile.read().replace("\'","\"").replace("NULL","\"NULL\"")
    decoded = json.loads(rawtext)
    for content in decoded:
        uid = content['id']
        pa = content['event_name']
        
        # Create new entries if needed
        if uid_dict.has_key(uid):
            row = uid_dict[uid]
            if not pa_dict.has_key(pa):
                pa_index = len(pa_dict)
                col_dict[pa_index] = pa
                pa_dict[pa] = pa_index
                temp_punitive_actions = punitive_actions
                punitive_actions = np.zeros((n_senders,pa_index+1))
                if not pa_index == 0:
                    punitive_actions[:,:-1] = temp_punitive_actions
            else:
                pa_index = pa_dict[pa]
            
            punitive_actions[row,pa_index] += 1.0
        else:
            count += 1
    print(count)
        
    return punitive_actions,col_dict,pa_dict
        
            
            
            
            
        
        
        
