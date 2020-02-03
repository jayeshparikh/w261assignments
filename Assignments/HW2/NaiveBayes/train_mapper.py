#!/usr/bin/env python
"""
Mapper reads in text documents and emits word counts by class.
INPUT:                                                    
    DocID \t true_class \t subject \t body                
OUTPUT:                                                   
    partitionKey \t word \t class0_partialCount,class1_partialCount       
    

Instructions:
    You know what this script should do, go for it!
    (As a favor to the graders, please comment your code clearly!)
    
    A few reminders:
    1) To make sure your results match ours please be sure
       to use the same tokenizing that we have provided in
       all the other jobs:
         words = re.findall(r'[a-z]+', text-to-tokenize.lower())
         
    2) Don't forget to handle the various "totals" that you need
       for your conditional probabilities and class priors.
       
Partitioning:
    In order to send the totals to each reducer, we need to implement
    a custom partitioning strategy.
    
    We will generate a list of keys based on the number of reduce tasks 
    that we read in from the environment configuration of our job.
    
    We'll prepend the partition key by hashing the word and selecting the
    appropriate key from our list. This will end up partitioning our data
    as if we'd used the word as the partition key - that's how it worked
    for the single reducer implementation. This is not necessarily "good",
    as our data could be very skewed. However, in practice, for this
    exercise it works well. The next step would be to generate a file of
    partition split points based on the distribution as we've seen in 
    previous exercises.
    
    Now that we have a list of partition keys, we can send the totals to 
    each reducer by prepending each of the keys to each total.
       
"""

import re                                                   
import sys                                                  
import numpy as np      

from operator import itemgetter
import os

#################### YOUR CODE HERE ###################

def getPartitionKey(word,count):
    """ 
    Helper function to assign partition key ('A', 'B', or 'C').
    Args:  word (str) ; count (int)
    """   
    if count < 4:
        return 'A'
    elif count < 8:
        return 'B'
    else:
        return 'C'

# initialize trackers
spam_count, ham_count = 0, 0
total_spam_count, total_ham_count = 0, 0
classprior_spam_count, classprior_ham_count = 0, 0
ClassPriors='ClassPriors'
ClassTotal ='!Total'

# read from standard input
for line in sys.stdin:
    # parse input
    docID, _class, subject, body = line.split('\t')
    
    cur_word = None
    if int(_class):
        classprior_spam_count += int(1)
    else:
        classprior_ham_count += int(1)
    #print(body)
    
    # tokenize
    words = re.findall(r'[a-z]+', subject.lower()+ ' ' + body.lower())
    #words = re.findall(r'[a-z]+', subject + ' ' + body)
    #print(words)
    for word in words:
        if word == cur_word: 
            #print('if: '+word+' '+cur_word+' '+_class)
            #print(f'{word}\t{cur_word}')
            if int(_class):
                spam_count += int(1)
                total_spam_count += int(1)
            else:
                ham_count += int(1)
                total_ham_count += int(1)
        # OR emit current total and start a new tally 
        else: 
            #print('else: '+word+' '+cur_word+' '+_class)
            #print({word}+' '+{cur_word})
            if cur_word:
                partitionKey = getPartitionKey(word, spam_count) 
                print(f'{partitionKey}\t{cur_word}\t{ham_count}\t{spam_count}')
            cur_word = word
            spam_count, ham_count = 0,0
            if int(_class):
                spam_count += int(1)
                total_spam_count += int(1)
            else:
                ham_count += int(1)
                total_ham_count += int(1)
                
    partitionKey = getPartitionKey(word, spam_count)        
    print(f'{partitionKey}\t{cur_word}\t{ham_count}\t{spam_count}')

# print ClassPrior count
partitionKey = getPartitionKey(word, classprior_spam_count) 
print(f'{partitionKey}\t{ClassPriors}\t{classprior_ham_count}\t{classprior_spam_count}')
print(f'{partitionKey}\t{ClassTotal}\t{total_ham_count}\t{total_spam_count}')
    


#################### (END) YOUR CODE ###################