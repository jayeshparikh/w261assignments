#!/usr/bin/env python
"""
Reducer aggregates word counts by class and emits frequencies.

INPUT:
    partitionKey \t word \t ham_count \t spam_count 
OUTPUT:
    word \t ham_count,spam_count,ham conditional probability,spam conditional probability
    
Instructions:
    Again, you are free to design a solution however you see 
    fit as long as your final model meets our required format
    for the inference job we designed in Question 8. Please
    comment your code clearly and concisely.
    
    A few reminders: 
    1) Don't forget to emit Class Priors (with the right key).
    2) In python2: 3/4 = 0 and 3/float(4) = 0.75
"""
##################### YOUR CODE HERE ####################

import re                                                   
import sys                                                  
import numpy as np      

from operator import itemgetter
import os

cur_word = None
word_ham_count, word_spam_count = 0, 0
total_ham_count, total_spam_count = 0, 0

# read input key-value pairs from standard input
for line in sys.stdin:
    key, word, ham_count, spam_count = line.strip().split()   
    #print(f"{word}\t{ham_count}\t{spam_count}")    

    # tally counts from current key
    if word == '!Total':
        total_ham_count = ham_count
        total_spam_count = spam_count
    elif word == 'ClassPriors':
        ham_cProb = int(ham_count) / (int(ham_count) + int(spam_count))
        spam_cProb = int(spam_count) / (int(ham_count) + int(spam_count))
        print(f'{word}\t{ham_count},{spam_count},{ham_cProb},{spam_cProb}')
    elif word == cur_word: 
        word_ham_count += int(ham_count)
        word_spam_count += int(spam_count)
    # OR ...  
    else:
        #print('else ', word)
        # emit realtive frequency
        if cur_word:
            #print(f'{word}\t{ham_count},{spam_count},{ham_cProb},{spam_cProb}')
            ham_cProb = int(word_ham_count) / int(total_ham_count)
            spam_cProb = int(word_spam_count) / int(total_spam_count)
            print(f'{cur_word}\t{word_ham_count},{word_spam_count},{ham_cProb},{spam_cProb}')
            word_ham_count, word_spam_count = 0,0
        # and start a new tally 
        cur_word  = word
        word_ham_count += int(ham_count)
        word_spam_count += int(spam_count)

# don't forget the last record!
ham_cProb = int(word_ham_count) / int(total_ham_count)
spam_cProb = int(word_spam_count) / int(total_spam_count)
print(f'{cur_word}\t{word_ham_count},{word_spam_count},{ham_cProb},{spam_cProb}')
#print(f'{cur_word}\t{word_ham_count},{word_spam_count}')
    


































##################### (END) CODE HERE ####################