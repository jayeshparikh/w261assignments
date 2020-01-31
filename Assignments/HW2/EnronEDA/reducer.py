#!/usr/bin/env python
"""
Reducer takes words with their class and partial counts and computes totals.
INPUT:
    word \t class \t partialCount 
OUTPUT:
    word \t class \t totalCount  
"""
import re
import sys

# initialize trackers
current_word = None
spam_count, ham_count = 0,0

# read from standard input
for line in sys.stdin:
    # parse input
    word, is_spam, count = line.split('\t')
    
############ YOUR CODE HERE #########

    if word == current_word:
        if int(is_spam):
            spam_count += int(count)
        else:
            ham_count += int(count)
    # OR ...  
    else:
        if current_word:
            if spam_count > 0:
                print(f'{current_word}\t1\t{spam_count}')
            if ham_count > 0:
                print(f'{current_word}\t0\t{ham_count}')
        # and start a new tally 
        spam_count, ham_count = 0,0
        current_word = word
        if int(is_spam):
            spam_count += int(count)
        else:
            ham_count += int(count)
        

# don't forget the last record! 
if spam_count > 0:
    print(f'{current_word}\t1\t{spam_count}')
if ham_count > 0:
    print(f'{current_word}\t0\t{ham_count}')


############ (END) YOUR CODE #########