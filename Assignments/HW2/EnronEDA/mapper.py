#!/usr/bin/env python
"""
Mapper tokenizes and emits words with their class.
INPUT:
    ID \t SPAM \t SUBJECT \t CONTENT \n
OUTPUT:
    word \t class \t count 
"""
import re
import sys

# initialize trackers

# read from standard input
for line in sys.stdin:
    cur_word = None
    cur_count = 0
    # parse input
    docID, _class, subject, body = line.split('\t')
    # tokenize
    words = re.findall(r'[a-z]+', subject + ' ' + body)
    #print(words)
############ YOUR CODE HERE #########
    for word in words:
        if word == cur_word: 
            #print('if: '+word+' '+cur_word+' '+_class)
            #print(f'{word}\t{cur_word}')
            cur_count += int(1)
        # OR emit current total and start a new tally 
        else: 
            #print('else: '+word+' '+cur_word+' '+_class)
            #print({word}+' '+{cur_word})
            if cur_word:
                print(f'{cur_word}\t{_class}\t{cur_count}')
            cur_word, cur_count  = word, int(1)
    print(f'{cur_word}\t{_class}\t{cur_count}')       

# don't forget the last record! 
#print(f'{cur_word}\t{_class}\t{cur_count}')

############ (END) YOUR CODE #########