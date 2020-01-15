#!/usr/bin/env python
"""
This script reads word counts from STDIN and aggregates
the counts for any duplicated words.

INPUT & OUTPUT FORMAT:
    word \t count
USAGE (standalone):
    python aggregateCounts_v2.py < yourCountsFile.txt

Instructions:
    For Q7 - Your solution should not use a dictionary or store anything   
             other than a single total count - just print them as soon as  
             you've added them. HINT: you've modified the framework script 
             to ensure that the input is alphabetized; how can you 
             use that to your advantage?
"""

# imports
import sys
from collections import defaultdict


################# YOUR CODE HERE #################

pWord=""
pCount=0
# print(type(pCount))
# stream over lines from Standard Input
for line in sys.stdin:
    # extract words & counts
    word, count  = line.split()
#     print(type(count))
#    print(word + count)
#     print(pWord + str(pCount))
    # tally counts
    if (word == pWord): 
       pCount += int(count)
    else:
       if (pWord != ""):
          print("{}\t{}".format(pWord, pCount))
       pCount=int(count)
       pWord=word

print("{}\t{}".format(pWord, pCount))









################ (END) YOUR CODE #################
