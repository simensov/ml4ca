# -*- coding: utf-8 -*-
"""
Simple print functions used if debugging level is on or not.
Can include timestamp if needed.

Created on Fri Mar 16 14:39:09 2018

@author: jonom
"""
from timeit import default_timer as timer


DEBUG = False
#DEBUG = False
#TIMESTAMP = True
TIMESTAMP = False
TIME0 = timer()

#print if debug is on
def log(msg):
    if DEBUG:
        if TIMESTAMP:
            print('('+str(timer()-TIME0)+') '+msg)
        else:
            print(msg)

#print anyway = print
def forcelog(msg):
    if TIMESTAMP:
        print('('+str(timer()-TIME0)+') '+msg)
    else:
        print(msg)
