#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:16:59 2023

@author: sairachawla
"""

#function to query twitter api 
## CITATION OF CODE
from twitter.config import *
import os
os.chdir('/Users/sairachawla/Developer/qmss_thesis')
def main(key_phrase, state):
    from twarc import Twarc2, expansions
    import datetime
    import json
    
    client = Twarc2(bearer_token=TWITTER_BEARER_TOKEN)
    
    # Specify the start time to be Jan 1, 2018, 0:0:0:0
    start_time = datetime.datetime(2022, 1, 1, 0, 0, 0, 0, datetime.timezone.utc)
    
    # Specify the end time to be Dec 31, 2018, 23:59:59:59
    end_time = datetime.datetime(2022, 12, 31, 23, 59, 59, 59, datetime.timezone.utc)
    
    # specify query
    query = key_phrase + " place:" + state + " -is:retweet"
    
    # The search_all method call the full-archive search endpoint to get Tweets based on the query, start and end times
    search_results = client.search_all(query=query, start_time=start_time, end_time=end_time)

    temp = []
    # Twarc returns all Tweets for the criteria set above, so we page through the results
    for page in search_results:
        # The Twitter API v2 returns the Tweet information and the user, media etc.  separately
        # so we use expansions.flatten to get all the information in a single JSON
        result = expansions.flatten(page)
        for tweet in result:
            # Here we are printing the full Tweet object JSON to the console
            temp.append(json.dumps(tweet))
    
    return temp 

# capitalization, 
def clean_text(str_in):
    import re
    tmp = re.sub("[^A-Za-z']+", " ",str_in).lower().strip()
    #tmp = re.sub("(?:\@|https?\://)\S+", " ",str_in).lower().strip()
    return tmp

def rem_sw(var_in):
    from nltk.corpus import stopwords
    sw = stopwords.words("english")
    tmp = var_in.split()
    # tmp_ar = list()
    # for word_t in tmp:
    #     if word_t not in sw:
    #         tmp_ar.append(word_t)
    tmp_ar = [word_t for word_t in tmp if word_t not in sw]
    tmp_o = ' '.join(tmp_ar)
    return tmp_o

def stem_fun(txt_in):
    from nltk.stem import PorterStemmer
    stem_tmp = PorterStemmer()
    tmp = [stem_tmp.stem(word) for word in txt_in.split()]
    tmp = ' '.join(tmp)
    # tmp = list()
    # for word in txt_in.split():
    #     tmp.append(stem_tmp.stem(word))
    return tmp

def tokenize(txt_in):
    import nltk
    tmp = nltk.word_tokenize(txt_in)
    tmp = ' '.join(tmp)
    return tmp

def replace_plus(txt_in):
    tmp = txt_in.replace('+', ' ')
    return tmp

def wrd_freq(df, col):
    import collections
    temp = " ".join(df[col])
    temp = stem_fun(rem_sw(clean_text(temp)))
    temp = temp.split()
    return collections.Counter(temp)
