#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 22:16:15 2023

@author: sairachawla
"""

## importing necessary libraries
from twarc import Twarc2, expansions
import datetime
import json
from twitter.config import *
import pandas as pd

client = Twarc2(bearer_token=TWITTER_BEARER_TOKEN)

#function to query twitter api 
def main(key_phrase, state):
    # Specify the start time to be Jan 1, 2018, 0:0:0:0
    start_time = datetime.datetime(2018, 1, 1, 0, 0, 0, 0, datetime.timezone.utc)
    
    # Specify the end time to be Dec 31, 2018, 23:59:59:59
    end_time = datetime.datetime(2018, 12, 31, 23, 59, 59, 59, datetime.timezone.utc)
    
    # specify query
    query = key_phrase + " place:" + state
    
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
    
key_phrases = ['abortion', 'abortion+pill', 'birth+control']
states = pd.read_csv('plosonedata2018.csv')['State']

states = [s.replace(' ', '+') for s in states]

data_lst = []

for phrase in key_phrases:
    for state in states:
        jsons = [json.loads(info) for info in main(phrase, state)]
        texts = [json['text'] for json in jsons]
        
        temp = [state, phrase, texts]
        
        data_lst.append(temp)
        
tweets = pd.DataFrame(data_lst)

## this what the df looks like currently, idk what the best way to organize the data is 
## i want it to reflect the data structure we used for the_data 
## now the dataset wpuld be ready for sentiment analysis- how would we proceed with that?
## 

# temp_phrase = key_phrases[0]
# temp_state = states[39]

# jsons_ny_abortion_2018 = [json.loads(info) for info in main(temp_phrase, temp_state)]

# text_ny_abortion_2018 = [json['text'] for json in jsons_ny_abortion_2018]



                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          