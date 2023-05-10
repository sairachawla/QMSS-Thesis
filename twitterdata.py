#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 22:16:15 2023

@author: sairachawla
"""

## importing necessary libraries
import os
os.chdir('/Users/sairachawla/Developer/qmss_thesis')
import json
import datetime
from utils import *
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from traintwitterdata import *
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from plotly.offline import plot
  
## prep for queries  
key_phrases = ['abortion', 'abortion+pill', 'birth+control']
states = pd.read_csv('plosonedata2018.csv')['State']
states = [s.replace(' ', '+') for s in states]

data_lst = []

## querying for all states and all key_phrases
for phrase in key_phrases:
    for state in states:
        jsons = [json.loads(info) for info in main(phrase, state)] # loads into a json dump
        texts = [json['text'] for json in jsons] 
        datetimes = [json['created_at'] for json in jsons]
        public_metrics = [json['public_metrics'] for json in jsons]
        author_public_metrics = [json['author']['public_metrics'] for json in jsons]
        
        temp = [[state]*len(texts), [phrase]*len(texts), texts, datetimes, public_metrics, author_public_metrics]
        
        data_lst.append(temp)
        
## fill in a dataframe with api data
tweets = pd.DataFrame()

for lst in data_lst:
    temp = pd.DataFrame()
    temp = temp.assign(State = lst[0], Phrase = lst[1], Tweet = lst[2], Timestamp = lst[3], Public_Metrics = lst[4], Author_Public_Metrics = lst[5])
    tweets = pd.concat([tweets, temp], ignore_index=True)
  
## Engagement Rate Calculation
tweets['Engagement Rate Sum'] = [tweet['retweet_count']+tweet['reply_count']+tweet['like_count']+tweet['quote_count'] for tweet in tweets['Public_Metrics']]
tweets['No. Tweets'] = [tweet['tweet_count'] for tweet in tweets['Author_Public_Metrics']]
tweets['No. Followers'] = [tweet['followers_count'] if tweet['followers_count'] > 0 else np.nan for tweet in tweets['Author_Public_Metrics']]
tweets['Engagement Rate'] = ((tweets['Engagement Rate Sum'] / tweets['No. Tweets']) / tweets['No. Followers']) * 100
tweets.drop(['Public_Metrics', 'Author_Public_Metrics', 'Engagement Rate Sum', 'No. Tweets', 'No. Followers'], axis=1, inplace=True)

## more data cleaning
tweets['Timestamp'] = [datetime.datetime.strptime(timestamp[0:10], '%Y-%m-%d').date() for timestamp in tweets['Timestamp']]

#tweets['Tweet_clean'] = tweets['Tweet'].apply(clean_text)
#tweets['Tweet_sw'] = tweets['Tweet_clean'].apply(rem_sw)
#tweets['Tweet_stem'] = tweets['Tweet_sw'].apply(stem_fun)

## Vader Sentiment Analysis, use vader column
analyzer = SentimentIntensityAnalyzer()
tweets['vader'] = [analyzer.polarity_scores(tweet)['compound'] for tweet in tweets['Tweet']]
#tweets['vader_clean'] = [analyzer.polarity_scores(tweet)['compound'] for tweet in tweets['Tweet_clean']]
#tweets['vader_sw'] = [analyzer.polarity_scores(tweet)['compound'] for tweet in tweets['Tweet_sw']]
#tweets['vader_stem'] = [analyzer.polarity_scores(tweet)['compound'] for tweet in tweets['Tweet_stem']]

## aggregate avg(sentiment) by week; vizualization
## focus on states with the most deviation from each other 
## how correlated they are with each other; nneed patricks help for this 

tweets.rename(columns={'vader':'Vader Sentiment'}, inplace=True)
statespop2022 = pd.read_csv('statepop2022.csv', sep='\t', encoding='utf-8').rename(columns={'US States':'State'})
# final_df = pd.merge(final_df, statespop2022, on='State')
## STANCE PREDICTIONS
test1 = tweets['Tweet'].apply(clean_text)
test2 = test1.apply(rem_sw)
test3 = test2.apply(stem_fun)
test4 = test3.apply(tokenize)
tform_test = tform.transform(test4)
tweets['Predicted Stance'] = t_svc.predict(tform_test.toarray())

## avg(sentiment) and avg(stance) for plosone data
temp2 = []

## not sure about this calculation
for stance in tweets['Predicted Stance']:
    if stance == 'AGAINST':
        temp2.append(100)
    elif stance == 'FAVOR':
        temp2.append(0)
    else:
        temp2.append(50)
        
tweets['Numerical Predicted Stance'] = temp2

tweets['State'] = tweets['State'].apply(replace_plus)
statepop2022 = pd.read_csv('statepop2022.csv', sep='\t', encoding='utf-8')
stateabbrevs = pd.read_csv('stateabbrevs.csv', sep='\t', encoding='utf-8')
tweets = pd.merge(tweets, stateabbrevs, on='State')
tweets.to_csv('tweets.csv', sep='\t', encoding='utf-8')

by_state = tweets.groupby('State')[['Vader Sentiment', 'Engagement Rate', 'Numerical Predicted Stance']].mean()


# =============================================================================
# ## TOPIC MODELING
# tf = TfidfVectorizer()
# test_transformed = tf.fit_transform(test4)
# feature_names = tf.get_feature_names_out()
# n_topics = 5
# lda = LatentDirichletAllocation(n_components=n_topics)
# lda.fit(test_transformed)
# n_top_words = 5
# topic_words = []
# for topic_idx, topic in enumerate(lda.components_):
#     top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
#     top_words = [feature_names[i] for i in top_words_idx]
#     topic_words.append(top_words)
# tweet_topics = lda.transform(test_transformed)
# topic_names = ["Topic " + str(i) + ": " + " ".join(topic_words[i]) for i in range(n_topics)]
# tweet_topics_df = pd.DataFrame(tweet_topics, columns=topic_names)
# =============================================================================

## VISUALIZATIONS
tweets.info() # no missing values and all of correct type
tweets.shape # no. rows by no. cols
tweets[['Engagement Rate', 'Vader Sentiment', 'Numerical Predicted Stance']].describe()
tweets.count(0) # checking there is no missing data
tweets.groupby(['Predicted Stance'])['Tweet'].agg('count') / len(tweets) * 100 # Percent of Tweets Per Category 
tweets['Predicted Stance'].value_counts() # No. of Tweets per categorY
tweets['State'].value_counts() # No. of Tweets Per State; states w/ over 100 tweets only? those with less are more skewed ASK PATRICK
tweets['Phrase'].value_counts() # No. of Tweets Per Phrase
pd.crosstab(tweets['State'], tweets['Predicted Stance']) # no in each cross category
pd.crosstab(tweets['Phrase'], tweets['Predicted Stance'], margins=True, margins_name="Total") # most birth control tweets are in favor, is that an issue
#pd.crosstab(tweets['Phrase'], tweets['State'], tweets['Predicted Stance'])

# section - Results: answers hypothesis 1
by_date1 = pd.DataFrame(tweets.groupby(['Timestamp'])['Tweet'].count()) 
plt.plot(by_date1.index, by_date1['Tweet']) # volume of tweets by date 
plt.xlabel('Timestamp - Month')
plt.ylabel('No. of Tweets')

# section - Results: Classification of Tweets
tweets['Predicted Stance'].hist()
plt.xlabel('Predicted Stance against Legalization of Abortion')
plt.ylabel('No. of Tweets')
tweets.groupby(['Predicted Stance'])['Timestamp'].hist(legend=True) # volume of tweets by date separated by predicted stance
plt.xlabel('Timestamp - Month')
plt.ylabel('No. of Tweets')
tweets.groupby(['Predicted Stance'])['Engagement Rate'].mean() # average engagement rate for tweets separated by predicted stance
tweets.groupby(['Predicted Stance'])['Engagement Rate'].median() # median engagement rate for tweets separated by predicted stance
tweets.groupby(['Predicted Stance'])['Engagement Rate'].max() # max engagement rate for tweets separated by predicted stance
against_max_engagement = tweets[tweets['Engagement Rate'] > 60] # tweet with the maximum engagement score for against stance; TWEET SHOULD HAVE BEEN CLASSIFIED AS FAVOR 
favor_max_engagement = tweets[tweets['Engagement Rate'] == 7.692308] # tweet with the maximum engagement score for favor stance

# section - Results: Sentiment; answers Hypothesis 2
tweets['Vader Sentiment'].hist() # sentiment variation
plt.xlabel('Sentiment')
plt.ylabel('No. of Tweets')
tweets.groupby(['Predicted Stance'])['Vader Sentiment'].hist(legend=True) # sentiment variation by predicted stance
plt.xlabel('Sentiment')
plt.ylabel('No. of Tweets')
tbl4 = pd.DataFrame(tweets.groupby(['Predicted Stance'])['Vader Sentiment'].mean()) # shows AGAINST tweets had more negative sentiment 
tbl4.to_html("tbl4.html")

by_date2 = pd.DataFrame(tweets.groupby(['Timestamp'])['Vader Sentiment'].mean()) # sentiment variation over time
by_date2['Standard Deviation'] = by_date2["Vader Sentiment"].rolling(30).std()
by_date2['Standard Deviation'] = by_date2['Standard Deviation'].shift(1)
by_date2['Average'] = by_date2['Vader Sentiment'].rolling(30).mean()
by_date2['Average'] = by_date2['Average'].shift(1)
## shift row to start a day 31
by_date2['z-score'] = (by_date2['Vader Sentiment'] - by_date2['Average']) / by_date2['Standard Deviation']
#by_date2.to_csv('by_date2.csv')

plt.plot(by_date2.index, by_date2['Vader Sentiment']) # line plot?
plt.xlabel('Timestamp - Month')
plt.ylabel('Average Sentiment Score')
plt.plot(by_date2.index, by_date2['z-score']) # line plot?
plt.xlabel('Timestamp - Month')
plt.ylabel('Rolling Z-Score')
tweets['Month'] = tweets['Timestamp'].apply(lambda x: x.month)
march_tweets = tweets[tweets['Month'] == 3]
june_tweets = tweets[tweets['Month'] == 6]
december_tweets = tweets[tweets['Month'] == 12]
len(march_tweets)
len(june_tweets)
len(december_tweets)
march_tweets_state1 = march_tweets.groupby('State')['Tweet'].count()
march_tweets_state2 = march_tweets.groupby('State')['Vader Sentiment'].median()
june_tweets_state2 = june_tweets.groupby('State')['Vader Sentiment'].median()
december_tweets_state2 = december_tweets.groupby('State')['Vader Sentiment'].median()
by_date3 = pd.DataFrame(tweets.groupby(['Timestamp'])['Tweet'].count())

# section - Results: State-Level
tweets_per_state = pd.DataFrame(tweets.groupby(['Abbrev', 'State'])['Tweet'].count()) # No. of Tweets from each state
tweets_per_state.reset_index(inplace=True)
fig1 = px.choropleth(tweets_per_state,
                    locations='Abbrev', 
                    locationmode='USA-states', 
                    scope="usa",
                    color='Tweet',
                    color_continuous_scale="sunset", 
                    
                    )
fig1 = fig1.update_layout(legend={'title':'No. of Tweets'})
plot(fig1)

statepop2022.rename(columns={'US States': 'State'}, inplace=True)
#tweets_per_state.rename(columns={'Abbrev':'State'}, inplace=True)
tweets_per_state = pd.merge(tweets_per_state, statepop2022, on='State')
tweets_per_state['No. of Tweets per Person'] = tweets_per_state['Tweet']/tweets_per_state['Population 2022']
fig2 = px.choropleth(tweets_per_state,
                    locations='Abbrev', 
                    locationmode='USA-states', 
                    scope="usa",
                    color='No. of Tweets per Person',
                    color_continuous_scale="sunset", 
                    
                    )
plot(fig2)

## aggressive preprocessing
tweets_t = tweets
tweets_t['Tweet'] = tweets_t['Tweet'].apply(clean_text).apply(rem_sw).apply(stem_fun)

washington = tweets_t[tweets_t['State'] == 'Washington']
nh = tweets_t[tweets_t['State'] == 'New Hampshire']
ri = tweets_t[tweets_t['State'] == 'Rhode Island']
oregon = tweets_t[tweets_t['State'] == 'Oregon']
idaho = tweets_t[tweets_t['State'] == 'Idaho']
alabama = tweets_t[tweets_t['State'] == 'Alabama']
michigan = tweets_t[tweets_t['State'] == 'Michigan']
south_dakota = tweets_t[tweets_t['State'] == 'South Dakota']


washington_freq = wrd_freq(washington, 'Tweet')
nh_freq = wrd_freq(nh, 'Tweet')
ri_freq = wrd_freq(ri, 'Tweet')
oregon_freq = wrd_freq(oregon, 'Tweet')
idaho_freq = wrd_freq(idaho, 'Tweet')
alabama_freq = wrd_freq(alabama, 'Tweet')
michigan_freq = wrd_freq(michigan, 'Tweet')
south_dakota_freq = wrd_freq(south_dakota, 'Tweet')

tbl8 = pd.DataFrame(alabama_freq.most_common()[0:11]).to_html('tbl8.html')
tbl9 = pd.DataFrame(idaho_freq.most_common()[0:11]).to_html('tbl9.html')
tbl10 = pd.DataFrame(ri_freq.most_common()[0:11]).to_html('tbl10.html')
tbl11 = pd.DataFrame(nh_freq.most_common()[0:11]).to_html('tbl11.html')

tbl12 = pd.DataFrame(washington_freq.most_common()[0:11]).to_html('tbl12.html')
tbl13 = pd.DataFrame(oregon_freq.most_common()[0:11]).to_html('tbl13.html')
tbl14 = pd.DataFrame(michigan_freq.most_common()[0:11]).to_html('tbl14.html')
tbl15 = pd.DataFrame(south_dakota_freq.most_common()[0:11]).to_html('tbl15.html')
                     
# section - Results: States w/ Abortion Trigger laws, FIND THE NUMBER OF TWEETS PER PERSON
trigger_states = {'State':['Arkansas', 'Idaho', 'Kentucky', 'Louisiana', 'Mississippi', 'Missouri', 'North Dokota',
                   'Oklahona', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Wyoming']}
trigger_states = pd.DataFrame.from_dict(trigger_states)
trigger_states = pd.merge(trigger_states, statespop2022, on='State')
trigger_states.drop('Unnamed: 0', axis=1, inplace=True)
tweets.drop('Unnamed: 0', axis=1, inplace=True)
trigger_tweets = tweets[tweets['State'].isin(trigger_states['State'])]
tbl7 = trigger_tweets.describe()
tbl7.to_html("tbl7.html")                                                  
trigger_prop = len(trigger_tweets) / sum(trigger_states['Population 2022'])  # the no. of tweets per person in trigger states


non_trigger_tweets = tweets[~tweets['State'].isin(trigger_states['State'])] 
non_trigger_tweets = pd.merge(non_trigger_tweets, statespop2022, on='State')
non_trigger_prop = len(non_trigger_tweets) / sum(non_trigger_tweets['Population 2022'])  # the no. of tweets per person in non trigger states
   

# section - Data
# 1 tweet to represent - true against, true favor, true neutral, false favor, false against
tbl1 = tweets.iloc[[8049, 15165, 31578, 16413, 13893],:].head() # indices change every time the data is pulled
tbl1.drop(['Numerical Predicted Stance'], axis=1, inplace=True)  
#tbl1.rename(columns={'vader':'Vader Sentiment'}, inplace=True)  
tbl1.to_html("tbl1.html")         


# section - Discussion
# example tweets with negative sentiment from against and favor predicted stance
tbl4 = tweets[(tweets['Predicted Stance'] == 'AGAINST') & (tweets['Vader Sentiment'] < 0)].to_html('tbl4.html')


# plt.scatter(tweets[tweets['Engagement Rate'] >= 0.0037]['Engagement Rate'], tweets[tweets['Engagement Rate'] >= 0.0037]['vader'])



# tweets.groupby(['Predicted Stance'])['Engagement Rate'].agg('describe')

# temp = tweets[tweets['Engagement Rate'] >= 0.0037]
# temp.groupby(['Predicted Stance'])['Engagement Rate'].agg('describe')
# temp.groupby(['State'])['Engagement Rate'].agg('mean')
                          
                          
# against = pd.DataFrame(tweets[tweets['Predicted Stance'] == 'AGAINST'].groupby(['Timestamp'])['Tweet'].count())
# plt.plot(by_state['Timestamp'], by_state['Tweet']) # line plot?










                          