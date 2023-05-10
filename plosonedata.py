#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 22:11:01 2023

@author: sairachawla
"""

## importing necessary libraries
import os
os.chdir('/Users/sairachawla/Developer/qmss_thesis')
import pandas as pd
import numpy as np
import requests
from twitterdata import *
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

## importing first dataset that describes healthcare and abortion related features in 2018
df = pd.read_excel('plosone/plosonedata.xlsx')
## cleaning the data
## reassigning column names
columnNames = df.iloc[0] 
df = df[1:] 
df.columns = columnNames
## dropping unnecessary rows
df = df.drop([52, 53])
## converting columns to correct dtype and fromat
for i in df.columns[1:]:
    df[i] = pd.to_numeric(df[i])
df['% Opine that abortion should be illegal'] = 100*df['% Opine that abortion should be illegal']
df['% Population 18-24 years old'] = 100*df['% Population 18-24 years old']

## adding information from twitter
final_df = pd.merge(df, by_state, on='State')
final_df.rename(columns={'vader':'Vader Sentiment'}, inplace=True)

statespop2022 = pd.read_csv('statepop2022.csv', sep='\t', encoding='utf-8').rename(columns={'US States':'State'})
final_df = pd.merge(final_df, statespop2022, on='State')
final_df.to_csv('plosoneupdated.csv', sep='\t', encoding='utf-8')

## VISUALIZATIONS

# section - Results: State-Level
final_df.describe().to_html('tbl6.html')

final_df.sort_values('Vader Sentiment',inplace=True)
plt.bar(final_df['State'], final_df['Vader Sentiment'])
plt.xticks(fontsize=8, rotation = 90)
plt.xlabel('State')
plt.ylabel('Average Sentiment')

final_df.sort_values('Engagement Rate',inplace=True)
plt.bar(final_df['State'], final_df['Engagement Rate'])
plt.xticks(fontsize=8, rotation = 90)
plt.xlabel('State')
plt.ylabel('Average Engagement Rate')

final_df.sort_values('State',inplace=True)
plt.bar(final_df['State'], final_df['Numerical Predicted Stance'], color='green')
plt.xticks(fontsize=8, rotation = 90)
plt.xlabel('State')
plt.ylabel('Numerical Predicted Stance')
plt.twinx()
plt.bar(final_df['State'], final_df['% Opine that abortion should be illegal'], alpha=0.8)
plt.xticks(fontsize=8, rotation = 90)
plt.ylabel('% Opine that abortion should be illegal')


plt.scatter(final_df['% Opine that abortion should be illegal'], final_df['Numerical Predicted Stance']) 
plt.xlabel('% Opine that abortion should be illegal')
plt.ylabel('Numerical Predicted Stance')
final_df.corr(method='pearson').loc['% Opine that abortion should be illegal', 'Numerical Predicted Stance'] # correlation

# t-test
public_mean = np.mean(final_df['% Opine that abortion should be illegal'])
twitter_mean = np.mean(final_df['Numerical Predicted Stance'])
public_std = np.std(final_df['% Opine that abortion should be illegal'])
twitter_std = np.std(final_df['Numerical Predicted Stance'])
ttest, pval = ttest_ind(final_df['% Opine that abortion should be illegal'], final_df['Numerical Predicted Stance'])




# section - Results: States w/ Abortion Trigger laws
trigger_states = ['Arkansas', 'Idaho', 'Kentucky', 'Louisiana', 'Mississippi', 'Missouri', 'North Dokota',
                  'Oklahona', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Wyoming']
trigger_df = final_df[final_df['State'].isin(trigger_states)]
tbl6 = trigger_df.describe()
tbl6.to_html('tbl6.html')



# section - Data
tbl2 = final_df.head()
tbl2.to_html('tbl2.html')

lst1 = ['State',
        'RSV For "abortion" (2018)',
        'RSV For "abortion pill" (2018)',
        'RSV For "birth control" (2018)',
        'Overall Health Systems Performance Score', 
        'Cost of Care',
        'Access to Care',
        'Health Outcomes',
        'Number of Abortion Facilities',
        'Number of women ages 15-49 per abortion facility',
        'Number abortion restrictions',
        'Number abortion protections',
        '% Opine that abortion should be illegal',
        '% Unintended pregnancy',
        '% Population 18-24 years old',
        '% Rural population', 
        'Vader Sentiment',
        'Engagement Rate',
        'Numerical Predicted Stance']
lst2 = ['',
        'relative search value for "abortion" in 2018',
        'relative search value for "abortion pill" in 2018',
        'relative search value for "birth control" in 2018'
        'aggregate performance score of health systems',
        '',
        '',
        '',
        '',
        '', 
        '',
        '',
        'survey opinion from US 2014 Religious Census Survey',
        '',
        '', 
        '',
        '',
        'average sentiment of tweets from given state',
        'average engagement rate of tweets for given state',
        'average opinion on whether or not abortion should be illegal of tweets for given state']

tbl3 = pd.DataFrame({'Column Name': lst1, 'Definition': lst2})
tbl3.to_html('tbl3.html')


































