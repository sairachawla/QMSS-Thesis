#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:30:07 2023

@author: sairachawla
"""

## importing necessary libraries
import os
os.chdir('/Users/sairachawla/Developer/qmss_thesis')
import pandas as pd
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
import numpy as np


# reading in the dataset
df = pd.read_csv('semeval2016-task6-trainingdata.txt', sep='\t', encoding='Windows-1252')


## data cleaning
## aggressive preprocessing!
df = df[df['Target'] == 'Legalization of Abortion']
df['Tweet'] = df['Tweet'].apply(clean_text)
df['Tweet'] = df['Tweet'].apply(rem_sw)
df['Tweet'] = df['Tweet'].apply(stem_fun)
df['Tweet'] = df['Tweet'].apply(tokenize)

## since the signal is higher for against stance tweets, we are balancing the data 
df_majority = df[df['Stance'] == 'AGAINST']
df_minority1 = df[df['Stance'] == 'NONE']
df_minority2 = df[df['Stance'] == 'FAVOR']

df_minority1_upsampled = resample(df_minority1,
                                  replace=True,
                                  n_samples = df['Stance'].value_counts()['AGAINST'],
                                  random_state=42)
df_minority2_upsampled = resample(df_minority2,
                                  replace=True,
                                  n_samples = df['Stance'].value_counts()['AGAINST'],
                                  random_state=42)

df = pd.concat([df_majority, df_minority1_upsampled, df_minority2_upsampled])



# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['Stance'], test_size=0.2, random_state=42)

## TOKENIZATION METHODS
# =============================================================================
# ## count vectorization
# cform = CountVectorizer(ngram_range=(1, 3))
# cform_train_df = pd.DataFrame(cform.fit_transform(X_train).toarray())
# cform_train_df.columns = cform.get_feature_names_out()
# cform_test_df = cform.transform(X_test)
# =============================================================================

## tfidf vectorization
tform = TfidfVectorizer(ngram_range=(1, 3))
tform_train_df = pd.DataFrame(tform.fit_transform(X_train).toarray())
tform_train_df.columns = tform.get_feature_names_out()
tform_test_df = tform.transform(X_test)

## feature selection with chi squared; REDUCED ACCURACY SCORE
#k = 1000
#ch2 = SelectKBest(chi2, k=k)
#tform_train_df = ch2.fit_transform(tform_train_df, y_train)
#tform_test_df = ch2.transform(tform_test_df)

## MODELING - SVC, RF, Naive Bayes with GridSearchCV

## count models 
# =============================================================================
# c_svc = SVC(kernel='linear', probability=True)
# c_svc.fit(cform_train_df, y_train)
# y_pred_c_svc = c_svc.predict(cform_test_df.toarray())
# accuracy_c_svc = (y_pred_c_svc == y_test).mean()
# =============================================================================

# =============================================================================
# c_rf = RandomForestClassifier()
# c_rf.fit(cform_train_df, y_train)
# y_pred_c_rf = c_rf.predict(cform_test_df)
# accuracy_c_rf = (y_pred_c_rf == y_test).mean()
# =============================================================================

# =============================================================================
# c_nb_param_grid = {'alpha': [0.1, 1, 10]}
# c_nb = MultinomialNB()
# c_nb_grid_search = GridSearchCV(c_nb, c_nb_param_grid, cv=5)
# c_nb_grid_search.fit(cform_train_df, y_train)
# c_nb = c_nb_grid_search.best_estimator_
# y_pred_c_nb = c_nb.predict(cform_test_df)
# accuracy_c_nb = (y_pred_c_nb == y_test).mean()
# =============================================================================

## tfidf models 

t_svc_param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
t_svc = SVC()
t_svc_grid_search = GridSearchCV(t_svc, t_svc_param_grid, cv=5)
t_svc_grid_search.fit(tform_train_df, y_train)
# Print the best parameters and the corresponding score
#print(f"Best parameters: {t_svc_grid_search.best_params_}")
#print(f"Best score: {t_svc_grid_search.best_score_}")
t_svc = t_svc_grid_search.best_estimator_
y_pred_t_svc = t_svc.predict(tform_test_df.toarray())
accuracy_t_svc = (y_pred_t_svc == y_test).mean()
precision = precision_score(y_test, y_pred_t_svc, average='weighted')
recall = recall_score(y_test, y_pred_t_svc, average='weighted')
cv_score = np.mean(cross_val_score(t_svc, tform_train_df, y_train, cv=5))

# =============================================================================
# t_rf = RandomForestClassifier()
# t_rf.fit(tform_train_df, y_train)
# y_pred_t_rf = t_rf.predict(tform_test_df)
# accuracy_t_rf = (y_pred_t_rf == y_test).mean()
# =============================================================================

# =============================================================================
# t_nb = MultinomialNB()
# t_nb.fit(tform_train_df, y_train)
# accuracy_t_nb = t_nb.score(tform_test_df, y_test)
# =============================================================================


## MISC:
## after classification, break up into in favor VS against
## look at sentiment/emotions (emotional dimension apis) (maybe try 2-3)
## which category is more passionate
## apply best model to classify tweets 
## gridearchcv ; chi-squared ; aggressive preprocessing ; majority votes of three models 