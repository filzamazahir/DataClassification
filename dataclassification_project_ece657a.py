# Project - ECE657A  (Group __)
# Filza Mazahir 20295951  &  Tarneem Barayyan 20645942 

# Libraries used: pandas, numpy, scikit-learn, matplotlib

# Algorithm oriented project on Data Classification
# Data Source: https://archive.ics.uci.edu/ml/datasets/online+news+popularity

# Import Libraries 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# Load the online news popularity dataset and store it as a pandas dataframe
file_location = 'OnlineNewsPopularity.csv'
news_df_original = pd.read_csv(file_location, sep=', ', engine='python')


# Data Preprocessing

# Drop non-predictive attributes
news_df = news_df_original.drop(['url', 'timedelta'], axis = 1) 

# Check for outliers/noise/NaN on news_df




# Getting dataset ready
news_y = news_df['shares']
news_y = news_y.apply(lambda x: 1 if x>=1400 else 0)
news_x = news_df.drop(['shares'], axis = 1)


# Split dataset into test and train set - 25% (9911 instances out of 39644) used for testing
news_x_train, news_x_test, news_y_train, news_y_test = train_test_split(news_x, news_y, test_size=0.25, random_state=42)


# Random Forest Classifier




# Extra Trees Classifier




# AdaBoost Classifier




# Trees of Predictors Classifier





