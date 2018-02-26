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
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import zscore


import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import pyplot

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

#detecting outliers using Z-score method
z_scores= news_x.apply(zscore)
threshold = 5.5 #this value is selected after testing many values and watching x
x = news_x[np.abs(z_scores) < threshold]

#using moving mean:
rolling_mean = x.rolling(15, min_periods=1).mean()
#values of x after filtering outliers is saved in rolling_mean

# Split dataset into test and train set - 25% (9911 instances out of 39644) used for testing
news_x_train, news_x_test, news_y_train, news_y_test = train_test_split(rolling_mean, news_y, test_size=0.25, random_state=42)

#AdaBoost Classifier:
Ada = AdaBoostClassifier()
Ada.fit(news_x_train, news_y_train)

#finding accuracy:
test_prediction = Ada.predict(news_x_test)
print(np.count_nonzero(test_prediction == news_y_test) / float(news_y_test.size))
#it gave us 56.6%
print(np.mean(test_prediction==news_y_test))
print(Ada.score(news_x_test,news_y_test))

#scattering original data:
plt.scatter(news_x_train.iloc[:, 0], news_x_train.iloc[:, 1], c=news_y_train, s=32, cmap='summer')
plt.show()
# Random Forest Classifier




# Extra Trees Classifier




# AdaBoost Classifier

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
bdt.fit(news_x_train, news_y_train)


# Trees of Predictors Classifier





