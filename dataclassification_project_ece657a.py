# Project - ECE657A  (Group __)
# Filza Mazahir 20295951  &  Tarneem Barayyan 20645942 

# Libraries used: pandas, numpy, scikit-learn, matplotlib

# Algorithm oriented project on Data Classification
# Data Source: https://archive.ics.uci.edu/ml/datasets/online+news+popularity

# Import Libraries 
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
# Load the online news popularity dataset and store it as a pandas dataframe
file_location = 'OnlineNewsPopularity.csv'
news_df_original = pd.read_csv(file_location, sep=', ', engine='python')


# DATA PREPROCESSING

# Drop non-predictive attributes
news_df = news_df_original.drop(['url', 'timedelta'], axis = 1) 

# Detecting outliers using Z-score method
z_scores= news_df.apply(zscore)
threshold = 5.5 #this value is selected after testing many values and watching x
news_df = news_df[np.abs(z_scores) < threshold]

# Using moving mean to fix outliers:
news_df = news_df.rolling(15, min_periods=1).mean() # values after filtering outliers is saved in news_df again


# Getting dataset ready for training
news_y = news_df['shares']
news_y = news_y.apply(lambda x: 1 if x>=1400 else 0)
news_x = news_df.drop(['shares'], axis = 1)
class_names = ['Unpopular (<1400)', 'Popular (>=1400)']

#choosing importance features to reduce the number of features from 58 to 39
clf = ExtraTreesClassifier()
clf = clf.fit(news_x, news_y)
news_x_importance = clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
news_x_reduced = model.transform(news_x) 



# Split dataset into test and train set - 50% 
news_x_train, news_x_test, news_y_train, news_y_test = train_test_split(news_x_reduced, news_y, test_size=0.50, random_state=42)
# news_x_test_reset = news_x_test.reset_index(drop=True)










