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


import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import pyplot


# Import ToPs class
from TreesOfPredictors import *

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

# Split dataset into test and train set - 25% (9911 instances out of 39644) used for testing
news_x_train, news_x_test, news_y_train, news_y_test = train_test_split(news_x, news_y, test_size=0.25, random_state=42)
# news_x_test_reset = news_x_test.reset_index(drop=True)




# RANDOM FOREST CLASSIFIER
rf_clf = RandomForestClassifier()

# Train the data
rf_clf.fit(news_x_train, news_y_train)

# Predict using test data, and calculate score
rfc_prediction = rf_clf.predict(news_x_test)

# rfc_score = rf_clf.score(news_x_test, news_y_test)
# print('Random Forest Classifier Score: ', rfc_score)

# # Merge testing data with Random Forest Classifier predictions for plots
# rfc_prediction_df = pd.DataFrame(rfc_prediction, columns=['y'])
# rfc_df = pd.concat([news_x_test_reset, rfc_prediction_df], axis=1)

print('RANDOM FOREST CLASSIFIER')
rfc_accuracy = cross_val_score(rf_clf, news_x, news_y, scoring='accuracy')
print('Accuracy: {0:.3f} ({1:.3f})'.format(rfc_accuracy.mean(), rfc_accuracy.std()))

rfc_log_loss = cross_val_score(rf_clf, news_x, news_y, scoring='neg_log_loss')
print('Logarithmic Loss: {0:.3f} ({1:.3f})'.format(rfc_log_loss.mean(), rfc_log_loss.std()))

rfc_area_roc = cross_val_score(rf_clf, news_x, news_y, scoring='roc_auc')
print('Area under ROC Curve: {0:.3f} ({1:.3f})'.format(rfc_area_roc.mean(), rfc_area_roc.std()))

rfc_confusion_matrix = confusion_matrix(news_y_test, rfc_prediction)
print('Confusion Matrix: \n', rfc_confusion_matrix)

rfc_classification_report = classification_report(news_y_test, rfc_prediction, target_names=class_names)
print(' Classification Report:')
print(rfc_classification_report)

# rfc_precision, rfc_recall, rfc_threshold = precision_recall_curve(news_y_test, rfc_prediction)
# print('Precission: ', rfc_precision)
# print('Recall: ', rfc_recall)
# print('Threshold: ', rfc_threshold)

print('\n')




# EXTRA TREES CLASSIFIER
xt_clf = ExtraTreesClassifier()

# Train the data
xt_clf.fit(news_x_train, news_y_train)

# Predict using test data, and calculate score
xtc_prediction = xt_clf.predict(news_x_test)

# xtc_score = xt_clf.score(news_x_test, news_y_test)
# print('Extra Trees Classifier Score: ', xtc_score)

# # Merge testing data with Extra Trees Classifier predictions for plots
# xtc_prediction_df = pd.DataFrame(xtc_prediction, columns=['y'])
# xtc_df = pd.concat([news_x_test_reset, xtc_prediction_df], axis=1)

print('EXTRA TREES CLASSIFIER')
xtc_accuracy = cross_val_score(xt_clf, news_x, news_y, scoring='accuracy')
print('Accuracy: {0:.3f} ({1:.3f})'.format(xtc_accuracy.mean(), xtc_accuracy.std()))

xtc_log_loss = cross_val_score(xt_clf, news_x, news_y, scoring='neg_log_loss')
print('Logarithmic Loss: {0:.3f} ({1:.3f})'.format(xtc_log_loss.mean(), xtc_log_loss.std()))

xtc_area_roc = cross_val_score(xt_clf, news_x, news_y, scoring='roc_auc')
print('Area under ROC Curve: {0:.3f} ({1:.3f})'.format(xtc_area_roc.mean(), xtc_area_roc.std()))

xtc_confusion_matrix = confusion_matrix(news_y_test, xtc_prediction)
print('Confusion Matrix: \n', xtc_confusion_matrix)

xtc_classification_report = classification_report(news_y_test, xtc_prediction, target_names=class_names)
print(' Classification Report:')
print(xtc_classification_report)

# xtc_precision, xtc_recall, xtc_threshold = precision_recall_curve(news_y_test, xtc_prediction)
# print('Precission: ', xtc_precision)
# print('Recall: ', xtc_recall)
# print('Threshold: ', xtc_threshold)

print('\n')




# ADABOOST CLASSIFIER
ada_clf = AdaBoostClassifier()

# Train the data
ada_clf.fit(news_x_train, news_y_train)

# Predict using test data, and calculate score
ada_prediction = ada_clf.predict(news_x_test)

# ada_score = ada_clf.score(news_x_test, news_y_test)
# print('AdaBoost Classifier Score: ', ada_score)

# # Merge testing data with AdaBoost Classifier predictions for plots
# ada_prediction_df = pd.DataFrame(ada_prediction, columns=['y'])
# ada_df = pd.concat([news_x_test_reset, ada_prediction_df], axis=1)

print('ADABOOST CLASSIFIER')
ada_accuracy = cross_val_score(ada_clf, news_x, news_y, scoring='accuracy')
print('Accuracy: {0:.3f} ({1:.3f})'.format(ada_accuracy.mean(), ada_accuracy.std()))

ada_log_loss = cross_val_score(ada_clf, news_x, news_y, scoring='neg_log_loss')
print('Logarithmic Loss: {0:.3f} ({1:.3f})'.format(ada_log_loss.mean(), ada_log_loss.std()))

ada_area_roc = cross_val_score(ada_clf, news_x, news_y, scoring='roc_auc')
print('Area under ROC Curve: {0:.3f} ({1:.3f})'.format(ada_area_roc.mean(), ada_area_roc.std()))

ada_confusion_matrix = confusion_matrix(news_y_test, ada_prediction)
print('Confusion Matrix: \n', ada_confusion_matrix)

ada_classification_report = classification_report(news_y_test, ada_prediction, target_names=class_names)
print(' Classification Report:')
print(ada_classification_report)

# ada_precision, ada_recall, ada_threshold = precision_recall_curve(news_y_test, ada_prediction)
# print('Precission: ', ada_precision)
# print('Recall: ', ada_recall)
# print('Threshold: ', ada_threshold)

print('\n')


# print(np.count_nonzero(ada_prediction == news_y_test) / float(news_y_test.size))
# print(np.mean(ada_prediction==news_y_test))

# #scattering original data:
# plt.scatter(news_x_train.iloc[:, 0], news_x_train.iloc[:, 1], c=news_y_train, s=32, cmap='summer')
# plt.show()

# # AdaBoost Classifier

# # Create and fit an AdaBoosted decision tree
# bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
# bdt.fit(news_x_train, news_y_train)





# TREES OF PREDICTORS CLASSIFIER (ToPs)









