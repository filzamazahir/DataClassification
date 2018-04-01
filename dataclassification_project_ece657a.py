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

import time


# Import ToPs class
from TreesOfPredictors import *


# Open an output text file to output the Tree
f = open('output.txt', 'w')


# Load the online news popularity dataset and store it as a pandas dataframe
file_location = 'OnlineNewsPopularity.csv'
news_df_original = pd.read_csv(file_location, sep=', ', engine='python')


# DATA PREPROCESSING

# Drop non-predictive attributes
news_df = news_df_original.drop(['url', 'timedelta'], axis = 1) 

# Remove outliers here - FIX THIS!!!

# # Detecting outliers using Z-score method
# z_scores= news_df.apply(zscore)
# threshold = 5.5 #this value is selected after testing many values and watching x
# news_df = news_df[np.abs(z_scores) < threshold]

# # Using moving mean to fix outliers:
# news_df = news_df.rolling(15, min_periods=1).mean() # values after filtering outliers is saved in news_df again


# Getting dataset ready for training
news_y = news_df['shares']
news_y = news_y.apply(lambda x: 1 if x>=1400 else 0)

news_x = news_df.drop(['shares'], axis = 1)
class_names = ['Unpopular (<1400)', 'Popular (>=1400)']

# Scale Data from 0 to 1, so threshold could be applied on it (news_y already on that scale)
minmax = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(news_x)
news_x = pd.DataFrame(minmax, columns=list(news_x.columns.values))

# Split dataset into test and train set - 20% ( instances out of ) used for testing
news_x_train, news_x_test, news_y_train, news_y_test = train_test_split(news_x, news_y, test_size=0.20, stratify=news_y)
# news_x_test_reset = news_x_test.reset_index(drop=True)





# RANDOM FOREST CLASSIFIER
print('RANDOM FOREST CLASSIFIER')

rf_clf = RandomForestClassifier()
rf_clf.fit(news_x_train, news_y_train)  # Train the data
rfc_prediction = rf_clf.predict(news_x_test)  # Predict using test data, and calculate score

# rfc_score = rf_clf.score(news_x_test, news_y_test)
# print('Random Forest Classifier Score: ', rfc_score)

# # Merge testing data with Random Forest Classifier predictions for plots
# rfc_prediction_df = pd.DataFrame(rfc_prediction, columns=['y'])
# rfc_df = pd.concat([news_x_test_reset, rfc_prediction_df], axis=1)

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
print('EXTRA TREES CLASSIFIER')

xt_clf = ExtraTreesClassifier()
xt_clf.fit(news_x_train, news_y_train)  # Train the data
xtc_prediction = xt_clf.predict(news_x_test)  # Predict using test data

# xtc_score = xt_clf.score(news_x_test, news_y_test)
# print('Extra Trees Classifier Score: ', xtc_score)

# # Merge testing data with Extra Trees Classifier predictions for plots
# xtc_prediction_df = pd.DataFrame(xtc_prediction, columns=['y'])
# xtc_df = pd.concat([news_x_test_reset, xtc_prediction_df], axis=1)

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
print('ADABOOST CLASSIFIER')

ada_clf = AdaBoostClassifier()
ada_clf.fit(news_x_train, news_y_train)  # Train the data
ada_prediction = ada_clf.predict(news_x_test)  # Predict using test data

# ada_score = ada_clf.score(news_x_test, news_y_test)
# print('AdaBoost Classifier Score: ', ada_score)

# # Merge testing data with AdaBoost Classifier predictions for plots
# ada_prediction_df = pd.DataFrame(ada_prediction, columns=['y'])
# ada_df = pd.concat([news_x_test_reset, ada_prediction_df], axis=1)

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

t0 = time.time()
print('TREES OF PREDICTORS - ToPs')
# news_ToPs = ToPs(news_x_train, news_y_train, news_x_test, news_y_test, ['RandomForest', 'ExtraTrees', 'AdaBoost'])  # ToPs made of RandomForest, ExtraTrees and AdaBoost
news_ToPs = ToPs(news_x_train, news_y_train, news_x_test, news_y_test, ['LinearSGD'])  #ToPs made of Linear SGD Classifier
news_ToPs.create_tree(3) # Algorithm 1 & 2 - Create tree
y_true, y_pred_prob = news_ToPs.predict_proba() # Algorithm 3 - Test

# print("Y True: ", y_true)
# print("Y Pred Prob: ", y_pred_prob)
# print(news_ToPs.root_node)

# Evaluation metrics - ToPs
log_loss_ToPs = log_loss(y_true, y_pred_prob)
print('Logarithmic Loss: {0:.3f}'.format(log_loss_ToPs))
f.write('\nogarithmic Loss: {0:.3f}'.format(log_loss_ToPs))

roc_auc_score_ToPs = roc_auc_score(y_true, y_pred_prob)
print('Area under ROC Curve: {0:.3f}'.format(roc_auc_score_ToPs))
f.write('\nArea under ROC Curve: {0:.3f}'.format(roc_auc_score_ToPs))

# NEED ACCURACY, CONFUSION MATRIX, and CLASSIFICATION REPORT STILL - FIX THIS!!


# Output ToPs results to output.txt
f.write('\n\n----- ToPs -----\n\n')
f.write(str(news_ToPs.root_node))
f.write('\ny_true_test\n{0}\n'.format(y_true))
f.write('\ny_pred_prob\n{0}\n'.format(y_pred_prob))

t1 = time.time()
print('Time taken - ToPs: {0:.1f} mins'.format((t1-t0)/60))
f.write('\nTime taken - ToPs: {0:.1f} mins'.format((t1-t0)/60))

# loss_on_leafs = news_ToPs.loss_validation1_of_all_leaf_nodes()
# depth_of_tree = news_ToPs.get_depth_of_tree()

# print('\nLoss values of all leafs {0}'.format(loss_on_leafs))
# f.write('\nLoss values of all leafs {0}'.format(loss_on_leafs))

# print('\nMax Depth of Tree: {0}'.format(depth_of_tree))
# f.write('\nMax Depth of Tree: {0}'.format(depth_of_tree))








