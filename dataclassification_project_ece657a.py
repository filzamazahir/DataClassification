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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support

import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import pyplot

from math import inf

import time


# Import ToPs class
from TreesOfPredictors import *


# Open an output text file to output the Tree
f = open('output.txt', 'w')

print('Loading Dataset and Data preprocessing')

# Load the online news popularity dataset and store it as a pandas dataframe
file_location = 'OnlineNewsPopularity.csv'
news_df_original = pd.read_csv(file_location, sep=', ', engine='python')


# DATA PREPROCESSING

# Drop non-predictive attributes
news_df = news_df_original.drop(['url', 'timedelta'], axis = 1) 


# Removing outliers 
news_df = news_df[(np.abs(zscore(news_df)) < 5).all(axis = 1)] # outliers are those that are 5 standard deviations away from mean
# news_df = news_df.rolling(15, min_periods=1).mean() # Using rolling mean

news_df = news_df.reset_index(drop=True)

# Drop columns that have a low correlation - FIX THIS!!




# Getting dataset ready for training
news_y = news_df['shares']
news_y = news_y.apply(lambda x: 1 if x>=1400 else 0)

news_x = news_df.drop(['shares'], axis = 1)
class_names = ['Unpopular (<1400)', 'Popular (>=1400)']

print('Class Balance\n', news_y.value_counts())

# Scale Data from 0 to 1, so threshold could be applied on it (news_y already on that scale)
minmax = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(news_x)
news_x = pd.DataFrame(minmax, columns=list(news_x.columns.values))

# Split dataset into test and train set - 20% ( instances out of ) used for testing
news_x_train, news_x_test, news_y_train, news_y_test = train_test_split(news_x, news_y, test_size=0.20, stratify=news_y)
news_x_test_reset = news_x_test.reset_index(drop=True)


# Output results in a csv file
prediction_data = {}
prediction_data['y_true_rfc_xtc_ada'] = news_y_test.reset_index(drop=True).as_matrix()




# RANDOM FOREST CLASSIFIER
print('RANDOM FOREST CLASSIFIER')
f.write('\nRANDOM FOREST CLASSIFIER\n')

t_rf1 = time.time()

rf_clf = RandomForestClassifier(n_estimators=50, max_depth=8)
rf_clf.fit(news_x_train, news_y_train)  # Train the data
rfc_prediction = rf_clf.predict(news_x_test)  # Predict using test data, and calculate score
rfc_predict_prob = rf_clf.predict_proba(news_x_test) # Predict class probabilities

prediction_data['rfc_pred_prob'] = rfc_predict_prob[:, 1]
prediction_data['rfc_pred'] = rfc_prediction


t_rf2 = time.time()

# # Merge testing data with Random Forest Classifier predictions for plots
rfc_prediction_df = pd.DataFrame(rfc_prediction, columns=['y'])
rfc_df = pd.concat([news_x_test_reset, rfc_prediction_df], axis=1)

rfc_accuracy = accuracy_score(news_y_test, rfc_prediction)
print('Accuracy: {0:.3f}'.format(rfc_accuracy))
f.write('Accuracy: {0:.3f}\n'.format(rfc_accuracy))

rfc_log_loss = log_loss(news_y_test, rfc_predict_prob[:, 1])
# print('Logarithmic Loss: {0:.3f} ({1:.3f})'.format(rfc_log_loss.mean(), rfc_log_loss.std()))
print('Logarithmic Loss: {0:.3f}'.format(rfc_log_loss))
f.write('Logarithmic Loss: {0:.3f}\n'.format(rfc_log_loss))

rfc_area_roc = roc_auc_score(news_y_test, rfc_predict_prob[:, 1])
# print('Area under ROC Curve: {0:.3f} ({1:.3f})'.format(rfc_area_roc.mean(), rfc_area_roc.std()))
print('Area under ROC Curve: {0:.3f}'.format(rfc_area_roc))
f.write('Area under ROC Curve: {0:.3f}\n'.format(rfc_area_roc))

rfc_time_taken = t_rf2 - t_rf1
print('Time taken: {0:.3f} seconds'.format(rfc_time_taken))
f.write('Time taken: {0:.3f} seconds\n'.format(rfc_time_taken))

rfc_confusion_matrix = confusion_matrix(news_y_test, rfc_prediction)
print('Confusion Matrix: \n', rfc_confusion_matrix)
f.write('Confusion Matrix: \n')
f.write(str(rfc_confusion_matrix))
f.write('\n')


rfc_classification_report = classification_report(news_y_test, rfc_prediction, target_names=class_names)
RF_precision, RF_recall, RF_f1score, RF_support = precision_recall_fscore_support(news_y_test, rfc_prediction, average=None)
print('Classification Report:')
print(rfc_classification_report)
f.write('Classification Report:\n')
f.write(rfc_classification_report)


## rfc_precision, rfc_recall, rfc_threshold = precision_recall_curve(news_y_test, rfc_predict_prob)
## print('Precission: ', rfc_precision)
## print('Recall: ', rfc_recall)
## print('Threshold: ', rfc_threshold)

print('\n')
f.write('\n')




# EXTRA TREES CLASSIFIER
print('EXTRA TREES CLASSIFIER')
f.write('\nEXTRA TREES CLASSIFIER\n')

t_xt1 = time.time()

xt_clf = ExtraTreesClassifier(n_estimators=50, max_depth=8)
xt_clf.fit(news_x_train, news_y_train)  # Train the data
xtc_prediction = xt_clf.predict(news_x_test)  # Predict using test data
xtc_predict_prob = xt_clf.predict_proba(news_x_test) # Predict class probabilities

prediction_data['xtc_pred_prob'] = xtc_predict_prob[:, 1]
prediction_data['xtc_pred'] = xtc_prediction

t_xt2 = time.time()

# # Merge testing data with Extra Trees Classifier predictions for plots
xtc_prediction_df = pd.DataFrame(xtc_prediction, columns=['y'])
xtc_df = pd.concat([news_x_test_reset, xtc_prediction_df], axis=1)

xtc_accuracy = accuracy_score(news_y_test, xtc_prediction)
# print('Accuracy: {0:.3f} ({1:.3f})'.format(xtc_accuracy.mean(), xtc_accuracy.std()))
print('Accuracy: {0:.3f}'.format(xtc_accuracy))
f.write('Accuracy: {0:.3f}\n'.format(xtc_accuracy))

xtc_log_loss = log_loss(news_y_test, xtc_predict_prob[:, 1])
# print('Logarithmic Loss: {0:.3f} ({1:.3f})'.format(xtc_log_loss.mean(), xtc_log_loss.std()))
print('Logarithmic Loss: {0:.3f}'.format(xtc_log_loss))
f.write('Logarithmic Loss: {0:.3f}\n'.format(xtc_log_loss))

xtc_area_roc = roc_auc_score(news_y_test, xtc_predict_prob[:, 1])
# print('Area under ROC Curve: {0:.3f} ({1:.3f})'.format(xtc_area_roc.mean(), xtc_area_roc.std()))
print('Area under ROC Curve: {0:.3f}'.format(xtc_area_roc))
f.write('Area under ROC Curve: {0:.3f}\n'.format(xtc_area_roc))

xtc_time_taken = t_xt2 - t_xt1
print('Time taken: {0:.3f} seconds'.format(xtc_time_taken))
f.write('Time taken: {0:.3f} seconds\n'.format(xtc_time_taken))

xtc_confusion_matrix = confusion_matrix(news_y_test, xtc_prediction)
print('Confusion Matrix: \n', xtc_confusion_matrix)
f.write('Confusion Matrix: \n')
f.write(str(xtc_confusion_matrix))
f.write('\n')

xtc_classification_report = classification_report(news_y_test, xtc_prediction, target_names=class_names)
XT_precision, XT_recall, XT_f1score, XT_support = precision_recall_fscore_support(news_y_test, xtc_prediction, average=None)
print('Classification Report:')
print(xtc_classification_report)
f.write('Classification Report:\n')
f.write(xtc_classification_report)


## xtc_precision, xtc_recall, xtc_threshold = precision_recall_curve(news_y_test, xtc_predict_prob)
## print('Precission: ', xtc_precision)
## print('Recall: ', xtc_recall)
## print('Threshold: ', xtc_threshold)

print('\n')
f.write('\n')



# ADABOOST CLASSIFIER
print('ADABOOST CLASSIFIER')
f.write('\nADABOOST CLASSIFIER\n')

t_ada1 = time.time()

ada_clf = AdaBoostClassifier(n_estimators=50)
ada_clf.fit(news_x_train, news_y_train)  # Train the data
ada_prediction = ada_clf.predict(news_x_test)  # Predict using test data
ada_predict_prob = ada_clf.predict_proba(news_x_test) # Predict class probabilities

prediction_data['ada_pred_prob'] = ada_predict_prob[:, 1]
prediction_data['ada_pred'] = ada_prediction

t_ada2 = time.time()

# # Merge testing data with AdaBoost Classifier predictions for plots
ada_prediction_df = pd.DataFrame(ada_prediction, columns=['y'])
ada_df = pd.concat([news_x_test_reset, ada_prediction_df], axis=1)

ada_accuracy = accuracy_score(news_y_test, ada_prediction)
# print('Accuracy: {0:.3f} ({1:.3f})'.format(ada_accuracy.mean(), ada_accuracy.std()))
print('Accuracy: {0:.3f}'.format(ada_accuracy))
f.write('Accuracy: {0:.3f}\n'.format(ada_accuracy))

ada_log_loss = log_loss(news_y_test, ada_predict_prob[:, 1])
# print('Logarithmic Loss: {0:.3f} ({1:.3f})'.format(ada_log_loss.mean(), ada_log_loss.std()))
print('Logarithmic Loss: {0:.3f}'.format(ada_log_loss))
f.write('Logarithmic Loss: {0:.3f}\n'.format(ada_log_loss))

ada_area_roc = roc_auc_score(news_y_test, ada_predict_prob[:, 1])
# print('Area under ROC Curve: {0:.3f} ({1:.3f})'.format(ada_area_roc.mean(), ada_area_roc.std()))
print('Area under ROC Curve: {0:.3f}'.format(ada_area_roc))
f.write('Area under ROC Curve: {0:.3f}\n'.format(ada_area_roc))

ada_time_taken = t_ada2 - t_ada1
print('Time taken: {0:.3f} seconds'.format(ada_time_taken))
f.write('Time taken: {0:.3f} seconds\n'.format(ada_time_taken))

ada_confusion_matrix = confusion_matrix(news_y_test, ada_prediction)
print('Confusion Matrix: \n', ada_confusion_matrix)
f.write('Confusion Matrix: \n')
f.write(str(ada_confusion_matrix))
f.write('\n')

ada_classification_report = classification_report(news_y_test, ada_prediction, target_names=class_names)
ADA_precision, ADA_recall, ADA_f1score, ADA_support = precision_recall_fscore_support(news_y_test, ada_prediction, average=None)
print('Classification Report:')
print(ada_classification_report)
f.write('Classification Report:\n')
f.write(ada_classification_report)


## ada_precision, ada_recall, ada_threshold = precision_recall_curve(news_y_test, ada_predict_prob)
## print('Precission: ', ada_precision)
## print('Recall: ', ada_recall)
## print('Threshold: ', ada_threshold)

print('\n')
f.write('\n')




# TREES OF PREDICTORS CLASSIFIER (ToPs)

print('TREES OF PREDICTORS - LINEAR')
f.write('\nTREES OF PREDICTORS - LINEAR\n')

t_ToPs_l1 = time.time()

news_ToPs_linear = ToPs(news_x_train, news_y_train, news_x_test, news_y_test, ['LinearSGD'])  #ToPs made of Linear SGD Classifier
news_ToPs_linear.create_tree(inf) # Algorithm 1 & 2 - Create tree
ToPs_linear_y_true, ToPs_linear_y_pred_prob = news_ToPs_linear.predict_proba() # Algorithm 3 - Test

prediction_data['y_true_ToPs_linear'] = ToPs_linear_y_true.as_matrix()
prediction_data['ToPs_linear_pred_prob'] = ToPs_linear_y_pred_prob.as_matrix()


ToPs_linear_precision, ToPs_linear_recall, ToPs_linear_threshold = precision_recall_curve(ToPs_linear_y_true, ToPs_linear_y_pred_prob)
prc_data_ToPs_linear = {'precision': ToPs_linear_precision[:-1], 'recall': ToPs_linear_recall[:-1], 'threshold': ToPs_linear_threshold}
precision_recall_threshold_df = pd.DataFrame(prc_data_ToPs_linear)
precision_recall_threshold_df.to_csv('Precision_Recall_Threshold.csv', index=False)

# print('Precision', ToPs_linear_precision)
# print('Recall', ToPs_linear_recall)
# print('Threshold', ToPs_linear_threshold)


ToPs_linear_prediction = ToPs_linear_y_pred_prob.apply(lambda x: 1 if x >=0.5 else 0)
prediction_data['ToPs_linear_pred'] = ToPs_linear_prediction.as_matrix()


t_ToPs_l2 = time.time()


ToPs_linear_accuracy = accuracy_score(ToPs_linear_y_true, ToPs_linear_prediction) 
print('Accuracy: {0:.3f}'.format(ToPs_linear_accuracy))
f.write('Accuracy: {0:.3f}\n'.format(ToPs_linear_accuracy))

ToPs_linear_log_loss = log_loss(ToPs_linear_y_true, ToPs_linear_y_pred_prob)
print('Logarithmic Loss: {0:.3f}'.format(ToPs_linear_log_loss))
f.write('Logarithmic Loss: {0:.3f}\n'.format(ToPs_linear_log_loss))

ToPs_linear_area_roc = roc_auc_score(ToPs_linear_y_true, ToPs_linear_y_pred_prob)
print('Area under ROC Curve: {0:.3f}'.format(ToPs_linear_area_roc))
f.write('Area under ROC Curve: {0:.3f}\n'.format(ToPs_linear_area_roc))

ToPs_linear_time_taken = t_ToPs_l2 - t_ToPs_l1
print('Time taken: {0:.3f} minutes'.format(ToPs_linear_time_taken/60))
f.write('Time taken: {0:.3f} minutes\n'.format(ToPs_linear_time_taken/60))

ToPs_linear_confusion_matrix = confusion_matrix(ToPs_linear_y_true, ToPs_linear_prediction)
print('Confusion Matrix: \n', ToPs_linear_confusion_matrix)
f.write('Confusion Matrix: \n')
f.write(str(ToPs_linear_confusion_matrix))
f.write('\n')

ToPs_linear_classification_report = classification_report(ToPs_linear_y_true, ToPs_linear_prediction, target_names=class_names)
ToPs_linear_precision, ToPs_linear_recall, ToPs_linear_f1score, ToPs_linear_support = precision_recall_fscore_support(ToPs_linear_y_true, ToPs_linear_prediction, average=None)
print('Classification Report:')
print(ToPs_linear_classification_report)
f.write('Classification Report:\n')
f.write(ToPs_linear_classification_report)

# Output ToPs tree to output.txt
f.write('\n\n----- ToPs classifier tree (linear) -----\n\n')
f.write(str(news_ToPs_linear.root_node))
f.write('\nMax Depth of Tree: {0}'.format(news_ToPs_linear.get_depth_of_tree()))
f.write('\n')


print('\n')
f.write('\n')




print('TREES OF PREDICTORS - 3 CLASSIFIERS (RandomForest ExtraTrees & AdaBoost')
f.write('\nTREES OF PREDICTORS - 3 CLASSIFIERS (RandomForest ExtraTrees & AdaBoost\n')

t_ToPs_3clf1 = time.time()

news_ToPs_three_clf = ToPs(news_x_train, news_y_train, news_x_test, news_y_test, ['RandomForest', 'ExtraTrees', 'AdaBoost'])  # ToPs made of RandomForest, ExtraTrees and AdaBoost
news_ToPs_three_clf.create_tree(3) # Algorithm 1 & 2 - Create tree
ToPs_three_clf_y_true, ToPs_three_clf_y_pred_prob = news_ToPs_three_clf.predict_proba() # Algorithm 3 - Test

prediction_data['y_true_ToPs_3clf'] = ToPs_three_clf_y_true.as_matrix()
prediction_data['ToPs_3clf_pred_prob'] = ToPs_three_clf_y_pred_prob.as_matrix()


ToPs_three_clf_prediction = ToPs_three_clf_y_pred_prob.apply(lambda x: 1 if x >=0.5 else 0)
prediction_data['ToPs_3clf_pred'] = ToPs_three_clf_prediction.as_matrix()


t_ToPs_3clf2 = time.time()


ToPs_three_clf_accuracy = accuracy_score(ToPs_three_clf_y_true, ToPs_three_clf_prediction) 
print('Accuracy: {0:.3f}'.format(ToPs_three_clf_accuracy))
f.write('Accuracy: {0:.3f}\n'.format(ToPs_three_clf_accuracy))

ToPs_three_clf_log_loss = log_loss(ToPs_three_clf_y_true, ToPs_three_clf_y_pred_prob)
print('Logarithmic Loss: {0:.3f}'.format(ToPs_three_clf_log_loss))
f.write('Logarithmic Loss: {0:.3f}\n'.format(ToPs_three_clf_log_loss))

ToPs_three_clf_area_roc = roc_auc_score(ToPs_three_clf_y_true, ToPs_three_clf_y_pred_prob)
print('Area under ROC Curve: {0:.3f}'.format(ToPs_three_clf_area_roc))
f.write('Area under ROC Curve: {0:.3f}\n'.format(ToPs_three_clf_area_roc))

ToPs_three_clf_time_taken = t_ToPs_3clf2 - t_ToPs_3clf1
print('Time taken: {0:.3f} hours'.format(ToPs_three_clf_time_taken/3600))
f.write('Time taken: {0:.3f} hours\n'.format(ToPs_three_clf_time_taken/3600))

ToPs_three_clf_confusion_matrix = confusion_matrix(ToPs_three_clf_y_true, ToPs_three_clf_prediction)
print('Confusion Matrix: \n', ToPs_three_clf_confusion_matrix)
f.write('Confusion Matrix: \n')
f.write(str(ToPs_three_clf_confusion_matrix))
f.write('\n')

ToPs_three_clf_classification_report = classification_report(ToPs_three_clf_y_true, ToPs_three_clf_prediction, target_names=class_names)
ToPs_three_clf_precision, ToPs_three_clf_recall, ToPs_three_clf_f1score, ToPs_three_clf_support = precision_recall_fscore_support(ToPs_three_clf_y_true, ToPs_three_clf_prediction, average=None)
print('Classification Report:')
print(ToPs_three_clf_classification_report) 
f.write('Classification Report:')
f.write(ToPs_three_clf_classification_report) 

# Output ToPs results to output.txt
f.write('\n\n----- ToPs classifier tree (3 classifiers) -----\n\n')
f.write(str(news_ToPs_three_clf.root_node))
f.write('\nMax Depth of Tree: {0}'.format(news_ToPs_three_clf.get_depth_of_tree()))
f.write('\n')



# Output predictions from all classifiers to a csv
prediction_df = pd.DataFrame(prediction_data)
prediction_df.to_csv('AllPredictions.csv', index=False)



####### PLOT COMPARATIVE FIGURES FOR EVALUATION METRICS #######

# ROC Curve
plt.figure()
# Random Forest
fpr, tpr, thresholds = roc_curve(news_y_test, rfc_prediction)
rfc_roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr,label='Random Forest AUC = {0:.2f}'.format(rfc_roc_auc), color='r', linestyle = '-.') 

# Extra Trees
fpr, tpr, thresholds = roc_curve(news_y_test, xtc_prediction)
xtc_roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr,label='Extra Tree AUC = {0:.2f}'.format(xtc_roc_auc), color='g', linestyle = '--') 

# AdaBoost
fpr, tpr, thresholds = roc_curve(news_y_test, ada_prediction)
ada_roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr,label='AdaBoost AUC = {0:.2f}'.format(ada_roc_auc), color='b', linestyle = ':') 

# ToPs Linear
fpr, tpr, thresholds = roc_curve(ToPs_linear_y_true, ToPs_linear_prediction)
ToPs_linear_roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr,label='ToPs Linear AUC = {0:.2f}'.format(ToPs_linear_roc_auc), color='c', linestyle = '--')

# ToPs 3 Classifiers 
fpr, tpr, thresholds = roc_curve(ToPs_three_clf_y_true, ToPs_three_clf_prediction)
ToPs_three_clf_roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr,label='ToPs 3 Classifiers AUC = {0:.2f}'.format(ToPs_three_clf_roc_auc), color='m', linestyle = ':')

# fpr, tpr, thresholds = roc_curve(news_y_test, ada_prediction)
# ToPs_three_clf_roc_auc = auc(fpr, tpr)
# plt.plot(fpr,tpr,label='ToPs 3 Classifiers AUC = {0:.2f}'.format(ToPs_three_clf_roc_auc), color='m', linestyle = ':')



plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')



# Bar Charts to compare metrics


# Area Under ROC
plt.figure()
index = np.arange(5)
# plt.bar(index, [rfc_area_roc, xtc_area_roc, ada_area_roc, ToPs_linear_area_roc, ada_area_roc], align='center')
plt.bar(index, [rfc_area_roc, xtc_area_roc, ada_area_roc, ToPs_linear_area_roc, ToPs_three_clf_area_roc], align='center')
plt.xticks(index, ('Random Forest', 'Extra Trees','AdaBoost', 'ToPs(linear)', 'ToPs (3 classifiers)'))
plt.ylabel('Area Under ROC')
plt.title('Area Under ROC Curve')
plt.savefig('area_roc.png')


# Accuracy
plt.figure()
index = np.arange(5)
# plt.bar(index, [rfc_accuracy, xtc_accuracy, ada_accuracy, ToPs_linear_accuracy, ada_accuracy], align='center')
plt.bar(index, [rfc_accuracy, xtc_accuracy, ada_accuracy, ToPs_linear_accuracy, ToPs_three_clf_accuracy], align='center')
plt.xticks(index, ('Random Forest', 'Extra Trees','AdaBoost', 'ToPs(linear)', 'ToPs (3 classifiers)'))
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.savefig('accuracy.png')

# Log Loss
plt.figure()
index = np.arange(5)
# plt.bar(index, [rfc_log_loss, xtc_log_loss, ada_log_loss, ToPs_linear_log_loss, ada_log_loss], align='center')
plt.bar(index, [rfc_log_loss, xtc_log_loss, ada_log_loss, ToPs_linear_log_loss, ToPs_three_clf_log_loss], align='center')
plt.xticks(index, ('Random Forest', 'Extra Trees','AdaBoost', 'ToPs(linear)', 'ToPs (3 classifiers)'))
plt.ylabel('Logarithmic Loss')
plt.title('Logarithmic Loss')
plt.savefig('log_loss.png')


# Time Taken
plt.figure()
index = np.arange(5)
# plt.bar(index, [rfc_time_taken, xtc_time_taken, ada_time_taken, ToPs_linear_time_taken, ada_time_taken], align='center')
plt.bar(index, [rfc_time_taken, xtc_time_taken, ada_time_taken, ToPs_linear_time_taken, ToPs_three_clf_time_taken], align='center')
plt.xticks(index, ('Random Forest', 'Extra Trees','AdaBoost', 'ToPs(linear)', 'ToPs (3 classifiers)'))
plt.ylabel('Run Time (seconds)')
plt.title('Run Time')
plt.savefig('runtime.png')




# Precision
plt.figure()
index = np.arange(2)
bar_width = 0.1
plt.bar(index, RF_precision, bar_width, color='r', label='Random Forest')
plt.bar(index+bar_width, XT_precision, bar_width, color='g', label='Extra Trees')
plt.bar(index+2*(bar_width), ADA_precision, bar_width, color='b', label = 'AdaBoost')
plt.bar(index+3*(bar_width), ToPs_linear_precision, bar_width, color='c', label = 'ToPs (linear)')
# plt.bar(index+4*(bar_width), ADA_precision, bar_width, color='m', label = 'ToPs (3 classifiers)') #- FIX THIS!!!
plt.bar(index+4*(bar_width), ToPs_three_clf_precision, bar_width, color='m', label = 'ToPs (3 classifiers)')
plt.xlabel('Class')
plt.ylabel('Precision Values')
plt.title('Precision')
plt.xticks(index + bar_width, ('Unpopular', 'Popular'))
plt.legend(loc='upper center')
plt.savefig('precision.png')


# Recall
plt.figure()
index = np.arange(2)
bar_width = 0.1
plt.bar(index, RF_recall, bar_width, color='r', label='Random Forest')
plt.bar(index+bar_width, XT_recall, bar_width, color='g', label='Extra Trees')
plt.bar(index+2*(bar_width), ADA_recall, bar_width, color='b', label = 'AdaBoost')
plt.bar(index+3*(bar_width), ToPs_linear_recall, bar_width, color='c', label = 'ToPs (linear)')
# plt.bar(index+4*(bar_width), ADA_recall, bar_width, color='m', label = 'ToPs (3 classifiers)') #- FIX THIS!!!
plt.bar(index+4*(bar_width), ToPs_three_clf_recall, bar_width, color='m', label = 'ToPs (3 classifiers)')
plt.xlabel('Class')
plt.ylabel('Recall Values')
plt.title('Recall')
plt.xticks(index + bar_width, ('Unpopular', 'Popular'))
plt.legend(loc='upper center')
plt.savefig('recall.png')


# F1-Score
plt.figure()
index = np.arange(2)
bar_width = 0.1
plt.bar(index, RF_f1score, bar_width, color='r', label='Random Forest')
plt.bar(index+bar_width, XT_f1score, bar_width, color='g', label='Extra Trees')
plt.bar(index+2*(bar_width), ADA_f1score, bar_width, color='b', label = 'AdaBoost')
plt.bar(index+3*(bar_width), ToPs_linear_f1score, bar_width, color='c', label = 'ToPs (linear)')
# plt.bar(index+4*(bar_width), ADA_f1score, bar_width, color='m', label = 'ToPs (3 classifiers)') #- FIX THIS!!!
plt.bar(index+4*(bar_width), ToPs_three_clf_f1score, bar_width, color='m', label = 'ToPs (3 classifiers)')
plt.xlabel('Class')
plt.ylabel('F1-Score Values')
plt.title('F1-Score')
plt.xticks(index + bar_width, ('Unpopular', 'Popular'))
plt.legend(loc='upper center')
plt.savefig('f1_score.png')







plt.show()












