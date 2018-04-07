# Project - ECE657A  (Group __)
# Filza Mazahir 20295951  &  Tarneem Barayyan 20645942 

# Libraries used: pandas, numpy, scikit-learn, matplotlib

# Algorithm oriented project on Data Classification
# Data Source: https://archive.ics.uci.edu/ml/datasets/online+news+popularity

# Import Libraries 
import pandas as pd
import numpy as np

from sklearn import preprocessing
from scipy.stats import zscore

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import roc_curve, roc_auc_score
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
news_df = news_df.reset_index(drop=True)

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


# Select important features
clf = RandomForestClassifier(n_estimators=50, max_depth=8)
clf.fit(news_x_train, news_y_train) 
model = SelectFromModel(clf, prefit=True, threshold=0.02)
important_features = model.get_support()

# Drop the unimportant features
feature_names = news_x.columns.values
for i, feature_importance in enumerate(important_features):
	if feature_importance == False:
		news_x_train = news_x_train.drop([feature_names[i]], axis = 1)
		news_x_test = news_x_test.drop([feature_names[i]], axis = 1)

print('\nNumber of features selcted: {0}'.format(np.sum(important_features)))
print('Feature Selected\n', news_x_train.columns.values)
print('\n')

f.write('\n\nNumber of features selcted: {0}'.format(np.sum(important_features)))
f.write('\nFeature Selected\n')
f.write(str(news_x_train.columns.values))
f.write('\n')



# Save evaluation metrics for all runs in metrics_all_runs dict
metrics_all_runs = {
'log_loss':{},
'accuracy': {},
'area_roc': {},
'training_time': {},
'precision_unpopular': {},
'precision_popular': {},
'recall_unpopular': {},
'recall_popular':{},
'f1score_unpopular' : {},
'f1score_popular': {}
}


# Function to add metrics to the nested metric_all_runs dict
def add_metric(metric_name, predictor_name, value):
	if predictor_name not in metrics_all_runs[metric_name]:
		metrics_all_runs[metric_name][predictor_name] = [value]
	else:
		metrics_all_runs[metric_name][predictor_name].append(value)


# Loop to the run the experiment 20 times 
# Edited to 2 experiments for submission so code finishes in 20 minutes, but ran it for 20 times for the report
for i in range(2):
	print('\n----------------------------')
	print('Experiment No.:  {0}\n'.format(i+1))
	f.write('\n\n----------------------------')
	f.write('\n\nExperiment No.:  {0}\n'.format(i+1))

	# Output results in a csv file
	prediction_data = {}
	prediction_data['y_true_rfc_xtc_ada'] = news_y_test.reset_index(drop=True).as_matrix()


	# kNN CLASSIFIER - BASELINE METHOD
	print('K NEAREST NEIGHBORS CLASSIFIER - BASELINE METHOD')
	f.write('\nK NEAREST NEIGHBORS CLASSIFIER - BASELINE METHOD\n')

	t_kNN1 = time.time()

	kNN_clf = KNeighborsClassifier()
	kNN_clf.fit(news_x_train, news_y_train)
	kNN_prediction = kNN_clf.predict(news_x_test)
	kNN_predict_prob = kNN_clf.predict_proba(news_x_test) # Predict class probabilities

	prediction_data['kNN_pred_prob'] = kNN_predict_prob[:, 1]
	prediction_data['kNN_pred'] = kNN_prediction


	t_kNN2 = time.time()

	# Calculate metrics and save to metrics_all_runs dict using add_metric funtion
	kNN_accuracy = accuracy_score(news_y_test, kNN_prediction)
	add_metric('accuracy', 'kNN', kNN_accuracy)
	print('Accuracy: {0:.3f}'.format(kNN_accuracy))
	f.write('Accuracy: {0:.3f}\n'.format(kNN_accuracy))

	kNN_log_loss = log_loss(news_y_test, kNN_predict_prob[:, 1])
	add_metric('log_loss', 'kNN', kNN_log_loss)
	print('Logarithmic Loss: {0:.3f}'.format(kNN_log_loss))
	f.write('Logarithmic Loss: {0:.3f}\n'.format(kNN_log_loss))

	kNN_area_roc = roc_auc_score(news_y_test, kNN_predict_prob[:, 1])
	add_metric('area_roc', 'kNN', kNN_area_roc)
	print('Area under ROC Curve: {0:.3f}'.format(kNN_area_roc))
	f.write('Area under ROC Curve: {0:.3f}\n'.format(kNN_area_roc))

	kNN_time_taken = t_kNN2 - t_kNN1
	add_metric('training_time', 'kNN', kNN_time_taken)
	print('Time taken: {0:.3f} seconds'.format(kNN_time_taken))
	f.write('Time taken: {0:.3f} seconds\n'.format(kNN_time_taken))

	kNN_confusion_matrix = confusion_matrix(news_y_test, kNN_prediction)
	print('Confusion Matrix: \n', kNN_confusion_matrix)
	f.write('Confusion Matrix: \n')
	f.write(str(kNN_confusion_matrix))
	f.write('\n')


	kNN_classification_report = classification_report(news_y_test, kNN_prediction, target_names=class_names)
	kNN_precision, kNN_recall, kNN_f1score, kNN_support = precision_recall_fscore_support(news_y_test, kNN_prediction, average=None)

	add_metric('precision_unpopular', 'kNN', kNN_precision[0])
	add_metric('precision_popular', 'kNN', kNN_precision[1])

	add_metric('recall_unpopular', 'kNN', kNN_recall[0])
	add_metric('recall_popular', 'kNN', kNN_recall[1])

	add_metric('f1score_unpopular', 'kNN', kNN_f1score[0])
	add_metric('f1score_popular', 'kNN', kNN_f1score[1])

	print('Classification Report:')
	print(kNN_classification_report)

	f.write('Classification Report:\n')
	f.write(kNN_classification_report)


	print('\n')
	f.write('\n')



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

	# Calculate metrics and save to metrics_all_runs dict using add_metric funtion
	rfc_accuracy = accuracy_score(news_y_test, rfc_prediction)
	add_metric('accuracy', 'rfc', rfc_accuracy)
	print('Accuracy: {0:.3f}'.format(rfc_accuracy))
	f.write('Accuracy: {0:.3f}\n'.format(rfc_accuracy))

	rfc_log_loss = log_loss(news_y_test, rfc_predict_prob[:, 1])
	add_metric('log_loss', 'rfc', rfc_log_loss)
	print('Logarithmic Loss: {0:.3f}'.format(rfc_log_loss))
	f.write('Logarithmic Loss: {0:.3f}\n'.format(rfc_log_loss))

	rfc_area_roc = roc_auc_score(news_y_test, rfc_predict_prob[:, 1])
	add_metric('area_roc', 'rfc', rfc_area_roc)
	print('Area under ROC Curve: {0:.3f}'.format(rfc_area_roc))
	f.write('Area under ROC Curve: {0:.3f}\n'.format(rfc_area_roc))

	rfc_time_taken = t_rf2 - t_rf1
	add_metric('training_time', 'rfc', rfc_time_taken)
	print('Time taken: {0:.3f} seconds'.format(rfc_time_taken))
	f.write('Time taken: {0:.3f} seconds\n'.format(rfc_time_taken))

	rfc_confusion_matrix = confusion_matrix(news_y_test, rfc_prediction)
	print('Confusion Matrix: \n', rfc_confusion_matrix)
	f.write('Confusion Matrix: \n')
	f.write(str(rfc_confusion_matrix))
	f.write('\n')


	rfc_classification_report = classification_report(news_y_test, rfc_prediction, target_names=class_names)
	RF_precision, RF_recall, RF_f1score, RF_support = precision_recall_fscore_support(news_y_test, rfc_prediction, average=None)

	add_metric('precision_unpopular', 'rfc', RF_precision[0])
	add_metric('precision_popular', 'rfc', RF_precision[1])

	add_metric('recall_unpopular', 'rfc', RF_recall[0])
	add_metric('recall_popular', 'rfc', RF_recall[1])

	add_metric('f1score_unpopular', 'rfc', RF_f1score[0])
	add_metric('f1score_popular', 'rfc', RF_f1score[1])


	print('Classification Report:')
	print(rfc_classification_report)

	f.write('Classification Report:\n')
	f.write(rfc_classification_report)


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

	# Calculate metrics and save to metrics_all_runs dict using add_metric funtion
	xtc_accuracy = accuracy_score(news_y_test, xtc_prediction)
	add_metric('accuracy', 'xtc', xtc_accuracy)
	print('Accuracy: {0:.3f}'.format(xtc_accuracy))
	f.write('Accuracy: {0:.3f}\n'.format(xtc_accuracy))

	xtc_log_loss = log_loss(news_y_test, xtc_predict_prob[:, 1])
	add_metric('log_loss', 'xtc', xtc_log_loss)
	print('Logarithmic Loss: {0:.3f}'.format(xtc_log_loss))
	f.write('Logarithmic Loss: {0:.3f}\n'.format(xtc_log_loss))

	xtc_area_roc = roc_auc_score(news_y_test, xtc_predict_prob[:, 1])
	add_metric('area_roc', 'xtc', xtc_area_roc)
	print('Area under ROC Curve: {0:.3f}'.format(xtc_area_roc))
	f.write('Area under ROC Curve: {0:.3f}\n'.format(xtc_area_roc))

	xtc_time_taken = t_xt2 - t_xt1
	add_metric('training_time', 'xtc', xtc_time_taken)
	print('Time taken: {0:.3f} seconds'.format(xtc_time_taken))
	f.write('Time taken: {0:.3f} seconds\n'.format(xtc_time_taken))

	xtc_confusion_matrix = confusion_matrix(news_y_test, xtc_prediction)
	print('Confusion Matrix: \n', xtc_confusion_matrix)
	f.write('Confusion Matrix: \n')
	f.write(str(xtc_confusion_matrix))
	f.write('\n')

	xtc_classification_report = classification_report(news_y_test, xtc_prediction, target_names=class_names)
	XT_precision, XT_recall, XT_f1score, XT_support = precision_recall_fscore_support(news_y_test, xtc_prediction, average=None)

	add_metric('precision_unpopular', 'xtc', XT_precision[0])
	add_metric('precision_popular', 'xtc', XT_precision[1])

	add_metric('recall_unpopular', 'xtc', XT_recall[0])
	add_metric('recall_popular', 'xtc', XT_recall[1])

	add_metric('f1score_unpopular', 'xtc', XT_f1score[0])
	add_metric('f1score_popular', 'xtc', XT_f1score[1])

	print('Classification Report:')
	print(xtc_classification_report)
	f.write('Classification Report:\n')
	f.write(xtc_classification_report)

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

	# Calculate metrics and save to metrics_all_runs dict using add_metric funtion
	ada_accuracy = accuracy_score(news_y_test, ada_prediction)
	add_metric('accuracy', 'ada', ada_accuracy)
	print('Accuracy: {0:.3f}'.format(ada_accuracy))
	f.write('Accuracy: {0:.3f}\n'.format(ada_accuracy))

	ada_log_loss = log_loss(news_y_test, ada_predict_prob[:, 1])
	add_metric('log_loss', 'ada', ada_log_loss)
	print('Logarithmic Loss: {0:.3f}'.format(ada_log_loss))
	f.write('Logarithmic Loss: {0:.3f}\n'.format(ada_log_loss))

	ada_area_roc = roc_auc_score(news_y_test, ada_predict_prob[:, 1])
	add_metric('area_roc', 'ada', ada_area_roc)
	print('Area under ROC Curve: {0:.3f}'.format(ada_area_roc))
	f.write('Area under ROC Curve: {0:.3f}\n'.format(ada_area_roc))

	ada_time_taken = t_ada2 - t_ada1
	add_metric('training_time', 'ada', ada_time_taken)
	print('Time taken: {0:.3f} seconds'.format(ada_time_taken))
	f.write('Time taken: {0:.3f} seconds\n'.format(ada_time_taken))

	ada_confusion_matrix = confusion_matrix(news_y_test, ada_prediction)
	print('Confusion Matrix: \n', ada_confusion_matrix)
	f.write('Confusion Matrix: \n')
	f.write(str(ada_confusion_matrix))
	f.write('\n')

	ada_classification_report = classification_report(news_y_test, ada_prediction, target_names=class_names)
	ADA_precision, ADA_recall, ADA_f1score, ADA_support = precision_recall_fscore_support(news_y_test, ada_prediction, average=None)

	add_metric('precision_unpopular', 'ada', ADA_precision[0])
	add_metric('precision_popular', 'ada', ADA_precision[1])

	add_metric('recall_unpopular', 'ada', ADA_recall[0])
	add_metric('recall_popular', 'ada', ADA_recall[1])

	add_metric('f1score_unpopular', 'ada', ADA_f1score[0])
	add_metric('f1score_popular', 'ada', ADA_f1score[1])

	print('Classification Report:')
	print(ada_classification_report)
	f.write('Classification Report:\n')
	f.write(ada_classification_report)

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

	ToPs_linear_prediction = ToPs_linear_y_pred_prob.apply(lambda x: 1 if x >=0.5 else 0)
	prediction_data['ToPs_linear_pred'] = ToPs_linear_prediction.as_matrix()

	t_ToPs_l2 = time.time()


	ToPs_linear_accuracy = accuracy_score(ToPs_linear_y_true, ToPs_linear_prediction) 
	add_metric('accuracy', 'ToPs_linear', ToPs_linear_accuracy)
	print('Accuracy: {0:.3f}'.format(ToPs_linear_accuracy))
	f.write('Accuracy: {0:.3f}\n'.format(ToPs_linear_accuracy))

	ToPs_linear_log_loss = log_loss(ToPs_linear_y_true, ToPs_linear_y_pred_prob)
	add_metric('log_loss', 'ToPs_linear', ToPs_linear_log_loss)
	print('Logarithmic Loss: {0:.3f}'.format(ToPs_linear_log_loss))
	f.write('Logarithmic Loss: {0:.3f}\n'.format(ToPs_linear_log_loss))

	ToPs_linear_area_roc = roc_auc_score(ToPs_linear_y_true, ToPs_linear_y_pred_prob)
	add_metric('area_roc', 'ToPs_linear', ToPs_linear_area_roc)
	print('Area under ROC Curve: {0:.3f}'.format(ToPs_linear_area_roc))
	f.write('Area under ROC Curve: {0:.3f}\n'.format(ToPs_linear_area_roc))

	ToPs_linear_time_taken = t_ToPs_l2 - t_ToPs_l1
	add_metric('training_time', 'ToPs_linear', ToPs_linear_time_taken)
	print('Time taken: {0:.3f} seconds'.format(ToPs_linear_time_taken))
	f.write('Time taken: {0:.3f} seconds\n'.format(ToPs_linear_time_taken))

	ToPs_linear_confusion_matrix = confusion_matrix(ToPs_linear_y_true, ToPs_linear_prediction)
	print('Confusion Matrix: \n', ToPs_linear_confusion_matrix)
	f.write('Confusion Matrix: \n')
	f.write(str(ToPs_linear_confusion_matrix))
	f.write('\n')

	ToPs_linear_classification_report = classification_report(ToPs_linear_y_true, ToPs_linear_prediction, target_names=class_names)
	ToPs_linear_precision, ToPs_linear_recall, ToPs_linear_f1score, ToPs_linear_support = precision_recall_fscore_support(ToPs_linear_y_true, ToPs_linear_prediction, average=None)

	add_metric('precision_unpopular', 'ToPs_linear', ToPs_linear_precision[0])
	add_metric('precision_popular', 'ToPs_linear', ToPs_linear_precision[1])

	add_metric('recall_unpopular', 'ToPs_linear', ToPs_linear_recall[0])
	add_metric('recall_popular', 'ToPs_linear', ToPs_linear_recall[1])

	add_metric('f1score_unpopular', 'ToPs_linear', ToPs_linear_f1score[0])
	add_metric('f1score_popular', 'ToPs_linear', ToPs_linear_f1score[1])

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




	print('TREES OF PREDICTORS - 3 CLASSIFIERS (RandomForest ExtraTrees & AdaBoost)')
	f.write('\nTREES OF PREDICTORS - 3 CLASSIFIERS (RandomForest ExtraTrees & AdaBoost)\n')

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
	add_metric('accuracy', 'ToPs_three_clf', ToPs_three_clf_accuracy)
	print('Accuracy: {0:.3f}'.format(ToPs_three_clf_accuracy))
	f.write('Accuracy: {0:.3f}\n'.format(ToPs_three_clf_accuracy))

	ToPs_three_clf_log_loss = log_loss(ToPs_three_clf_y_true, ToPs_three_clf_y_pred_prob)
	add_metric('log_loss', 'ToPs_three_clf', ToPs_three_clf_log_loss)
	print('Logarithmic Loss: {0:.3f}'.format(ToPs_three_clf_log_loss))
	f.write('Logarithmic Loss: {0:.3f}\n'.format(ToPs_three_clf_log_loss))

	ToPs_three_clf_area_roc = roc_auc_score(ToPs_three_clf_y_true, ToPs_three_clf_y_pred_prob)
	add_metric('area_roc', 'ToPs_three_clf', ToPs_three_clf_area_roc)
	print('Area under ROC Curve: {0:.3f}'.format(ToPs_three_clf_area_roc))
	f.write('Area under ROC Curve: {0:.3f}\n'.format(ToPs_three_clf_area_roc))

	ToPs_three_clf_time_taken = t_ToPs_3clf2 - t_ToPs_3clf1
	add_metric('training_time', 'ToPs_three_clf', ToPs_three_clf_time_taken)
	print('Time taken: {0:.3f} mins'.format(ToPs_three_clf_time_taken/60))
	f.write('Time taken: {0:.3f} mins\n'.format(ToPs_three_clf_time_taken/60))

	ToPs_three_clf_confusion_matrix = confusion_matrix(ToPs_three_clf_y_true, ToPs_three_clf_prediction)
	print('Confusion Matrix: \n', ToPs_three_clf_confusion_matrix)
	f.write('Confusion Matrix: \n')
	f.write(str(ToPs_three_clf_confusion_matrix))
	f.write('\n')

	ToPs_three_clf_classification_report = classification_report(ToPs_three_clf_y_true, ToPs_three_clf_prediction, target_names=class_names)
	ToPs_three_clf_precision, ToPs_three_clf_recall, ToPs_three_clf_f1score, ToPs_three_clf_support = precision_recall_fscore_support(ToPs_three_clf_y_true, ToPs_three_clf_prediction, average=None)

	add_metric('precision_unpopular', 'ToPs_three_clf', ToPs_three_clf_precision[0])
	add_metric('precision_popular', 'ToPs_three_clf', ToPs_three_clf_precision[1])

	add_metric('recall_unpopular', 'ToPs_three_clf', ToPs_three_clf_recall[0])
	add_metric('recall_popular', 'ToPs_three_clf', ToPs_three_clf_recall[1])

	add_metric('f1score_unpopular', 'ToPs_three_clf', ToPs_three_clf_f1score[0])
	add_metric('f1score_popular', 'ToPs_three_clf', ToPs_three_clf_f1score[1])


	print('Classification Report:')
	print(ToPs_three_clf_classification_report) 
	f.write('Classification Report:\n')
	f.write(ToPs_three_clf_classification_report) 

	# Output ToPs results to output.txt
	f.write('\n\n----- ToPs classifier tree (3 classifiers) -----\n\n')
	f.write(str(news_ToPs_three_clf.root_node))
	f.write('\nMax Depth of Tree: {0}'.format(news_ToPs_three_clf.get_depth_of_tree()))
	f.write('\n')



	# Output predictions from all classifiers to a csv
	prediction_df = pd.DataFrame(prediction_data)
	prediction_df.to_csv('AllPredictions.csv', index=False)


# Calculate mean and std from all runs
metric_mean = {}
metric_std = {}
csv_heading = ""
mean_csv_values = ""
std_csv_values = ""
for metric_name, metric_dict in metrics_all_runs.items():
	metric_mean[metric_name] = {}
	metric_std[metric_name] = {}
	for classifier_name, metric_values in metric_dict.items():
		mean_val = np.mean(metric_values)
		std_val = np.std(metric_values)
		metric_mean[metric_name][classifier_name] = mean_val
		metric_std[metric_name][classifier_name] = std_val
		csv_heading += metric_name + " - " + classifier_name + ","
		mean_csv_values += str(mean_val) + ","
		std_csv_values += str(std_val) + ","


metrics_file_handle = open('metric_means.csv', 'w')
metrics_file_handle.write(csv_heading + "\n")
metrics_file_handle.write(mean_csv_values + "\n")
metrics_file_handle.close()

metrics_file_handle = open('metric_stds.csv', 'w')
metrics_file_handle.write(csv_heading + "\n")
metrics_file_handle.write(std_csv_values + "\n")
metrics_file_handle.close()

####### PLOT COMPARATIVE FIGURES FOR EVALUATION METRICS #######


# ROC Curve
plt.figure()

fpr, tpr, thresholds = roc_curve(news_y_test, rfc_predict_prob[:, 1])
plt.plot(fpr,tpr,label='Random Forest AUC = {0:.2f}'.format(metric_mean['area_roc']['rfc']), color='r', linestyle = '-.') 

fpr, tpr, thresholds = roc_curve(news_y_test, xtc_predict_prob[:, 1])
plt.plot(fpr,tpr,label='Extra Tree AUC = {0:.2f}'.format(metric_mean['area_roc']['xtc']), color='g', linestyle = '--') 

fpr, tpr, thresholds = roc_curve(news_y_test, ada_predict_prob[:, 1])
plt.plot(fpr,tpr,label='AdaBoost AUC = {0:.2f}'.format(metric_mean['area_roc']['ada']), color='b', linestyle = ':') 

fpr, tpr, thresholds = roc_curve(ToPs_linear_y_true, ToPs_linear_y_pred_prob)
plt.plot(fpr,tpr,label='ToPs Linear AUC = {0:.2f}'.format(metric_mean['area_roc']['ToPs_linear']), color='c', linestyle = '--')

fpr, tpr, thresholds = roc_curve(ToPs_three_clf_y_true, ToPs_three_clf_y_pred_prob)
plt.plot(fpr,tpr,label='ToPs 3 Classifiers AUC = {0:.2f}'.format(metric_mean['area_roc']['ToPs_three_clf']), color='m', linestyle = ':')

plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('fig_roc_curve.png')



# Bar Charts to compare metrics


# Area Under ROC
plt.figure()
index = np.arange(5)
plt.bar(index, [metric_mean['area_roc']['rfc'], metric_mean['area_roc']['xtc'], metric_mean['area_roc']['ada'], metric_mean['area_roc']['ToPs_linear'], metric_mean['area_roc']['ToPs_three_clf']], align='center')
plt.xticks(index, ('Random Forest', 'Extra Trees','AdaBoost', 'ToPs(linear)', 'ToPs (3 classifiers)'))
plt.ylabel('Area Under ROC')
plt.title('Area Under ROC Curve')
plt.savefig('fig_area_roc.png')



# Accuracy
plt.figure()
index = np.arange(5)
plt.bar(index, [metric_mean['accuracy']['rfc'], metric_mean['accuracy']['xtc'], metric_mean['accuracy']['ada'], metric_mean['accuracy']['ToPs_linear'], metric_mean['accuracy']['ToPs_three_clf']], align='center')
plt.xticks(index, ('Random Forest', 'Extra Trees','AdaBoost', 'ToPs(linear)', 'ToPs (3 classifiers)'))
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.savefig('fig_accuracy.png')



# Log Loss
plt.figure()
index = np.arange(5)
plt.bar(index, [metric_mean['log_loss']['rfc'], metric_mean['log_loss']['xtc'], metric_mean['log_loss']['ada'], metric_mean['log_loss']['ToPs_linear'], metric_mean['log_loss']['ToPs_three_clf']], align='center')
plt.xticks(index, ('Random Forest', 'Extra Trees','AdaBoost', 'ToPs(linear)', 'ToPs (3 classifiers)'))
plt.ylabel('Logarithmic Loss')
plt.title('Logarithmic Loss')
plt.savefig('fig_log_loss.png')



# Time Taken
plt.figure()
index = np.arange(5)
plt.bar(index, [metric_mean['training_time']['rfc'], metric_mean['training_time']['xtc'], metric_mean['training_time']['ada'], metric_mean['training_time']['ToPs_linear'], metric_mean['training_time']['ToPs_three_clf']], align='center')
plt.xticks(index, ('Random Forest', 'Extra Trees','AdaBoost', 'ToPs(linear)', 'ToPs (3 classifiers)'))
plt.ylabel('Run Time (seconds)')
plt.title('Run Time')
plt.savefig('fig_runtime.png')



# Precision
plt.figure()
index = np.arange(2)
bar_width = 0.1
plt.bar(index, [metric_mean['precision_unpopular']['rfc'], metric_mean['precision_popular']['rfc']], bar_width, color='r', label='Random Forest')
plt.bar(index+bar_width, [metric_mean['precision_unpopular']['xtc'], metric_mean['precision_popular']['xtc']], bar_width, color='g', label='Extra Trees')
plt.bar(index+2*(bar_width), [metric_mean['precision_unpopular']['ada'], metric_mean['precision_popular']['ada']], bar_width, color='b', label = 'AdaBoost')
plt.bar(index+3*(bar_width), [metric_mean['precision_unpopular']['ToPs_linear'], metric_mean['precision_popular']['ToPs_linear']], bar_width, color='c', label = 'ToPs (linear)')
plt.bar(index+4*(bar_width), [metric_mean['precision_unpopular']['ToPs_three_clf'], metric_mean['precision_popular']['ToPs_three_clf']], bar_width, color='m', label = 'ToPs (3 classifiers)')
plt.xlabel('Class')
plt.ylabel('Precision Values')
plt.title('Precision')
plt.xticks(index + bar_width, ('Unpopular', 'Popular'))
plt.legend(loc='upper center')
plt.savefig('fig_precision.png')


# Recall
plt.figure()
index = np.arange(2)
bar_width = 0.1
plt.bar(index, [metric_mean['recall_unpopular']['rfc'], metric_mean['recall_popular']['rfc']], bar_width, color='r', label='Random Forest')
plt.bar(index+bar_width, [metric_mean['recall_unpopular']['xtc'], metric_mean['recall_popular']['xtc']], bar_width, color='g', label='Extra Trees')
plt.bar(index+2*(bar_width), [metric_mean['recall_unpopular']['ada'], metric_mean['recall_popular']['ada']], bar_width, color='b', label = 'AdaBoost')
plt.bar(index+3*(bar_width), [metric_mean['recall_unpopular']['ToPs_linear'], metric_mean['recall_popular']['ToPs_linear']], bar_width, color='c', label = 'ToPs (linear)')
plt.bar(index+4*(bar_width), [metric_mean['recall_unpopular']['ToPs_three_clf'], metric_mean['recall_popular']['ToPs_three_clf']], bar_width, color='m', label = 'ToPs (3 classifiers)')
plt.xlabel('Class')
plt.ylabel('Recall Values')
plt.title('Recall')
plt.xticks(index + bar_width, ('Unpopular', 'Popular'))
plt.legend(loc='upper center')
plt.savefig('fig_recall.png')





# F1-Score
plt.figure()
index = np.arange(2)
bar_width = 0.1
plt.bar(index, [metric_mean['f1score_unpopular']['rfc'], metric_mean['f1score_popular']['rfc']], bar_width, color='r', label='Random Forest')
plt.bar(index+bar_width, [metric_mean['f1score_unpopular']['xtc'], metric_mean['f1score_popular']['xtc']], bar_width, color='g', label='Extra Trees')
plt.bar(index+2*(bar_width), [metric_mean['f1score_unpopular']['ada'], metric_mean['f1score_popular']['ada']], bar_width, color='b', label = 'AdaBoost')
plt.bar(index+3*(bar_width), [metric_mean['f1score_unpopular']['ToPs_linear'], metric_mean['f1score_popular']['ToPs_linear']], bar_width, color='c', label = 'ToPs (linear)')
plt.bar(index+4*(bar_width), [metric_mean['f1score_unpopular']['ToPs_three_clf'], metric_mean['f1score_popular']['ToPs_three_clf']], bar_width, color='m', label = 'ToPs (3 classifiers)')
plt.xlabel('Class')
plt.ylabel('F1-Score Values')
plt.title('F1-Score')
plt.xticks(index + bar_width, ('Unpopular', 'Popular'))
plt.legend(loc='upper center')
plt.savefig('fig_f1_score.png')






f.close()
plt.show()












