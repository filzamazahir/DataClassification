# Assignment 2 - ECE657A (Group 21)
# Filza Mazahir 20295951  &  Tarneem Barayyan 20645942 

# Libraries used: pandas, numpy, scikit-learn, matplotlib


# Import Libraries 
import pandas as pd
import numpy as np
from math import pow
from numpy import genfromtxt
from sklearn.metrics import roc_curve, auc
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import pyplot
import scipy.io
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from time import time


#converting .mat to .csv
#data = scipy.io.loadmat("DataDNA.mat")

#for i in data:
#	if '__' not in i and 'readme' not in i:
		
#		np.savetxt(("DNAData"+i+".csv"),data[i],fmt='%s',delimiter=',')

#importing fea and gnd as array:
fea = genfromtxt('DNADatafea.csv', delimiter=',')
gnd = genfromtxt('DNADatagnd.csv')


# 2.1 DATA PREPROCESSING
#findig missing data
#print(sum(fea.isnull().sum()))  #output is 0

#visuallizing all features
#fea.iloc[:, 0:25].hist()
#fea.iloc[:, 26:57].hist()
#plt.show()
#plt.title(' Distribution of labels (gnd)')
#plt.hist(gnd)
#plt.show()

#Scalling dataset using min-max normalization
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
fea_min_max = scaler.fit_transform(fea)
# Scalling dataset using Z-score method
fea_zscore= stats.zscore(fea) #saved as array

plt.title(' min-max normalization of feature 27th')
plt.xlabel('categories values')
plt.ylabel('data points')
#plt.hist(fea_min_max[:, 27])
#plt.show()
plt.title(' z-score of feature 27th')
plt.xlabel('categories values')
plt.ylabel('data points')
#plt.hist(fea_zscore[:, 27])
#plt.show()

#for Question 2, first splitting data into test and train sets:
fea_train, fea_test, gnd_train, gnd_test = train_test_split(fea_zscore, gnd, test_size=0.50, random_state=42)
# for next part, kfold will be used on train sets, the test set will be used for final comparison for all classifier
# 2.2-a Knn:
#k values
kf = KFold(n_splits=5)
k= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] 
for i, val in enumerate(k):
	for j, (train, test) in enumerate(kf.split(fea_train, gnd_train)): 
		clf_knn = KNeighborsClassifier(n_neighbors=k[i]) 
		clf_knn.fit(fea_zscore[train], gnd[train]) 
		pred = clf_knn.predict(fea_zscore[test]) 
		clf_knn_accuracy = accuracy_score(gnd[test], pred)
	#print('Accuracy of k= %(k[i])d :'% {"k[i]": k[i]},clf_knn_accuracy.mean())
	#plt.plot(k[i], int(clf_knn_accuracy*100), 'ro') 
plt.title('Relationship Between Accuracy and K Values')
plt.xlabel('K values')
plt.ylabel('Accuracy')
#plt.show()

# 2.2-b SVM
C = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
Sigma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
gamma = [0, 0, 0, 0, 0, 0, 0, 0]
#calculating gamma:
for i in range (8):
	gamma[i] = 1/(2*pow(Sigma[i],2))


kf = KFold(n_splits=5)
j = 0
for i in range (8): 
	for k, (train, test) in enumerate(kf.split(fea_train, gnd_train)): 
		clf_SVM = svm.SVC(kernel='rbf', C=C[i], gamma=gamma[7]) 
		clf_SVM.fit(fea_zscore[train], gnd[train]) 
		pred = clf_SVM.predict(fea_zscore[test]) 
		clf_SVM_accuracy = accuracy_score(gnd[test], pred) 
		#ROC:
		fpr, tpr, thresholds = roc_curve(gnd[test], pred)
		accuracy_mean = clf_SVM_accuracy.mean() 
	#print('Accuracy of gamma %.2f and C %.2f:' %(gamma[7], C[i]),accuracy_mean)
	roc_auc = auc(fpr, tpr)
	#plt.plot(fpr,tpr,label='ROC fold %d (AUC = %0.2f)' % (j, roc_auc)) 
	j += 1
plt.title('  Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
#plt.show()

#for part 3, we will use test set
# 3-a
# for knn, using k= 13 and cross validation of 5 fold:
for i, (train, test) in enumerate(kf.split(fea_test, gnd_test)): 
		clf_knn = KNeighborsClassifier(n_neighbors=13) 
		clf_knn.fit(fea_test[train], gnd_test[train]) 
		pred = clf_knn.predict(fea_test[test]) 
		clf_knn_accuracy = accuracy_score(gnd_test[test], pred)
print('Accuracy of Knn {0:.3f} ({1:.3f})'.format(clf_knn_accuracy.mean(), clf_knn_accuracy.std()))
# for SVM with Sigma:10 (gamma:0.01), and C:20		 
for k, (train, test) in enumerate(kf.split(fea_test, gnd_test)): 
		clf_SVM = svm.SVC(kernel='rbf', C=20, gamma=0.01) 
		clf_SVM.fit(fea_test[train], gnd_test[train]) 
		pred = clf_SVM.predict(fea_test[test]) 
		clf_SVM_accuracy = accuracy_score(gnd_test[test], pred)
#print('Accuracy of SVM {0:.3f} ({1:.3f})'.format(clf_SVM_accuracy.mean(), clf_SVM_accuracy.std()))
#neural network:
from sklearn.preprocessing import StandardScaler
mlp = MLPClassifier()
for k, (train, test) in enumerate(kf.split(fea_test, gnd_test)):
	mlp.fit(fea_test[train], gnd_test[train])
	mpl_pred = mlp.predict(fea_test[test])
	mpl_accuracy = accuracy_score(gnd_test[test], pred) 
#print('MPL Defult Accuracy {0:.3f} ({1:.3f})'.format(mpl_accuracy.mean(), mpl_accuracy.std()))
# Random Forest:
clf_RF = RandomForestClassifier()
for k, (train, test) in enumerate(kf.split(fea_test, gnd_test)):
	clf_RF.fit(fea_test[train], gnd_test[train])
	clf_RF_pred = clf_RF.predict(fea_test[test])
	clf_RF_accuracy = accuracy_score(gnd_test[test], pred)
#print('RF Defult Accuracy {0:.3f} ({1:.3f})'.format(clf_RF_accuracy.mean(), clf_RF_accuracy.std()))

#3-b
class_names= ['-1', '+1']
#will be exploring parameters that do best for MPL and RF using the same dataset in part 3-a
from sklearn.preprocessing import StandardScaler
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), solver = 'lbfgs', learning_rate_init=0.01)
for k, (train, test) in enumerate(kf.split(fea_test, gnd_test)):
	mlp.fit(fea_test[train], gnd[train])
	mpl_pred = mlp.predict(fea_test[test])
	mpl_accuracy = accuracy_score(gnd_test[test], pred) 
	clf_mpl_report = classification_report(gnd_test[test], mpl_pred, target_names= class_names)
#print('MPL Accuracy {0:.3f} ({1:.3f})'.format(mpl_accuracy.mean(), mpl_accuracy.std()))
#print(clf_mpl_report)
# Random Forest:
clf_RF = RandomForestClassifier(n_estimators=10, max_depth=3, max_features='sqrt',  random_state=42)
for k, (train, test) in enumerate(kf.split(fea_test, gnd_test)): 
	clf_RF.fit(fea_test[train], gnd[train])
	clf_RF_pred = clf_RF.predict(fea_test[test])
	clf_RF_accuracy = accuracy_score(gnd_test[test], pred)
	clf_RF_report = classification_report(gnd_test[test], clf_RF_pred, target_names= class_names)
#print('RF  Accuracy {0:.3f} ({1:.3f})'.format(clf_RF_accuracy.mean(), clf_RF_accuracy.std()))
#print(clf_report)

# 3-c using all fea and gnd and k_fold =20
#knn, k=13:
t0 = time()
kf = KFold(n_splits=20)
for i, (train, test) in enumerate(kf.split(fea_zscore, gnd)): 
		clf_knn = KNeighborsClassifier(n_neighbors=13) 
		clf_knn.fit(fea_zscore[train], gnd[train]) 
		clf_knn_pred = clf_knn.predict(fea_zscore[test]) 
		clf_knn_accuracy = accuracy_score(gnd[test], clf_knn_pred)
		clf_knn_report = classification_report(gnd[test], clf_knn_pred, target_names= class_names)
#print('Accuracy of Knn {0:.3f} ({1:.3f})'.format(clf_knn_accuracy.mean(), clf_knn_accuracy.std()))
#print(clf_knn_report)
#print ("training time:", round(time()-t0, 3), "s")
#SVM with Sigma:10 (gamma:0.01), and C:20	
t0 = time()	 
for k, (train, test) in enumerate(kf.split(fea_zscore, gnd)): 
		clf_SVM = svm.SVC(kernel='rbf', C=20, gamma=0.01) 
		clf_SVM.fit(fea_zscore[train], gnd[train]) 
		clf_SVM_pred = clf_SVM.predict(fea_zscore[test]) 
		clf_SVM_accuracy = accuracy_score(gnd[test], clf_SVM_pred)
		clf_SVM_report = classification_report(gnd[test], clf_SVM_pred, target_names= class_names)
#print('Accuracy of SVM {0:.3f} ({1:.3f})'.format(clf_SVM_accuracy.mean(), clf_SVM_accuracy.std()))
#print(clf_SVM_report)
#print ("training time:", round(time()-t0, 3), "s")
#neural network:
t0 = time()
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), solver = 'lbfgs', learning_rate_init=0.01)
for k, (train, test) in enumerate(kf.split(fea_zscore, gnd)):
	mlp.fit(fea_zscore[train], gnd[train])
	mpl_pred = mlp.predict(fea_zscore[test])
	mpl_accuracy = accuracy_score(gnd[test], mpl_pred) 
	clf_mpl_report = classification_report(gnd[test], mpl_pred, target_names= class_names)
#print('MPL Accuracy {0:.3f} ({1:.3f})'.format(mpl_accuracy.mean(), mpl_accuracy.std()))
#print(clf_mpl_report)
#print ("training time:", round(time()-t0, 3), "s")

# Random Forest:
t0 = time()
clf_RF = RandomForestClassifier()
for k, (train, test) in enumerate(kf.split(fea_zscore, gnd)): 
	clf_RF.fit(fea_zscore[train], gnd[train])
	clf_RF_pred = clf_RF.predict(fea_zscore[test])
	clf_RF_accuracy = accuracy_score(gnd[test], clf_RF_pred)
	clf_RF_report = classification_report(gnd[test], clf_RF_pred, target_names= class_names)
print('RF  Accuracy {0:.3f} ({1:.3f})'.format(clf_RF_accuracy.mean(), clf_RF_accuracy.std()))
print(clf_RF_report)
print ("training time:", round(time()-t0, 3), "s")