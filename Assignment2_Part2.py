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

# 2.2-a Knn:
#k values
kf = KFold(n_splits=5)
k= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] 
for i, val in enumerate(k):
	for j, (train, test) in enumerate(kf.split(fea_zscore, gnd)): 
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
	for k, (train, test) in enumerate(kf.split(fea_zscore, gnd)): 
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

# 3-a
# for knn, using k= 13 and cross validation of 5 fold:
kf = KFold(n_splits=5)
for i, (train, test) in enumerate(kf.split(fea_zscore, gnd)): 
		clf_knn = KNeighborsClassifier(n_neighbors=13) 
		clf_knn.fit(fea_zscore[train], gnd[train]) 
		pred = clf_knn.predict(fea_zscore[test]) 
		clf_knn_accuracy = accuracy_score(gnd[test], pred)
		print('Accuracy of Knn %.2f' %(clf_knn_accuracy.mean(), clf_knn_accuracy.std()))
		plt.plot(k[i], int(clf_knn_accuracy*100)) 

plt.show()	
#neural network:
from sklearn.preprocessing import StandardScaler
mlp = MLPClassifier()
for k, (train, test) in enumerate(kf.split(fea_zscore, gnd)):
	mlp.fit(fea_zscore[train], gnd[train])
	mpl_pred = mlp.predict(fea_zscore[test])
	mpl_accuracy = accuracy_score(gnd[test], pred) 
	accuracy_mean = mpl_accuracy.mean() 
	print('MPL Defult Accuracy:',accuracy_mean)
	

