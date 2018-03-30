# Trees of Predictors (ToPs) ensemble learning method
# Source: https://arxiv.org/abs/1706.01396v2

# Project - ECE657A  (Group 21)  Filza Mazahir 20295951  &  Tarneem Barayyan 20645942 

# Import Libraries 
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.dummy import DummyClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import log_loss

from math import inf
import time

# Open an output text file to output the Tree
f = open('output.txt', 'w')


# LOAD DATA
file_location = 'OnlineNewsPopularity.csv'
news_df_original = pd.read_csv(file_location, sep=', ', engine='python')

# Drop non-predictive attributes
news_df = news_df_original.drop(['url', 'timedelta'], axis = 1)


# Getting dataset ready for training
news_y = news_df['shares']
news_y = news_y.apply(lambda x: 1 if x >=1400 else 0)

news_x = news_df.drop(['shares'], axis = 1)
class_names = ['Unpopular (<1400)', 'Popular (>=1400)']





# Global function - get classifier instance based on a given string
def get_classifier_instance(name):
	if name == 'RandomForest':
		classifier = RandomForestClassifier()
	elif name == 'ExtraTrees':
		classifier = ExtraTreesClassifier()
	elif name == 'AdaBoost':
		classifier = AdaBoostClassifier()
	elif name == 'LinearSGD':
		classifier = linear_model.SGDClassifier(loss='log', max_iter=10, tol=0.001)

	return classifier




######################################################################
# ToPs class - Initialize it with data and a set of classifiers to use
class ToPs:
	def __init__(self, data_x, data_y, classifiers):
		# Get dataset passed
		self.data_x = data_x
		self.data_y = data_y

		self.classifiers = classifiers

		self.root_node = None

		# Get column names of all columns, and the ones with just binary values - should go in ToPs class eventually
		self.column_names = self.data_x.columns.values
		self.binary_columns = set()

		for column in self.data_x.columns.values:
			if len(self.data_x[column].unique()) == 2:
				self.binary_columns.add(column)

		
		# Split data here
		self.x_train, self.y_train, self.x_validate1, self.y_validate1, self.x_validate2, self.y_validate2, self.x_test, self.y_test = self.split_data_into_test_train_validate()

		return

	def split_data_into_test_train_validate(self):
		# Standardization of the data
		# zscore = preprocessing.StandardScaler().fit_transform(news_x)
		# news_x = pd.DataFrame(zscore, columns=list(news_x.columns.values)) #convert to nice table 
		minmax = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(self.data_x)
		news_x = pd.DataFrame(minmax, columns=list(self.data_x.columns.values))  


		# Dataset split - Training - 50%, Validation 1: 15%, Validation 2: 15%, Test: 20%
		# Dataset x split: x_train, x_validate1, x_validate2, x_test
		# Dataset y split: y_train, y_validate1, y_validate2, y_test
		# Split dataset into test set - 20% for testing, rest for training and validation
		x_rest, x_test, y_rest, y_test = train_test_split(self.data_x, self.data_y, test_size=0.20, stratify=self.data_y)

		# Split again to get training and validation
		x_train, x_validate, y_train, y_validate = train_test_split(x_rest, y_rest, test_size=0.375, stratify=y_rest)

		# Split the validation set into two
		x_validate1, x_validate2, y_validate1, y_validate2 = train_test_split(x_validate, y_validate, test_size=0.5, stratify=y_validate)

		return(x_train, y_train, x_validate1, y_validate1, x_validate2, y_validate2, x_test, y_test)


	# Function to create root node from the given dataset
	def construct_root_node(self):
		# clf_root = linear_model.SGDClassifier(loss='log')
		loss_values_list = [0 for i in range(len(self.classifiers))]
		predictors = [0 for i in range(len(self.classifiers))]
		print('Root Node')
		for clf in self.classifiers:
			clf_root = get_classifier_instance(clf)
			clf_root.fit(self.x_train, self.y_train) # Pass dataset into this function
			clf_root_y_prediction = clf_root.predict(self.x_validate1)
			loss_on_validation1 = log_loss(self.y_validate1, clf_root_y_prediction, normalize= False, labels = [0,1])

			predictors.append(clf_root)
			loss_values_list.append(loss_on_validation1)

		predictor_index = loss_values_list.index(min(loss_values_list))


		self.root_node = Node(self, self.x_train, self.y_train, self.x_validate1, self.y_validate1, self.x_validate2, self.y_validate2, loss_on_validation1, predictors[predictor_index], 0)
		return
		
	def create_tree(self, maxdepth):
		self.construct_root_node() #Node Instance
		self.root_node.create_sub_tree(maxdepth)

		return

	def _get_depth_of_tree(self, node):
		current_depth = 0
		depth_left = 0
		depth_right = 0

		if node.left:
			depth_left = self._get_depth_of_tree(node.left)
		elif node.right: 
			depth_right = self._get_depth_of_tree(node.right)

		current_depth = max(depth_left, depth_right)+1

		return current_depth

	def get_depth_of_tree(self):
		return self._get_depth_of_tree(self.root_node)-1  # -1 because root is included








##################################################################
# Node class
class Node:
	def __init__(self, tree, x_train, y_train, x_validate1, y_validate1, x_validate2, y_validate2, loss_validate1, predictor, current_depth):
		self.tree = tree

		# Data available to the particular node (root node will have full data set)
		self.x_train = x_train 
		self.y_train = y_train
		self.x_validate1 = x_validate1
		self.y_validate1 = y_validate1
		self.x_validate2 = x_validate2
		self.y_validate2 = y_validate2

		# Set upon initialization
		self.loss_validate1 = loss_validate1
		self.predictor = predictor # Instance of a classifier predictor
		self.current_depth = current_depth

		# Children - Node Instances
		self.left = None 
		self.right = None 

		# Set in the create_subtree function
		self.feature_to_split = None
		self.threshold = None
		self.predictor_name = None

		# Set in add_weight function
		self.loss_validate2 = None

		# # Weight of each node (Algorithm 2)
		# self.weight = None


	def __str__(self):
		# string_to_print = 'Predictor: ' + str(self.predictor) + '\n'
		prefix = '   '*self.current_depth
		string_to_print = prefix + 'Current Depth: ' + str(self.current_depth) + '\n'
		string_to_print += prefix + 'Feature to split: ' + str(self.feature_to_split) + '\n'
		string_to_print += prefix + 'Threshold: ' + str(self.threshold) +'\n'
		string_to_print += prefix + 'Predictor: ' + str(self.predictor_name) + '\n'
		string_to_print += prefix + 'Log Loss (Validation 1): ' + str(self.loss_validate1) + '\n'
		if self.left == None:
			string_to_print += prefix + 'Left Child: ' + str(self.left) + '\n'
		else:
			string_to_print += prefix + 'Left Child: \n' + str(self.left) + '\n'

		if self.right == None:
			string_to_print += prefix + 'Right Child: ' + str(self.right) + '\n'
		else:
			string_to_print += prefix + 'Right Child: \n' + str(self.right) + '\n'

		# string_to_print += 'Features available: ' + str(self.training_data_x.columns.values) + '\n'

		return string_to_print
	


	# Split a node based on a given feature and threshold:
	def split_node(self, feature, threshold, classifier):
		# print('Feature: {0}, Threshold: {1:.1f}, Classifier: {2}'.format(feature, threshold, classifier))

		# Split the training data
		left_x_train = self.x_train[self.x_train[feature] < threshold]
		left_train_indices = left_x_train.index.values
		left_y_train = self.y_train.loc[left_train_indices]

		right_x_train = self.x_train[self.x_train[feature] >= threshold]
		right_train_indices = right_x_train.index.values
		right_y_train = self.y_train.loc[right_train_indices]

		# Split the validation 1 data
		left_x_validate1 = self.x_validate1[self.x_validate1[feature] < threshold]
		left_validate1_indices = left_x_validate1.index.values
		left_y_validate1 = self.y_validate1.loc[left_validate1_indices]

		right_x_validate1 = self.x_validate1[self.x_validate1[feature] >= threshold]
		right_validate1_indices = right_x_validate1.index.values
		right_y_validate1 = self.y_validate1.loc[right_validate1_indices]

		# Split the validation 2 data
		left_x_validate2 = self.x_validate2[self.x_validate2[feature] < threshold]
		left_validate2_indices = left_x_validate2.index.values
		left_y_validate2 = self.y_validate2.loc[left_validate2_indices]

		right_x_validate2 = self.x_validate2[self.x_validate2[feature] >= threshold]
		right_validate2_indices = right_x_validate2.index.values
		right_y_validate2 = self.y_validate2.loc[right_validate2_indices]

		

		# If data is too skewed one way and cannot be split 
		if len(right_x_train) == 0 or len(left_x_train) == 0 or len(right_x_validate1) == 0 or len(left_x_validate1) == 0:
			return None

		# Train classifier on the right data
		if len(right_y_train.unique()) == 1:
			clf_right = DummyClassifier()
		else:
			clf_right = get_classifier_instance(classifier)
		clf_right.fit(right_x_train, right_y_train)
		clf_right_y_prediction = clf_right.predict(right_x_validate1)
		log_loss_validation1_right = log_loss(right_y_validate1, clf_right_y_prediction, normalize=False, labels = [0,1])

		# Train classifier on the left data
		if len(left_y_train.unique()) == 1:
			clf_left = DummyClassifier()
		else:
			clf_left = get_classifier_instance(classifier)
		clf_left.fit(left_x_train, left_y_train) 
		clf_left_y_prediction = clf_left.predict(left_x_validate1)
		log_loss_validation1_left = log_loss(left_y_validate1, clf_left_y_prediction, normalize = False, labels = [0,1])
		
		# Create nodes based on this split, and return
		right_node = Node(self.tree, right_x_train, right_y_train, right_x_validate1, right_y_validate1, right_x_validate2, right_y_validate2, log_loss_validation1_right, clf_right, self.current_depth+1)
		left_node = Node(self.tree, left_x_train, left_y_train, left_x_validate1, left_y_validate1, left_x_validate2, left_y_validate2, log_loss_validation1_left, clf_left, self.current_depth+1)
	
		return(right_node, left_node)



	# Create a sub_tree (ToPs)
	# Figure out what the feature_to_split and threshold with minimum loss, then assign children based on that split
	def create_sub_tree(self, max_depth):
		print('Current Depth:', self.current_depth)
		threshold_binary = [0.5]
		threshold_continous = np.arange(0.1, 1.0, 0.1)

		if self.current_depth >= max_depth:
			print('Depth reached - no need to split further')
			return
		minimum_loss_so_far = inf
		feature_at_min_loss = None
		threshold_at_min_loss = None
		classifier_name_at_min_loss = None
		children_nodes_at_min_loss = None


		for feature in self.tree.column_names:
			threshold_range = threshold_binary if feature in self.tree.binary_columns else threshold_continous

			for threshold in threshold_range:
				threshold = round(threshold, 1)

				for classifier_name in self.tree.classifiers:

					children_nodes = self.split_node(feature, threshold, classifier_name)
					if children_nodes == None:
						continue

					right_node = children_nodes[0]
					left_node = children_nodes[1]

					# Compare log loss values of parent and children
					children_log_loss_validate1 = (right_node.loss_validate1 + left_node.loss_validate1)
					if children_log_loss_validate1 < minimum_loss_so_far:
						minimum_loss_so_far = children_log_loss_validate1
						feature_at_min_loss = feature
						threshold_at_min_loss = threshold
						classifier_name_at_min_loss = classifier_name
						children_nodes_at_min_loss = children_nodes


		if minimum_loss_so_far >= self.loss_validate1:
			print('END FUNCTION - All Children loss values bigger!!', minimum_loss_so_far)


		else: 
			print('---Parent loss bigger', self.loss_validate1) # we want this
			print('---All Children losses (smaller): ', minimum_loss_so_far)
			print('---Feature with this small loss:', feature_at_min_loss)

			# Assign threshold and feature_to_split, children attribute of this node
			self.feature_to_split = feature_at_min_loss
			self.threshold = threshold_at_min_loss
			self.predictor_name = classifier_name_at_min_loss

			self.right = children_nodes_at_min_loss[0]
			self.left = children_nodes_at_min_loss[1]

			# Assign children of this node based on the min loss 
			self.right.create_sub_tree(max_depth)
			self.left.create_sub_tree(max_depth)
	
		return

		# Calculating loss values of all leaf nodes
	def loss_validation1_of_all_leaf_nodes(self):
		sum_loss_value = 0

		if self.left == None and self.right == None:
			return self.loss_validate1
		else:
			sum_loss_value += self.left.loss_validation1_of_all_leaf_nodes()
			sum_loss_value += self.right.loss_validation1_of_all_leaf_nodes()

		return sum_loss_value
	


	# Algorithm 2 - Adding weights on the Path
	def add_weights_on_path(self):


		return



#######################################################
# Main function - outside all classes


# Test functions here - construct tree from news dataset
t0 = time.time()

# news_ToPs = Tops(news_x, news_y, ['RandomForest', 'ExtraTrees', 'AdaBoost'])
# news_ToPs = Tops(news_x, news_y, ['LinearSGD', 'RandomForest'])
news_ToPs = ToPs(news_x, news_y, ['LinearSGD'])
news_ToPs.create_tree(3)
loss_on_leafs = news_ToPs.root_node.loss_validation1_of_all_leaf_nodes()
depth_of_tree = news_ToPs.get_depth_of_tree()

print(news_ToPs.root_node)
f.write(str(news_ToPs.root_node))

print('\nLoss values of all leafs {0}'.format(loss_on_leafs))
f.write('\nLoss values of all leafs {0}'.format(loss_on_leafs))

print('\nMax Depth of Tree: {0}'.format(depth_of_tree))
f.write('\nMax Depth of Tree: {0}'.format(depth_of_tree))
		
t1 = time.time()

print('\nTotal Time taken: {0:.1f} mins'.format((t1-t0)/60))
f.write('\nTotal Time taken: {0:.1f} mins'.format((t1-t0)/60))

















