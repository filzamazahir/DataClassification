# Trees of Predictors (ToPs) ensemble learning method
# Source: https://arxiv.org/abs/1706.01396v2

# Project - ECE657A  (Group 21)  Filza Mazahir 20295951  &  Tarneem Barayyan 20645942 

# Import Libraries 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import log_loss
from scipy import optimize

from math import inf



# Global function - get classifier instance based on a given string
def get_classifier_instance(name):
	if name == 'RandomForest':
		classifier = RandomForestClassifier(n_estimators=50, max_depth=8)
	elif name == 'ExtraTrees':
		classifier = ExtraTreesClassifier(n_estimators=50, max_depth=8)
	elif name == 'AdaBoost':
		classifier = AdaBoostClassifier(n_estimators=50)
	elif name == 'LinearSGD':
		classifier = linear_model.SGDClassifier(loss='log', max_iter=200, tol=0.001)

	return classifier





######################################################################
# ToPs class - Initialize it with data and a set of classifiers to use
class ToPs:
	def __init__(self, data_x, data_y, x_test, y_test, classifiers):
		# Get dataset passed
		self.data_x = data_x
		self.data_y = data_y

		self.x_test = x_test
		self.y_test = y_test

		self.classifiers = classifiers

		self.root_node = None

		# Get column names of all columns, and the ones with just binary values 
		self.column_names = self.data_x.columns.values
		self.binary_columns = set()

		for column in self.data_x.columns.values:
			if len(self.data_x[column].unique()) == 2:
				self.binary_columns.add(column)

		
		# Scale all data, and then split data into training and validation sets here
		self.x_train, self.y_train, self.x_validate1, self.y_validate1, self.x_validate2, self.y_validate2 = self.split_data_into_train_validate()

		# Construct Root Node of the tree
		self.construct_root_node() # sets self.root_node to a Node instance
		return


	# Dataset split - Training - 50%, Validation 1: 15%, Validation 2: 15%, Test: 20%
	# From the given training set - 37.5% Validation, 62.5% Training
	def split_data_into_train_validate(self):
		
		# Split data_x and data_y to get training and validation set
		x_train, x_validate, y_train, y_validate = train_test_split(self.data_x, self.data_y, test_size=0.375, stratify=self.data_y)

		# Split the validation set into two
		x_validate1, x_validate2, y_validate1, y_validate2 = train_test_split(x_validate, y_validate, test_size=0.5, stratify=y_validate)

		return(x_train, y_train, x_validate1, y_validate1, x_validate2, y_validate2)


	# Function to create root node from the given dataset
	def construct_root_node(self):
		loss_values_list = []
		predictors = []
		# print('Root Node')

		for clf in self.classifiers:
			clf_root = get_classifier_instance(clf)
			clf_root.fit(self.x_train, self.y_train) # Train the classifier of the root node on the full dataset
			clf_root_y_pred_prob = clf_root.predict_proba(self.x_validate1)
			loss_on_validation1 = log_loss(self.y_validate1, clf_root_y_pred_prob, normalize= False, labels = [0,1])

			predictors.append(clf_root)
			loss_values_list.append(loss_on_validation1)

		predictor_index = loss_values_list.index(min(loss_values_list))

		self.root_node = Node(self.x_train, self.y_train, self.x_validate1, self.y_validate1, self.x_validate2, self.y_validate2, self.x_test, self.y_test, loss_on_validation1, predictors[predictor_index], 0)
		self.root_node.predictor_name = self.classifiers[predictor_index]

		return




	# Algorithm 1 - Figure out what the feature_to_split and threshold with minimum loss, then assign children based on that split
	def create_sub_tree(self, node, max_depth):

		threshold_binary = [0.5]
		threshold_continous = np.arange(0.1, 1.0, 0.1)

		if node.current_depth >= max_depth:
			return
		minimum_loss_so_far = inf
		# minimum_loss_so_far =float('inf')
		feature_at_min_loss = None
		threshold_at_min_loss = None
		children_nodes_at_min_loss = None

		# Iterate through all the features
		for feature in self.column_names:
			threshold_range = threshold_binary if feature in self.binary_columns else threshold_continous

			# Iterate through the range of thresholds
			for threshold in threshold_range:
				threshold = round(threshold, 1)

				children_nodes = self.split_node(node, feature, threshold)
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
					children_nodes_at_min_loss = children_nodes

		if minimum_loss_so_far < node.loss_validate1:

			# Assign threshold and feature_to_split, children attribute of this node
			node.feature_to_split = feature_at_min_loss
			node.threshold = threshold_at_min_loss

			node.right = children_nodes_at_min_loss[0]
			node.left = children_nodes_at_min_loss[1]

			# print('Creating children for node {0}'.format(node))  

			# Assign children of this node based on the min loss 
			self.create_sub_tree(node.right, max_depth)
			self.create_sub_tree(node.left, max_depth)
	
		return


	# Algorithm 1 - Split a node based on a given feature and threshold- split available dataset too. 
	def split_node(self, node, feature, threshold):

		# Split the training data
		left_x_train = node.x_train[node.x_train[feature] < threshold]
		left_train_indices = left_x_train.index.values
		left_y_train = node.y_train.loc[left_train_indices]

		right_x_train = node.x_train[node.x_train[feature] >= threshold]
		right_train_indices = right_x_train.index.values
		right_y_train = node.y_train.loc[right_train_indices]

		# Split the validation 1 data
		left_x_validate1 = node.x_validate1[node.x_validate1[feature] < threshold]
		left_validate1_indices = left_x_validate1.index.values
		left_y_validate1 = node.y_validate1.loc[left_validate1_indices]

		right_x_validate1 = node.x_validate1[node.x_validate1[feature] >= threshold]
		right_validate1_indices = right_x_validate1.index.values
		right_y_validate1 = node.y_validate1.loc[right_validate1_indices]

		# Split the validation 2 data
		left_x_validate2 = node.x_validate2[node.x_validate2[feature] < threshold]
		left_validate2_indices = left_x_validate2.index.values
		left_y_validate2 = node.y_validate2.loc[left_validate2_indices]

		right_x_validate2 = node.x_validate2[node.x_validate2[feature] >= threshold]
		right_validate2_indices = right_x_validate2.index.values
		right_y_validate2 = node.y_validate2.loc[right_validate2_indices]

		# Split the test data
		left_x_test = node.x_test[node.x_test[feature] < threshold]
		left_test_indices = left_x_test.index.values
		left_y_test = node.y_test.loc[left_test_indices]

		right_x_test = node.x_test[node.x_test[feature] >= threshold]
		right_test_indices = right_x_test.index.values
		right_y_test = node.y_test.loc[right_test_indices]
		

		# If data is too skewed one way and cannot be split 
		if len(right_x_train) == 0 or len(left_x_train) == 0 or len(right_x_validate1) == 0 or len(left_x_validate1) == 0 or len(right_y_train.unique()) == 1 or len(left_y_train.unique()) == 1:
			return None

		# Train classifier on the left data
		# min_loss_left = inf
		min_loss_left= float('inf')
		clf_left_at_min_loss = None
		clf_left_name_at_min_loss = None

		for classifier_name in self.classifiers:		
			clf_left = get_classifier_instance(classifier_name)
			clf_left.fit(left_x_train, left_y_train) 
			clf_left_y_predict_prob = clf_left.predict_proba(left_x_validate1)
			log_loss_validation1_left = log_loss(left_y_validate1, clf_left_y_predict_prob, normalize = False, labels = [0,1])
			
			if log_loss_validation1_left < min_loss_left:
				min_loss_left = log_loss_validation1_left
				clf_left_at_min_loss = clf_left
				
			clf_left_name_at_min_loss = classifier_name


		# Train classifier on the right data
		min_loss_right = inf
		# min_loss_right = float('inf')
		clf_right_at_min_loss = None
		clf_right_name_at_min_loss = None

		for classifier_name in self.classifiers:	
			clf_right = get_classifier_instance(classifier_name)
			clf_right.fit(right_x_train, right_y_train)
			clf_right_y_predict_prob = clf_right.predict_proba(right_x_validate1)
			log_loss_validation1_right = log_loss(right_y_validate1, clf_right_y_predict_prob, normalize=False, labels = [0,1])

			if log_loss_validation1_right < min_loss_right:
				min_loss_right = log_loss_validation1_right
				clf_right_at_min_loss = clf_right

			clf_right_name_at_min_loss = classifier_name

		# Create nodes based on this split, and return
		left_node = Node(left_x_train, left_y_train, left_x_validate1, left_y_validate1, left_x_validate2, left_y_validate2, left_x_test, left_y_test, log_loss_validation1_left, clf_left_at_min_loss, node.current_depth+1)
		right_node = Node(right_x_train, right_y_train, right_x_validate1, right_y_validate1, right_x_validate2, right_y_validate2, right_x_test, right_y_test, log_loss_validation1_right, clf_right_at_min_loss, node.current_depth+1)
		
		left_node.predictor_name = clf_left_name_at_min_loss
		right_node.predictor_name = clf_right_name_at_min_loss

		return(right_node, left_node)
	



	# Algorithm 2 - Get predictors on path from root to leaf, then optimize weights for all predictors on path
	def add_weights_to_predictors_on_path(self, node, predictors_on_path):

		# If leaf node - create a dataframe with predicted values from all predictors on the path
		if node.left==None and node.right==None:
			all_predictors_on_path = predictors_on_path + [node.predictor] 
			node.leaf_all_predictors_on_path = all_predictors_on_path


			if len(node.x_validate2) > 0 and len(node.y_validate2.unique()) == 2:
				# Loop through all predictors on path from root to leaf, and add predictions to data dict
				data = {}
				for i, clf in enumerate(all_predictors_on_path):
					y_pred_proba_validate2 = clf.predict_proba(node.x_validate2)
					y_pred_proba_validate2 = y_pred_proba_validate2[:, 1]
					data["classifier " + str(i)] = y_pred_proba_validate2

				# Create dataframe from the data dict
				all_pred_proba_on_path_df = pd.DataFrame(data)

				# Local Log Loss functions with weights for optimize function
				def log_loss_with_weights(weights):
					y_pred_proba = np.matmul(all_pred_proba_on_path_df.as_matrix(), weights)
					return log_loss(node.y_validate2, y_pred_proba)

				# Local function to check if some of weights is 1
				def is_sum_one(weights):
					return np.sum(weights) - 1.0


				# Create a matrix of initial weights, weight bounds, and constraints_dict for optimize function
				initial_weights = np.array([1.0/len(all_pred_proba_on_path_df.columns) for i in range(len(all_pred_proba_on_path_df.columns))])
				weight_bounds = np.array([[0.0, 1.0] for i in range(len(all_pred_proba_on_path_df.columns))])
				constraints_dict = {"fun": is_sum_one, "type": "eq"}

				optimization_result = optimize.minimize(log_loss_with_weights, initial_weights, method="SLSQP", constraints=constraints_dict, bounds=weight_bounds)
				node.leaf_optimized_weights = optimization_result.x


		# Continue traversing through tree and add predictors until leaf node
		else:
			if node.left:
				self.add_weights_to_predictors_on_path(node.left, predictors_on_path + [node.predictor])
			if node.right:
				self.add_weights_to_predictors_on_path(node.right, predictors_on_path + [node.predictor])

	

	# Algorithm 3 - Traverse the tree and use test values to predict y probability
	def _predict_traverse_tree(self, node):

		# If leaf node - create a dataframe with predicted values from all predictors on the path
		if node.left==None and node.right==None:

			# Loop through all predictors on path from root to leaf, and add predictions to data dict
			data = {}
			for i, clf in enumerate(node.leaf_all_predictors_on_path):
				if len(node.x_test) > 0:
					y_pred_test = clf.predict_proba(node.x_test)
					y_pred_test = y_pred_test[:, 1]
				else:
					y_pred_test = []

				data["classifier " + str(i)] = y_pred_test

			# Create dataframe from the data dict
			all_pred_proba_on_path_test_df = pd.DataFrame(data)

			# Get final prediction probability with optimized weights
			if node.leaf_optimized_weights.size > 0:
				y_pred_prob_final = np.matmul(all_pred_proba_on_path_test_df.as_matrix(), node.leaf_optimized_weights)

			# Assign equal weights if optimized weights don't exist (if len(x_validate2) = 0 or y_validate2 didn't have unique values)
			else:
				y_pred_prob_final = all_pred_proba_on_path_test_df.sum(axis=1)/len(all_pred_proba_on_path_test_df.columns)
			
			return(node.y_test, pd.DataFrame(y_pred_prob_final))	

		# Continue traversing through tree until leaf node
		else:
			if node.left:
				y_test_left, y_pred_prob_final_left = self._predict_traverse_tree(node.left)
			if node.right:
				y_test_right, y_pred_prob_final_right = self._predict_traverse_tree(node.right)

			# Combine the test and predict values for the leafs at the parent
			y_test_parent = pd.concat([y_test_left, y_test_right])
			y_pred_prob_final_parent = pd.concat([y_pred_prob_final_left, y_pred_prob_final_right])

			# print('Parent - Predictions for {0} values'.format(len(y_pred_prob_final_parent)))
			
			return (y_test_parent, y_pred_prob_final_parent)




	# Algorithm 1 and 2 Wrapper - Create a Tree Of Predictors
	def create_tree(self, maxdepth):

		self.create_sub_tree(self.root_node, maxdepth) # Algorithm 1
		self.add_weights_to_predictors_on_path(self.root_node, []) # Algorithm 2

		return




	# Algorithm 3 - Wrapper - Get probability of y_predict done on the test method
	def predict_proba(self): 

		y_true, y_pred_proba = self._predict_traverse_tree(self.root_node)

		# Reset index
		y_true = y_true.reset_index(drop=True)
		y_pred_proba = y_pred_proba.reset_index(drop=True)

		return y_true, y_pred_proba[0]



	# Helper function - Calculating loss values of all leaf nodes
	def _loss_validation1_of_all_leaf_nodes(self, node):
		sum_loss_value = 0

		if node.left == None and node.right == None:
			return node.loss_validate1
		else:
			sum_loss_value += self._loss_validation1_of_all_leaf_nodes(node.left)
			sum_loss_value += self._loss_validation1_of_all_leaf_nodes(node.right)

		return sum_loss_value


	# Helper function wrapper - Calculating loss values of all leaf nodes
	def loss_validation1_of_all_leaf_nodes(self):
		return self._loss_validation1_of_all_leaf_nodes(self.root_node)



	# Helper function - Get depth of tree
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



	# Helper function - Get depth of tree
	def get_depth_of_tree(self):
		return self._get_depth_of_tree(self.root_node)-1  # -1 because root is included







##################################################################
# Node class
class Node:
	def __init__(self, x_train, y_train, x_validate1, y_validate1, x_validate2, y_validate2, x_test, y_test, loss_validate1, predictor, current_depth):

		# Data available to the particular node (root node will have full data set)
		self.x_train = x_train 
		self.y_train = y_train
		self.x_validate1 = x_validate1
		self.y_validate1 = y_validate1
		self.x_validate2 = x_validate2
		self.y_validate2 = y_validate2
		self.x_test = x_test
		self.y_test = y_test

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

		# For leaf node - all predictors on the path from root to leaf, and optimized weights
		self.leaf_all_predictors_on_path = None
		self.leaf_optimized_weights = np.array([])


	def __str__(self):
		prefix = '   '*self.current_depth
		string_to_print = prefix + 'Current Depth: ' + str(self.current_depth) + '\n'
		string_to_print += prefix + 'Feature to split: ' + str(self.feature_to_split) + '\n'
		string_to_print += prefix + 'Threshold: ' + str(self.threshold) +'\n'
		string_to_print += prefix + 'Predictor Name: ' + str(self.predictor_name) + '\n'
		string_to_print += prefix + 'Log Loss (Validation 1): ' + str(self.loss_validate1) + '\n'
		if self.leaf_optimized_weights.size > 0:
			string_to_print += prefix + 'Optimized Weights: ' + str(self.leaf_optimized_weights) + '\n'
		if self.left == None:
			string_to_print += prefix + 'Left Child: ' + str(self.left) + '\n'
		else:
			string_to_print += prefix + 'Left Child: \n' + str(self.left) + '\n'

		if self.right == None:
			string_to_print += prefix + 'Right Child: ' + str(self.right) + '\n'
		else:
			string_to_print += prefix + 'Right Child: \n' + str(self.right) + '\n'


		return string_to_print
	




