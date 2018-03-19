import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd


class AutoML():
	def __init__(self, input_dir="", basename="", test_size=0, verbose=False): # test_size=None
		self.basename = basename
		if os.path.isdir(input_dir):
			self.input_dir = input_dir
		else:
			raise OSError('Input directory {} does not exist.'.format(input_dir))

		self.info = dict()
		self.init_info(os.path.join(self.input_dir, basename + '_public.info'))
		self.init_type(os.path.join(self.input_dir, basename + '_feat.type'))

		self.feat_name = []
		feat_name_file = os.path.join(input_dir, basename + '_feat.name')
		if os.path.exists(feat_name_file):
			self.feat_name = self.load_name(feat_name_file)
		
		self.label_name = []
		label_name_file = os.path.join(input_dir, basename + '_label.name')
		if os.path.exists(label_name_file):
			self.label_name = self.load_name(label_name_file)

		self.data = dict()
		if os.path.exists(os.path.join(input_dir, basename + '_train.data')):
			self.data['X_train'] = self.load_data(os.path.join(self.input_dir, basename + '_train.data'))
			self.data['y_train'] = self.load_label(os.path.join(self.input_dir, basename + '_train.solution'))
			self.data['X_test'] = self.load_data(os.path.join(self.input_dir, basename + '_test.data'))
			self.data['y_test'] = self.load_label(os.path.join(self.input_dir, basename + '_test.solution'))
		elif os.path.exists(os.path.join(input_dir, basename + '.data')):
			X = self.load_data(os.path.join(input_dir, basename + '.data'))
			y = self.load_label(os.path.join(input_dir, basename + '.solution'))
			#if not test_size:
			#	test_size = 0.6
			self.data['X_train'], self.data['X_test'], self.data['y_train'], self.data['y_test'] = \
				train_test_split(X, y, test_size=test_size)
		else:
			raise OSError('No .data files in {}.'.format(self.input_dir))

	def load_data(self, filepath):
		return pd.read_csv(filepath, sep=' ', header=None).values

	def load_label(self, filepath):
		return pd.read_csv(filepath, sep=' ', header=None).values
		
	def load_name(self, filepath):
		return pd.read_csv(filepath, sep='\n', header=None).values

	def init_type(self, filepath):
		if os.path.exists(filepath):
			self.info['feat_type'] = pd.read_csv(filepath, header=None).values.ravel()
		else:
			print('No features type file found.')
			self.info['feat_type'] = [self.info['feat_type']] * self.info['feat_num']

	def init_info(self, filepath):
		if os.path.exists(filepath):
			df = pd.read_csv(os.path.join(self.input_dir, self.basename + '_public.info'), header=None, sep='=').values
			for x in df:
				x[0] = x[0].replace("'", '').strip()
				x[1] = x[1].replace("'", '').strip()
			self.info = dict(zip(df[:, 0], df[:, 1]))
		else:
			print('No info file file found.')
			self.info['usage'] = 'No info file'
			self.info['name'] = self.basename
			self.info['has_categorical'] = 0
			self.info['has_missing'] = 0                            
			self.get_type_problem(os.path.join(self.input_dir, self.basename + '_train.solution'))
			# Finds the data format ('dense', 'sparse', or 'sparse_binary')   
			self.get_format_data(os.path.join(self.input_dir, self.basename + '_train.data'))
			
			if self.info['task']=='regression':
				self.info['metric'] = 'r2_metric'
			else:
				self.info['metric'] = 'auc_metric'     
			self.info['feat_type'] = 'Mixed'  

			self.get_nbr_features(
				os.path.join(self.input_dir, self.basename + '_train.data'), 
				os.path.join(self.input_dir, self.basename + '_test.data'), 
				os.path.join(self.input_dir, self.basename + '_valid.data'))

			self.info['time_budget'] = 600
		return self.info
		
	def get_data(self):
		''' Get data as a tuple (X_train, y_train, X_test, y_test) '''
		return self.data['X_train'], self.data['y_train'], self.data['X_test'], self.data['y_test']
		
	def get_data_as_df(self):
		''' Get data in pandas DataFrame format as a tuple (X_train, y_train, X_test, y_test) '''
		X_train, y_train, X_test, y_test = self.get_data()
		X_train_df = pd.DataFrame(X_train, columns=self.feat_name)
		y_train_df = pd.DataFrame(y_train, columns=self.label_name)
		X_test_df = pd.DataFrame(X_test, columns=self.feat_name)
		y_test_df = pd.DataFrame(y_test, columns=self.label_name)
		return X_train_df, y_train_df, X_test_df, y_test_df
		
	def get_info(self):
		return self.info

	def get_format_data(self,filename):
		''' Get the data format directly from the data file (in case we do not have an info file)'''
		self.info['format'] = 'dense'
		self.info['is_sparse'] = 0			
		return self.info['format']

	def get_nbr_features(self, *filenames):
		''' Get the number of features directly from the data file (in case we do not have an info file)'''
		if 'feat_num' not in self.info.keys():
			self.getFormatData(filenames[0])
			if self.info['format'] == 'dense':
				data = pd.read_csv(filenames[0], sep=' ', header=None)
				self.info['feat_num'] = data.shape[1]
		return self.info['feat_num']

	def get_type_problem(self, solution_filename):
		''' Get the type of problem directly from the solution file (in case we do not have an info file) '''
		if 'task' not in self.info.keys():
			solution = pd.read_csv(solution_filename, sep=' ', header=None).values
			target_num = solution.shape[1]
			self.info['target_num'] = target_num
			if target_num == 1: # if we have only one column
				solution = np.ravel(solution) # flatten
				nbr_unique_values = len(np.unique(solution))
				if nbr_unique_values < len(solution)/8:
					# Classification
					self.info['label_num'] = nbr_unique_values
					if nbr_unique_values == 2:
						self.info['task'] = 'binary.classification'
						self.info['target_type'] = 'Binary'
					else:
						self.info['task'] = 'multiclass.classification'
						self.info['target_type'] = 'Categorical'
				else:
					# Regression
					self.info['label_num'] = 0
					self.info['task'] = 'regression'
					self.info['target_type'] = 'Numerical'     
			else:
				# Multilabel or multiclass       
				self.info['label_num'] = target_num
				self.info['target_type'] = 'Binary' 
				if any(item > 1 for item in map(np.sum, solution.astype(int))):
					self.info['task'] = 'multilabel.classification'     
				else:
					self.info['task'] = 'multiclass.classification'        
		return self.info['task']
		
	def compute_descriptors(self):
		''' Compute descriptors or the dataset
			- 
			- 
		'''
		pass
		
