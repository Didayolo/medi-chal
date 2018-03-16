

import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd


class AutoML():
	def __init__(self, input_dir="", basename="", test_size=None, verbose=False):
		if os.path.isdir(input_dir):
			self.input_dir = input_dir
		else:
			raise OSError('Input directory {} does not exist.'.format(input_dir))

		self.data = dict()
		if os.path.exists(os.path.join(input_dir, basename + '_train.data')):
			self.data['X_train'] = self.loadData(os.path.join(input_dir, basename + '_train.data'))
			self.data['y_train'] = self.loadLabel(os.path.join(input_dir, basename + '_train.solution'))
			self.data['X_test'] = self.loadData(os.path.join(input_dir, basename + '_test.data'))
			self.data['y_test'] = self.loadLabel(os.path.join(input_dir, basename + '_test.solution'))
		elif os.path.exists(os.path.join(input_dir, basename + '.data')):
			X = self.loadData(os.path.join(input_dir, basename + '.data'))
			y = self.loadLabel(os.path.join(input_dir, basename + '.solution'))
			if not test_size:
				test_size = 0.6
			self.data['X_train'], self.data['X_test'], self.data['y_train'], self.data['y_test'] = \
				train_test_split(X, y, test_size=test_size)
		else:
			raise OSError('No .data files in {}.'.format(input_dir))

	def loadData(self, filepath):
		return pd.read_csv(filepath, sep=' ', header=None).values

	def loadLabel(self, filepath):
		return pd.read_csv(filepath, sep=' ', header=None).values