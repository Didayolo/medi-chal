import pandas as pd

def standard(df, column, mean=None, std=None):
	x = df.copy()
	if not mean and not std:
		mean = x[column].mean()
		std = x[column].std()
	x[column] = (x[column] - mean) / std
	return x, (mean, std)

def min_max(df, column, mini=None, maxi=None):
	x = df.copy()
	if not mini and not maxi:
		mini = x[column].min()
		maxi = x[column].max()
	x[column] = (x[column] - mini) / (maxi - mini)
	return x, (mini, maxi)
