import pandas as pd

def standard(df, column, mean=None, std=None):
	x = df.copy()
	if not mean and not std:
		mean = x[column].mean()
		std = x[column].std()
	x[column] = (x[column] - mean) / std
	return x, (mean, std)

def min_max(df, column, min=None, max=None):
	x = df.copy()
	if not min and not max:
		min = x[column].min()
		max = x[column].max()
	x[column] = (x[column] - min) / (max - min)
	return x, (min, max)