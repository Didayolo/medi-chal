import pandas as pd

def mean(df, column):
	x = df.copy()
	x[column] = x[column].fillna(x[column].mean())
	return x

def median(df, column):
	x = df.copy()
	x[column] = x[column].fillna(x[column].median())
	return x

def remove(df, columns):
	x = df.copy()
	x = x.dropna(axis=0, subset=columns)
	return x

def most(df, column):
    """ Replace by the most frequent value
    """
    x = df.copy()
    most_frequent_value = x[column].value_counts().idxmax()
    x[column] = x[column].fillna(most_frequent_value)
    return x
