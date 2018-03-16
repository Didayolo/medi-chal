# Functions performing various data conversions for the ChaLearn AutoML challenge

# Main contributors: Arthur Pesah and Isabelle Guyon, August-October 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

import numpy as np
from scipy.sparse import *
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelBinarizer
import os

'''def get_type(data):
    try:
        return np.array(data).astype(int)
    except ValueError:
        return np.array(data).astype(float)'''

def file_to_array(filename, verbose=False):
    ''' Converts a .data AutoML file to an array '''
    with open(filename, "r") as data_file:
        if verbose: print ("Reading {}...".format(filename))
        lines = data_file.readlines()
        if verbose: print ("Converting {} to array...".format(filename))
        data = [lines[i].strip().split(' ') for i in range (len(lines))]
    return data

def file_to_libsvm(filename, data_binary, n_features, outname=None, verbose=False):
    ''' Converts a .data AutoML to svmlib format and return scipy.sparse matrix '''
    with open(filename, "r") as sample_file:
        if verbose: print("Reading {} samples...".format(filename))
        samples = sample_file.readlines()
        with open("tmp.txt", "w") as f:
            for sample in samples:
                sample = sample.strip().split()
                f.write("0 ")
                for value in sample:
                    if data_binary:
                        f.write(str(value) + ":1 ")
                    else:
                        f.write(str(value) + " ")
                f.write("\n")

    if verbose: print("Converting {} to libsvm format...".format(filename))
    l = load_svmlight_file("tmp.txt", zero_based=False, n_features=n_features)
    
    if outname:
        if verbose: print("Writing {} file to disk...".format(outname))
        os.rename("tmp.txt", outname)
    else:
        if verbose: print("Removing temporary file from disk...")
        os.remove("tmp.txt")

    return l[0]

def file_to_df(filename, data_type='train'):
    ''' Function to read the AutoML format and return a Pandas DataFrame '''
    
    file = filename + '_' + data_type

    if isfile(file + '.csv'):
        print('Reading '+ file + ' from CSV')
        Xy = pd.read_csv(csvfile)
    else:
        print('Reading '+ file + ' from AutoML format')

        ''' Check if mandatory file .data exists '''
        if not isfile(file + '.data'):
            raise FileNotFoundError('Mandatory file {}.data does not exist.'.format(file))

        ''' Check if optional file _feat.name exists '''
        feat_name = None
        if isfile(basename + '_feat.name'):
            feat_name = np.ravel(pd.read_csv(basename + '_feat.name', header=None))
            print('Names of the features name added.')

        ''' Check if optional file _feat.type exists '''
        feat_type = None
        if isfile(basename + '_feat.type'):
            feat_type = pd.read_csv(basename + '_feat.type', header=None).to_dict()[0]
            print('Types of the features type added.')
        
        X = pd.read_csv(file + '.data', sep=' ', header=None, names=feat_name, dtype=feat_type)

        ex_num, feat_num = X.shape
        print('Number of examples = %d' % ex_num)
        print('Number of features = %d' % feat_num)

        ''' Check if optional file _label.name exists '''
        label_name = None
        if isfile(basename + '_label.name'):
            label_name = np.ravel(pd.read_csv(basename + '_label.name', header=None))
            print('Labels\' name added.')

        ''' Check if mandatory file .solution exists '''
        if isfile(file + '.solution'):
            y = pd.read_csv(file + '.solution', sep=' ', header=None, names=label_name)

            label_num, class_num = y.shape
            print('Number of classes = %d' % class_num)

            assert(ex_num == label_num)

            ''' Decode labels if encoded '''
            if y.ndim > 1:
                label = y.apply(lambda x: x.idxmax(), axis=1)

            ''' Add labels y to examples X (last column)'''
            X = X.assign(target=label.values)

    return X

def sparse_file_to_sparse_list(filename):
    sparse_list = list()
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            sparse_list.append(line.split())
    return sparse_list

def read_first_line(filename):
	''' Read fist line of file '''
	with open(filename, "r") as data_file:
		line = data_file.readline()
		line = line.strip().split()
	return line  
 
def num_lines(filename):
	''' Count the number of lines of file '''
	return sum(1 for line in open(filename))

def binarization(array):
	''' Take a class datafile and encode it '''	
	return LabelBinarizer().fit_transform(array)
	
def convert_to_num(Ybin, verbose=True):
	''' Convert binary targets to numeric vector (typically classification target values) '''
	if verbose: print("\tConverting to numeric vector")
	Ybin = np.array(Ybin)
	if len(Ybin.shape) ==1:
         return Ybin
	classid=range(Ybin.shape[1])
	Ycont = np.dot(Ybin, classid)
	if verbose: print(Ycont)
	return Ycont
 
def convert_to_bin(Ycont, nval, verbose=True):
    ''' Convert numeric vector to binary (typically classification target values) '''
    if verbose: print ("\t_______ Converting to binary representation")
    Ybin=[[0]*nval for x in xrange(len(Ycont))]
    for i in range(len(Ybin)):
        line = Ybin[i]
        line[np.int(Ycont[i])]=1
        Ybin[i] = line
    return Ybin


def tp_filter(X, Y, feat_num=1000, verbose=True):
    ''' TP feature selection in the spirit of the winners of the KDD cup 2001
    Only for binary classification and sparse matrices'''
        
    if issparse(X) and len(Y.shape)==1 and len(set(Y))==2 and (sum(Y)/Y.shape[0])<0.1: 
        if verbose: print("========= Filtering features...")
        Posidx=Y>0
        nz=X.nonzero()
        mx=X[nz].max()
        if X[nz].min()==mx: # sparse binary
            if mx!=1: X[nz]=1
            tp=csr_matrix.sum(X[Posidx,:], axis=0)
     
        else:
            tp=np.sum(X[Posidx,:]>0, axis=0)
  

        tp=np.ravel(tp)
        idx=sorted(range(len(tp)), key=tp.__getitem__, reverse=True)   
        return idx[0:feat_num]
    else:
        feat_num = X.shape[1]
        return range(feat_num)
    
def replace_missing(X):
    # This is ugly, but
    try:
        if X.getformat()=='csr':
            return X
    except:
        return np.nan_to_num(X)
