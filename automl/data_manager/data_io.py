# Functions performing various input/output operations for the ChaLearn AutoML challenge

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

from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

import numpy as np
import pandas as pd
import os
import shutil
from scipy.sparse import * # used in data_binary_sparse 
from zipfile import ZipFile, ZIP_DEFLATED
from contextlib import closing
import data_converter
from sys import stderr
from sys import version
from glob import glob as ls
from os import getcwd as pwd
from os.path import isfile
from pip import get_installed_distributions as lib
#import yaml
from shutil import copy2
import csv
#import psutil
import platform

# ================ Small auxiliary functions =================

from os.path import isfile
import numpy as np


def read_as_df(filename, data_type='train'):
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

# ================ Small auxiliary functions =================

swrite = stderr.write

if (os.name == "nt"):
    filesep = '\\'
else:
    filesep = '/'

def write_list(lst):
    ''' Write a list of items to stderr (for debug purposes)'''
    for item in lst:
        swrite(item + "\n") 
        
def print_dict(verbose, dct):
    ''' Write a dict to stderr (for debug purposes)'''
    if verbose:
        for item in dct:
            print(item + " = " + str(dct[item]))

def mkdir(d):
    ''' Create a new directory'''
    if not os.path.exists(d):
        os.makedirs(d)

def mvdir(source, dest):
    ''' Move a directory'''
    if os.path.exists(source):
        os.rename(source, dest)

def rmdir(d):
    ''' Remove an existingdirectory'''
    if os.path.exists(d):
        shutil.rmtree(d)

def vprint(mode, t):
    ''' Print to stdout, only if in verbose mode'''
    if(mode):
        print(t) 

# ================ Output prediction results and prepare code submission =================

def write(filename, predictions):
    ''' Write prediction scores in prescribed format'''
    with open(filename, "w") as output_file:
        for row in predictions:
            if type(row) is not np.ndarray and type(row) is not list:
                row = [row]
            for val in row:
                output_file.write('{0:g} '.format(float(val)))
                output_file.write('\n')

def zipdir(archivename, basedir):
    '''Zip directory, from J.F. Sebastian http://stackoverflow.com/'''
    assert os.path.isdir(basedir)
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z:
        for root, dirs, files in os.walk(basedir):
            #NOTE: ignore empty directories
            for fn in files:
                if fn[-4:]!='.zip':
                    absfn = os.path.join(root, fn)
                    zfn = absfn[len(basedir)+len(os.sep):] #XXX: relative path
                    z.write(absfn, zfn)
                    
# ================ Inventory input data and create data structure =================

def inventory_data(input_dir):
    ''' Inventory the datasets in the input directory and return them in alphabetical order'''
    # Assume first that there is a hierarchy dataname/dataname_train.data
    training_names = inventory_data_dir(input_dir)
    ntr=len(training_names)
    if ntr==0:
        # Try to see if there is a flat directory structure
        training_names = inventory_data_nodir(input_dir)
        ntr=len(training_names)
    if ntr==0:
        print('WARNING: Inventory data - No data file found')
        training_names = []
        training_names.sort()
    return training_names

def inventory_data_nodir(input_dir):
    ''' Inventory data, assuming flat directory structure'''
    training_names = ls(os.path.join(input_dir, '*_train.data'))
    for i in range(0,len(training_names)):
        name = training_names[i]
        training_names[i] = name[-name[::-1].index(filesep):-name[::-1].index('_')-1]
        check_dataset(input_dir, training_names[i])
    return training_names

def inventory_data_dir(input_dir):
    ''' Inventory data, assuming flat directory structure, assuming a directory hierarchy'''
    training_names = ls(input_dir + '/*/*_train.data') # This supports subdirectory structures obtained by concatenating bundles
    for i in range(0,len(training_names)):
        name = training_names[i]
        training_names[i] = name[-name[::-1].index(filesep):-name[::-1].index('_')-1]
        check_dataset(os.path.join(input_dir, training_names[i]), training_names[i])
    return training_names

def check_dataset(dirname, name):
    ''' Check the test and valid files are in the directory, as well as the solution'''
    valid_file = os.path.join(dirname, name + '_valid.data')
    if not os.path.isfile(valid_file):
      print('No validation file for ' + name)
      exit(1)  
      test_file = os.path.join(dirname, name + '_test.data')
    if not os.path.isfile(test_file):
        print('No test file for ' + name)
        exit(1)
	# Check the training labels are there
    training_solution = os.path.join(dirname, name + '_train.solution')
    if not os.path.isfile(training_solution):
        print('No training labels for ' + name)
        exit(1)
        return True


def data(filename, nbr_features=None, verbose = False):
    ''' The 2nd parameter makes possible a using of the 3 functions of data reading (data, data_sparse, data_binary_sparse) without changing parameters'''
    if verbose: print (np.array(data_converter.file_to_array(filename)))
    return data_converter.file_to_array(filename)

def data_sparse (filename, nbr_features):
    ''' This function takes as argument a file representing a sparse matrix
    sparse_matrix[i][j] = "a:b" means matrix[i][a] = basename and load it with the loadsvm load_svmlight_file
    '''
    return data_converter.file_to_libsvm(filename=filename, data_binary=False, n_features=nbr_features)

def data_binary_sparse (filename , nbr_features):
    ''' This fuction takes as argument a file representing a sparse binary matrix 
    sparse_binary_matrix[i][j] = "a" and transforms it temporarily into file svmlibs format( <index2>:<value2>)
    to load it with the loadsvm load_svmlight_file
    '''
    return data_converter.file_to_libsvm(filename=filename, data_binary=True, n_features=nbr_features)



# ================ Copy results from input to output ==========================

def copy_results(datanames, result_dir, output_dir, verbose):
    ''' This function copies all the [dataname.predict] results from result_dir to output_dir'''
    missing_files = []
    for basename in datanames:
        try:
            missing = False
            test_files = ls(result_dir + "/" + basename + "*_test*.predict")
            if len(test_files)==0: 
                vprint(verbose, "[-] Missing 'test' result files for " + basename) 
                missing = True
                valid_files = ls(result_dir + "/" + basename + "*_valid*.predict")
            if len(valid_files)==0: 
                vprint(verbose, "[-] Missing 'valid' result files for " + basename) 
                missing = True
            if missing == False:
                for f in test_files: copy2(f, output_dir)
                for f in valid_files: copy2(f, output_dir)
                vprint( verbose,  "[+] " + basename.capitalize() + " copied")
            else: 
                missing_files.append(basename)           
        except:
            vprint(verbose, "[-] Missing result files")
            return datanames
    return missing_files

# ================ Display directory structure and code version (for debug purposes) =================

def show_dir(run_dir):
	print('\n=== Listing run dir ===')
	write_list(ls(run_dir))
	write_list(ls(run_dir + '/*'))
	write_list(ls(run_dir + '/*/*'))
	write_list(ls(run_dir + '/*/*/*'))
	write_list(ls(run_dir + '/*/*/*/*'))

'''def show_io(input_dir, output_dir):     
    swrite('\n=== DIRECTORIES ===\n\n')
    # Show this directory
    swrite("-- Current directory " + pwd() + ":\n")
    write_list(ls('.'))
    write_list(ls('./*'))
    write_list(ls('./*/*'))
    swrite("\n")

    # List input and output directories
    swrite("-- Input directory " + input_dir + ":\n")
    write_list(ls(input_dir))
    write_list(ls(input_dir + '/*'))
    write_list(ls(input_dir + '/*/*'))
    write_list(ls(input_dir + '/*/*/*'))
    swrite("\n")
    swrite("-- Output directory  " + output_dir + ":\n")
    write_list(ls(output_dir))
    write_list(ls(output_dir + '/*'))
    swrite("\n")

    # write meta data to sdterr
    swrite('\n=== METADATA ===\n\n')
    swrite("-- Current directory " + pwd() + ":\n")
    try:
        metadata = yaml.load(open('metadata', 'r'))
        for key,value in metadata.items():
            swrite(key + ': ')
            swrite(str(value) + '\n')
    except:
        swrite("none\n");
    swrite("-- Input directory " + input_dir + ":\n")
    try:
        metadata = yaml.load(open(os.path.join(input_dir, 'metadata'), 'r'))
        for key,value in metadata.items():
            swrite(key + ': ')
            swrite(str(value) + '\n')
        swrite("\n")
    except:
        swrite("none\n");'''

def show_version():
    # Python version and library versions
    swrite('\n=== VERSIONS ===\n\n')
    # Python version
    swrite("Python version: " + version + "\n\n")
    # Give information on the version installed
    swrite("Versions of libraries installed:\n")
    map(swrite, sorted(["%s==%s\n" % (i.key, i.version) for i in lib()]))

 # Compute the total memory size of an object in bytes

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

                    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

    # write the results in a csv file
'''def platform_score ( basename , mem_used ,n_estimators , time_spent , time_budget ):
    # write the results and platform information in a csv file (performance.csv)
    with open('performance.csv', 'a') as fp:
        a = csv.writer(fp, delimiter=',')
        #['Data name','Nb estimators','System', 'Machine' , 'Platform' ,'memory used (Mb)' , 'number of CPU' ,' time spent (sec)' , 'time budget (sec)'],
        data = [
        [basename,n_estimators,platform.system(), platform.machine(),platform.platform() , float("{0:.2f}".format(mem_used/1048576.0)) , str(psutil.cpu_count()) , float("{0:.2f}".format(time_spent)) ,    time_budget ]
        ]
        a.writerows(data)'''

