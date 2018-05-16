# Imports
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from utilities import *
from preprocessing import *
import matplotlib.pyplot as plt
import seaborn as sns

class AutoML():
    def __init__(self, input_dir="", basename="", verbose=False):
        """
            Constructor.
            Recover all autoML files available and build the AutoML structure containing them.
            :param input_dir: The directory where the autoML files are stored.
            :param basename: The name of the dataset (i.e. the prefix in the name of the files) 
                                Example : files = ('iris.data', iris_feat.name', etc.)
                                          basename = 'iris'
            :param test_size: Proportion of the dataset to include in the test split.
            :param verbose: Display additional information during run.
        """
        if os.path.isdir(input_dir):
            self.input_dir = input_dir
        else:
            raise OSError(
                'Input directory {} does not exist.'.format(input_dir))

        self.basename = basename
        if os.path.exists(os.path.join(self.input_dir, basename + '_train.data')) or \
         os.path.exists(os.path.join(self.input_dir, basename + '.data')):
            self.basename = basename
        else:
            raise OSError('No .data files found with prefix {}'.format(basename))

        # Data.
        self.data, self.target = dict(), dict()
        self.init_data()

        # AutoML info.
        self.info = dict()
        self.init_info(os.path.join(self.input_dir, self.basename + '_public.info'), verbose=verbose)

        # Name of each variable.
        self.feat_name = self.load_name(os.path.join(self.input_dir, self.basename + '_feat.name'))
        # Name of each target.
        self.label_name = self.load_name(os.path.join(self.input_dir, self.basename + '_label.name'))
            
        # Type of each variable.
        self.feat_type = self.load_type(os.path.join(self.input_dir, self.basename + '_feat.type'))

        self.recap()

        self.descriptors = dict()

    @classmethod
    def from_df(cls, input_dir, basename, X, y=None):
        """
            Class Method
            Build AutoML structure from Pandas DataFrame.
            Generates autoML files from Pandas DataFrame, write them on disk and call the AutoML constructor.
            
            :param input_dir: The directory where the autoML files will be stored.
            :param basename: The name of the dataset.
            :param X: Dataset containing the samples.
            :param y: Dataset containing the labels (optional if no labels).
        """
        def write(filepath, X):
            np.savetxt(filepath, X, delimiter=' ', fmt='%s')

        input_dir += '/' + basename + '_automl'
        if not os.path.isdir(input_dir):
            os.mkdir(input_dir)

        path = input_dir + '/' + basename
        write(path + ".data", X.values)
        if X.columns.values.dtype == np.int64:
            X = X.add_prefix('X')
        write(path + "_feat.name", X.columns.values)
        write(path + "_feat.type", get_types(X))

        if y is not None:
            write(path + ".solution", y.values)
            if isinstance(y, pd.Series):
                write(path + "_label.name", [y.name])
            else:
                write(path + "_label.name", y.columns)

        return cls(input_dir, basename)

    @classmethod
    def from_csv(cls, input_dir, basename, data_path, target=None, seps=[',', ' '], headers=['infer', 'infer']):
        """
            Class Method
            Build AutoML structure from CSV file.
            Generates autoML files from CSV file, write them on disk and call the AutoML constructor.
            
            :param input_dir: The directory where the autoML files will be stored.
            :param basename: The name of the dataset.
            :param X: path of the .csv containing the samples.
            :param y: path of the .csv containing the labels (optional if no labels).
                           or column number (integer)
                           or list of column numbers (list of integers).
            :param header_X: header (parameter under review)
            :param header_y: header (parameter under review)
        """
        if os.path.exists(os.path.join(input_dir, data_path)):
            X = pd.read_csv(os.path.join(input_dir, data_path), sep=seps[0], header=headers[0], engine='python')
        else:
            print(os.path.join(input_dir, data_path))
            raise OSError('{} file does not exist'.format(data_path))

        if isinstance(target, str) and os.path.exists(os.path.join(input_dir, target)):
            y = pd.read_csv(os.path.join(input_dir, y), sep=seps[1], header=headers[1], engine='python')
        elif isinstance(target, int):
            y = pd.Series(X[X.columns[target]], name=X.columns[target])
            X = X.drop(X.columns[target], axis=1)
        elif isinstance(target, list):
            y = pd.DataFrame(X[X.columns[target]], columns=X.columns[target])
            X = X.drop(X.columns[target], axis=1)
        elif isinstance(target, str):
            y = pd.Series(X[target], name=target)
            X = X.drop([y], axis=1)
        else:
            y = None
        return cls.from_df(input_dir, basename, X, y)

    def init_data(self):
        """
            Load .data autoML files in a dictionary.
        """
        if os.path.exists(os.path.join(self.input_dir, self.basename + '.data')):
            self.data['X'] = self.load_data(os.path.join(self.input_dir, self.basename + '.data'))  
            if os.path.exists(os.path.join(self.input_dir, self.basename + '.solution')):
                self.target['y'] = self.load_label(os.path.join(self.input_dir, self.basename + '.solution'))

        if os.path.exists(os.path.join(self.input_dir, self.basename + '_train.data')):
            self.data['Xtr'] =  self.load_data(os.path.join(self.input_dir, self.basename + '_train.data'))
            if os.path.exists(os.path.join(self.input_dir, self.basename + '_train.solution')):
                self.target['ytr'] =  self.load_data(os.path.join(self.input_dir, self.basename + '_train.solution'))
            if os.path.exists(os.path.join(self.input_dir, self.basename + '_test.data')):
                self.data['Xte'] =  self.load_data(os.path.join(self.input_dir, self.basename + '_test.data')) 
                if os.path.exists(os.path.join(self.input_dir, self.basename + '_test.solution')):
                    self.target['yte'] =  self.load_data(os.path.join(self.input_dir, self.basename + '_test.solution'))

        if 'X' not in self.data and ('Xtr' and 'Xte' in self.data):
            self.data['X'] = np.vstack([self.data['Xtr'], self.data['Xte']])
            if 'y' not in self.target and ('ytr' and 'yte' in self.target):
                self.target['y'] = np.vstack([self.target['ytr'], self.target['yte']])

    def load_data(self, filepath):
        """
            Load a .data autoML file in an array.
            :param filepath: path of the file.
            :return: array containing the data. 
            :rtype: numpy array
        """
        return pd.read_csv(filepath, sep=' ', header=None).values if os.path.exists(filepath) \
          else []

    def load_label(self, filepath):
        """ 
            Load a .solution autoML file in an array.
            :param filepath: Path of the file.
            :return: Array containing the data labels. 
            :rtype: Numpy Array
        """
        return pd.read_csv(filepath, sep=' ', header=None).values if os.path.exists(filepath) \
          else []

    def load_name(self, filepath):
        """
            Load a _feat.name autoML file in an array.
            If None, return an array of variables [X1, ..., XN] (with N the number of features).
                   
            :param filepath: Path of the file.
            :return: Array containing the data names. 
            :rtype: Numpy Array
        """
        return pd.read_csv(filepath, header=None).values.ravel() if os.path.exists(filepath) \
          else ['X' + str(i) for i in range(self.info['feat_num'])]

    def load_type(self, filepath):
        """
            Load a _feat.type autoML file in an array.
            If None, compute it.
                   
            :param filepath: Path of the file.
            :return: Array containing the data types. 
            :rtype: Numpy Array
        """
        dtypes = []
        if os.path.exists(filepath):
            dtypes = pd.read_csv(filepath, header=None).values.ravel()
        else:
            dtypes = get_types(self.data['X'])
        return dtypes

    def init_info(self, filepath, verbose=True):
        """
            Load a _public.info autoML file in a dictionary.
            If None, build the dictionary on its own.
                   
            :param filepath: Path of the file.
            :return: Dictionary containing the data information. 
            :rtype: Dict
        """
        if os.path.exists(filepath):
            df = pd.read_csv(
                os.path.join(self.input_dir, self.basename + '_public.info'),
                header=None,
                sep='=').values
            for x in df:
                x[0] = x[0].replace("'", '').strip()
                x[1] = x[1].replace("'", '').strip()
                # Convert numerical information in int instead of str
                for v in x[1]:
                    try:
                        x[1] = int(x[1])
                    except:
                        pass
            self.info = dict(zip(df[:, 0], df[:, 1]))

        else:
            if verbose:
                print('No info file found.')

            self.info['format'] = 'dense'
            self.info['is_sparse'] = 0
            self.info['feat_num'] = self.data['X'].shape[1]
            if 'y' in self.target:
                self.info['target_num'] = self.target['y'].shape[1]
                self.problem_task()
                self.info['metric'] = 'r2_metric' if self.info['task'] == 'regression' else 'auc_metric'
            if ('ytr' and 'yte' in self.target) and ('Xtr' and 'Xte' in self.data):
                self.info['train_num'] = self.data['Xtr'].shape[0]
                self.info['test_num'] = self.data['Xte'].shape[0]
                assert (self.info['train_num'] == self.target['ytr'].shape[0])
                assert (self.info['feat_num'] == self.data['Xte'].shape[1])
                assert (self.info['test_num'] == self.target['yte'].shape[0])
                assert (self.info['target_num'] == self.target['yte'].shape[1])
            self.info['usage'] = 'No info file'
            self.info['name'] = self.basename
            self.info['has_categorical'] = 0
            self.info['has_missing'] = 0
            self.info['feat_type'] = 'mixed'
            self.info['time_budget'] = 600
        return self.info

    def problem_task(self):
        """ 
            Get the type of problem directly from the solution file (in case we do not have an info file).
            :param solution_filepath: Path of the file
            :return: Type of the problem stored in the info dict attribute as 'task'
            :rtype: str
        """
        if 'task' not in self.info.keys():
            solution = np.ravel(self.target['y'])
            if self.info['target_num'] == 1:  # if we have only one column
                if len(np.unique(solution)) < len(solution) / 8:
                    # Classification
                    self.info['label_num'] = len(np.unique(solution))
                    if len(np.unique(solution)) == 2:
                        self.info['task'] = 'binary.classification'
                        self.info['target_type'] = 'binary'
                    else:
                        self.info['task'] = 'multiclass.classification'
                        self.info['target_type'] = 'categorical'
                else:
                    # Regression
                    self.info['label_num'] = 0
                    self.info['task'] = 'regression'
                    self.info['target_type'] = 'numerical'
            else:
                # Multilabel or multiclass
                self.info['label_num'] = self.info['target_num']
                self.info['target_type'] = 'binary'
                if any(item > 1 for item in map(np.sum, solution.astype(int))):
                    self.info['task'] = 'multilabel.classification'
                else:
                    self.info['task'] = 'multiclass.classification'
        else:
            self.info['task'] = 'Unknown'

    def show_info(self):
        """ Show AutoML info 
        """
        for k in list(self.info.keys()):
            key = k.capitalize().replace('_', ' ')
            value = self.info[k]
            if isinstance(value, str):
                value = value.capitalize().replace('_', ' ').replace('.', ' ')

            print('{}: {}'.format(key, value)) 

    def get_data_as_df(self):
        """ 
            Get data as a dictionary of pandas DataFrame.
            
            :return: Dictionary containing the data as pandas DataFrame.
            :rtype: Dict
        """
        data, target = dict(), dict()
        
        # X/y
        if 'X' in self.data:
            data['X'] = pd.DataFrame(self.data['X'], columns=self.feat_name)
            if 'y' in self.target:
                target['y'] = pd.DataFrame(self.target['y'], columns=self.label_name)

        # Train/test
        if 'Xtr' and 'Xte' in self.data:
            data['Xtr'] = pd.DataFrame(self.data['Xtr'], columns=self.feat_name)
            data['Xte'] = pd.DataFrame(self.data['Xte'], columns=self.feat_name)
            if 'ytr' and 'yte' in self.target:
                target['ytr'] = pd.DataFrame(self.target['ytr'], columns=self.label_name)
                target['yte'] = pd.DataFrame(self.target['yte'], columns=self.label_name)       
        return data, target

    def recap(self):
        print('------- Recap of files ------')
        print('X: Yes') if 'X' in self.data else print('X: No')
        print('X train: Yes') if 'Xtr' in self.data else print('X train: No')
        print('X test: Yes') if 'Xte' in self.data else print('X test: No \n')
        
        print('y: Yes') if 'y' in self.target else print('y: No')
        print('y train: Yes') if 'ytr' in self.target else print('y train: No')
        print('y test: Yes') if 'yte' in self.target else print('y test: No \n')
        
    def save(self, out_path, out_name):
        """ Save data in auto_ml file format
        
            :param out_path: Path of output directory.
            :param out_name: Basename of output files.
        """
        def write_array(path, X):
            np.savetxt(path, X, fmt='%s')

        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        write_array(os.path.join(out_path, out_name + '.data'), self.data['X'])
        write_array(os.path.join(out_path, out_name + '_feat.name'), self.feat_name)

        if 'y' in self.target:
            write_array(os.path.join(out_path, out_name + '.solution'), self.target['y'])

        if 'Xtr' and 'Xte' in self.data:
            write_array(os.path.join(out_path, out_name + '_train.data'), self.data['Xtr'])
            write_array(os.path.join(out_path, out_name + '_test.data'), self.data['Xte'])
            if 'ytr' and 'yte' in self.target:
                write_array(os.path.join(out_path, out_name + '_test.solution'), self.target['ytr'])
                write_array(os.path.join(out_path, out_name + '_test.solution'), self.target['yte'])
                write_array(os.path.join(out_path, out_name + '_label.name'), self.label_name)

        with open(os.path.join(out_path, out_name + '_public.info'), 'w') as f:
            for key, item in self.info.items():
                f.write(str(key))
                f.write(' = ')
                f.write(str(item))
                f.write('\n')

    # Under Review ...
    def process_data(self, how='dependent', normalization='standard', categorical='label', target='label', inplace=False):
        data, target = self.get_data_as_df()

        self.data['X'] = preprocess(data['X'], self.feat_type, normalization=normalization, encoding=categorical)
        if 'Xtr' and 'Xte' in self.data:
            if how == 'dependent':
                self.data['Xtr'] = self.data['X'][:self.data['Xtr'].shape[0], :]
                self.data['Xte'] = self.data['X'][self.data['Xte'].shape[0]:, :]
            elif how == 'independent':
                self.data['Xtr'] = preprocess(data['Xtr'], self.feat_type, normalization=normalization, encoding=categorical)
                self.data['Xte'] = preprocess(data['Xte'], self.feat_type, normalization=normalization, encoding=categorical)
            else:
                raise OSError('how argument not valid.')

        if 'y' in self.target:
            if not self.target['y'].ndim > 1:
                self.target['y'] = preprocess(target['y'], ['Categorical'], normalization='none', encoding=target)
                if 'ytr' and 'yte' in self.target:
                    self.target['ytr'] = target['y'][:self.target['ytr'].shape[0]]
                    self.target['yte'] = target['y'][self.target['yte'].shape[0]:]  

    # Under review...
    def train_test_split(self, **kwargs):
        """ 
            Apply the train test split
        """
        if 'Xtr' and 'Xte' not in self.data:
            if 'y' in self.target:
                self.data['Xtr'], self.data['Xte'], self.target['ytr'], self.target['yte'] = \
                    train_test_split(self.data['X'], self.target['y'], **kwargs)
            else:
                self.data['Xtr'], self.data['Xte'], _, _ = \
                    train_test_split(self.data['X'], np.zeros(self.data['X'].shape), **kwargs)

    # Under review...
    def compute_descriptors(self):
        """ 
            Compute descriptors of the dataset and store them in the descriptors dictionary.
            - ratio: Dataset ratio
            - symb_ratio: Ratio of symbolic attributes
            - class_deviation: Standard deviation of class distribution
            - missing_proba: Probability of missing values
            - skewness_min: Minimum skewness over features 
            - skewness_max: Maximum skewness over features
            - skewness_mean: Average skewness over features
        """ # - defective_proba: Probability of defective records (columns with missing values)
        data, target = self.get_data_as_df()
            
        self.descriptors['symb_ratio'] = list(self.feat_type).count('Numerical') / len(self.feat_type)
        
        if 'X' in self.data:
            self.descriptors['missing_proba'] = (data['X'].isnull().sum() / len(data['X'])).mean()
            skewness = data['X'].skew()
            self.descriptors['skewness_min'] = skewness.min()
            self.descriptors['skewness_max'] = skewness.max()
            self.descriptors['skewness_mean'] = skewness.mean()

        if 'y' in self.target:
             self.descriptors['class_deviation'] = target['y'].std().mean()

        if 'Xtr' in self.data:
            self.descriptors['ratio'] = int(self.info['feat_num']) / int(self.info['train_num'])


    '''def show_descriptors(self, processed_data=False):
        """ 
            Show descriptors of the dataset.
            
            Descriptors:
            - ratio: Dataset ratio
            - symb_ratio: Ratio of symbolic attributes
            - class_deviation: Standard deviation of class distribution
            - missing_proba: Probability of missing values
            - skewness_min: Minimum skewness over features 
            - skewness_max: Maximum skewness over features
            - skewness_mean: Average skewness over features
            
            Plots:
            - Scatter plot features matrix
            - Classes distribution
            - Correlation matrix
            - Hierarchical clustering heatmap
            - First two principal components
            - First two LDA components
            - T-SNE plot
            
            :param processed_data: Boolean defining whether to display the
                                    descriptors of the raw data or the processed data
        """
        
        if processed_data:
            data = self.get_processed_data()
            train = data[:self.info['train_num'], :]
            test = data[self.info['train_num']:, :]
        else:
            data = self.get_data_as_df()

        # Text
        printmd('** Descriptors **')

        for k in list(self.descriptors.keys()):
            key = k.capitalize().replace('_', ' ')
            value = self.descriptors[k]
            if isinstance(value, str):
                value = value.capitalize().replace('_', ' ').replace('.', ' ')

            print('{}: {}'.format(key, value))

        # Plots
        print('')
        printmd('** Plots **')
        
        x_sets = ['X_train']
        y_sets = []
        
        # If there is a class
        if 'y_train' in self.data:
            y_sets.append('y_train')
            y_sets.append('y_test')
        
        # If there is a test set
        if (len(self.data['X_test']) > 0):
            x_sets.append('X_test')

        if int(self.info['feat_num']) < 20: # TODO selection, plot with y
            printmd('** Scatter plot matrix **')
            sns.set(style="ticks")
            for x in x_sets:
                print(x)
                sns.pairplot(data[x]) 
                plt.show()

        printmd('** Correlation matrix **')
        for x in x_sets:
            print(x)
            show_correlation(data[x])

        printmd('** Hierarchical clustering heatmap **')
        for x in x_sets:
            print(x)
            # row_method, column_method, row_metric, column_metric, color_gradient
            heatmap(data[x], 'average', 'single', 'euclidean',
                    'euclidean', 'coolwarm')

        if len(y_sets) > 0:
        
            printmd('** Classes distribution **')
            for y in y_sets:
                print(y)
                show_classes(data[y])
        
            printmd('** Principal components analysis **')
            for i in range(len(x_sets)):
                print(x_sets[i])
                print(y_sets[i])
                show_pca(data[x_sets[i]], data[y_sets[i]])

            printmd('** T-distributed stochastic neighbor embedding **')
            for i in range(len(x_sets)):
                print(x_sets[i])
                print(y_sets[i])
                show_tsne(data[x_sets[i]], data[y_sets[i]])

            printmd('** Linear discriminant analysis **')
            for i in range(len(x_sets)):
                print(x_sets[i])
                print(y_sets[i])
                show_lda(data[x_sets[i]], data[y_sets[i]])'''