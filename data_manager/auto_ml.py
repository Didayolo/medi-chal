# Imports
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from utilities import *
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
            raise OSError('No .data files found')

        self.data = dict()
        self.train_test = dict()
        self.init_data()

        self.info = dict()
        self.init_info(
            os.path.join(self.input_dir, self.basename + '_public.info'))

        self.feat_type = self.load_type(
            os.path.join(self.input_dir, self.basename + '_feat.type'))
        self.feat_name = self.load_name(
            os.path.join(self.input_dir, self.basename + '_feat.name'))
        self.label_name = self.load_name(
            os.path.join(self.input_dir, self.basename + '_label.name'))

        self.descriptors = dict()
        self.compute_descriptors()

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
            np.savetxt(filepath, X, fmt='%s')

        path = input_dir + '/' + basename
        write(path + ".data", X.values)
        write(path + "_feat.name", X.columns.values)
        write(path + "_feat.type", X.dtypes)

        if y:
            write(path + ".solution", y.values)
            write(path + "_label.name", y.columns.values)

        return cls(input_dir, basename)

    @classmethod
    def from_csv(cls, input_dir, basename, X_path, y_path=None, X_header='infer', y_header='infer'):
        """
            Class Method
            Build AutoML structure from CSV file.
            Generates autoML files from CSV file, write them on disk and call the AutoML constructor.
            
            :param input_dir: The directory where the autoML files will be stored.
            :param basename: The name of the dataset.
            :param X_path: path of the .csv containing the samples.
            :param y_path: path of the .csv containing the labels (optional if no labels).
            :param X_header: header (parameter under review)
            :param Y_header: header (parameter under review)
        """
        if os.path.exists(os.path.join(input_dir, X_path)):
            X = pd.read_csv(os.path.join(input_dir, X_path), header=X_header)
        else:
            raise OSError('{} file does not exist'.format(X_path))

        y = None
        if y_path and os.path.exists(os.path.join(input_dir, y_path)):
            y = pd.read_csv(os.path.join(input_dir, y_path), header=y_header)

        return cls.from_df(input_dir, basename, X, y)

    def init_data(self):
        """
            Load .data autoML files in a dictionary.
            
            :param test_size: If data is not splitted in autoML files, size of the test set.
                                Example : files = (i.e 'iris.data')
                                          test_size = 0.5
                                -> Data will be splitted 50% in X_train and 50% in X_test

            .. note:: If data is not splitted (i.e. no '_train.data', '_test.data'), samples are loaded in X_train.
        """
        if os.path.exists(
                os.path.join(self.input_dir, self.basename + '_train.data')):
            self.train_test['X_train'] = self.load_data(
                os.path.join(self.input_dir, self.basename + '_train.data'))
            self.train_test['X_test'] = self.load_data(
                os.path.join(self.input_dir, self.basename + '_test.data'))
            self.data['X'] = self.train_test['X_train'] + self.train_test['X_test']
            if os.path.exists(
                os.path.join(self.input_dir, self.basename + '_train.solution')):
                self.train_test['y_train'] = self.load_label(
                    os.path.join(self.input_dir, self.basename + '_train.solution'))
                self.train_test['y_test'] = self.load_label(
                    os.path.join(self.input_dir, self.basename + '_test.solution'))
                self.data['y'] = self.train_test['y_train'] + self.train_test['y_test']
        elif os.path.exists(
                os.path.join(self.input_dir, self.basename + '.data')):
            self.data['X'] = self.load_data(
                os.path.join(self.input_dir, self.basename + '.data'))
            if os.path.exists(os.path.join(self.input_dir, self.basename + '.solution')):
                self.data['y'] = self.load_label(
                    os.path.join(self.input_dir, self.basename + '.solution'))
        else:
            raise OSError('No .data files in {}.'.format(self.input_dir))

    def train_test_split(self, **kwargs):
        if 'y' in self.data:
            self.train_test['X_train'], self.train_test['X_test'], self.train_test['y_train'], self.train_test['y_test'] = \
                 train_test_split(self.data['X'], self.data['y'], **kwargs)
        else:
            if 'test_size' in kwargs:
                cut = np.floor(self.data['X'].shape[0] * (1 - kwargs.get('test_size')))
            elif 'train_size' in kwargs:
                cut = np.floot(self.data['X'].shape[0] * kwargs.get('train_size'))
            if 'shuffle' in kwargs:
                shuffle = kwargs.get('shuffle')
                if shuffle:
                    self.train_test['X_train'], self.train_test['X_test'] = np.random.permutation(self.data['X'])[:cut], \
                                                                            np.random.permutation(self.data['X'][cut:])
                else:
                    self.train_test['X_train'], self.train_test['X_test'] = self.data['X'][:cut], self.data['X'][cut:]
        return self.train_test


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
            If None, return an array of variables ['Unknown', ..., 'Unknown'].
                   
            :param filepath: Path of the file.
            :return: Array containing the data types. 
            :rtype: Numpy Array
        """
        return pd.read_csv(filepath, header=None).values.ravel() if os.path.exists(filepath) \
          else [self.info['feat_type']] * self.info['feat_num']

    def init_info(self, filepath):
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
            self.info = dict(zip(df[:, 0], df[:, 1]))
        else:
            print('No info file file found.')

            if os.path.exists(
                    os.path.join(self.input_dir, self.basename + '.data')):
                self.get_type_problem(
                    os.path.join(self.input_dir, self.basename + '.solution'))
            else:
                self.get_type_problem(
                    os.path.join(self.input_dir, self.basename + '_train.solution'))

            self.info['format'] = 'dense'
            self.info['is_sparse'] = 0
            self.info['train_num'], self.info['feat_num'] = self.train_test['X_train'].shape
            if 'y_train' and 'y_test' and 'X_train' in self.train_test:
                self.info['target_num'] = self.train_test['y_train'].shape[1]
                self.info['test_num'] = self.train_test['X_test'].shape[0]
                assert (self.info['train_num'] == self.train_test['y_train'].shape[0])
                assert (self.info['feat_num'] == self.train_test['X_test'].shape[1])
                assert (self.info['test_num'] == self.train_test['y_test'].shape[0])
                assert (self.info['target_num'] == self.info['y_test'].shape[1])
            self.info['usage'] = 'No info file'
            self.info['name'] = self.basename
            self.info['has_categorical'] = 0
            self.info['has_missing'] = 0
            self.info['feat_type'] = 'Mixed'
            self.info['time_budget'] = 600
            self.info['metric'] = 'r2_metric' if self.info['task'] == 'regression' else 'auc_metric'

        return self.info

    def get_data(self):
        return self.data

    def get_data_as_df(self):
        """ 
            Get data as a dictionary of pandas DataFrame.
            
            :return: Dictionary containing the data.
            :rtype: Dict
        """
        data = dict()
        data['X'] = pd.DataFrame(self.data['X'], columns=self.feat_name)
        if 'y' in self.data:
            data['y'] = pd.DataFrame(self.data['y'], columns=self.label_name)
        return data

    def get_train_test_as_df(self):
        """ 
            Get train test data as a dictionary of pandas DataFrame.
            
            :return: Dictionary containing the training sets and test sets.
            :rtype: Dict
        """
        train_test = dict()
        if 'X_train' and 'X_test' in train_test:
            train_test['X_train'] = pd.DataFrame(
                self.train_test['X_train'], columns=self.feat_name)
            train_test['X_test'] = pd.DataFrame(
                self.train_test['X_test'], columns=self.feat_name)
            if 'y_train' and 'y_test' in train_test:
                train_test['y_train'] = pd.DataFrame(
                    self.train_test['y_train'], columns=self.label_name)
                train_test['y_test'] = pd.DataFrame(
                    self.train_test['y_test'], columns=self.label_name)
        return train_test

    def get_info(self):
        return self.info
        
    def get_descriptors(self):
        return self.descriptors

    def save(self, out_path, out_name):
        def write_array(path, X):
            np.savetxt(path, X, fmt='%s')

        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        write_array(
            os.path.join(out_path, out_name + '.data'),
            self.data['X'])
        write_array(
            os.path.join(out_path, out_name + '_feat.name'), 
            self.feat_name)

        if 'y' in self.data:
            write_array(
                os.path.join(out_path, out_name + '.solution'),
                self.data['y'])

        if 'X_train' and 'X_test' in self.train_test:
            write_array(
                os.path.join(out_path, out_name + '_train.data'),
                self.train_test['X_train'])
            write_array(
                os.path.join(out_path, out_name + '_test.data'),
                self.train_test['X_test'])
            if 'y_train' and 'y_test' in self.train_test:
                write_array(
                    os.path.join(out_path, out_name + '_test.solution'),
                    self.train_test['y_train'])
                write_array(
                    os.path.join(out_path, out_name + '_test.solution'),
                    self.train_test['y_test'])
                write_array(
                    os.path.join(out_path, out_name + '_label.name'),
                    self.label_name)

        with open(os.path.join(out_path, out_name + '_public.info'), 'w') as f:
            for key, item in self.info.items():
                f.write(str(key))
                f.write(' = ')
                f.write(str(item))
                f.write('\n')

    def get_type_problem(self, solution_filepath):
        """ 
            Get the type of problem directly from the solution file (in case we do not have an info file).

            :param solution_filepath: Path of the file
            :return: Type of the problem stored in the info dict attribute as 'task'
            :rtype: str
        """
        if 'task' not in self.info.keys() and self.train_test['y_train'].size != 0:
            solution = pd.read_csv(
                solution_filepath, sep=' ', header=None).values
            target_num = solution.shape[1]
            self.info['target_num'] = target_num
            if target_num == 1:  # if we have only one column
                solution = np.ravel(solution)  # flatten
                nbr_unique_values = len(np.unique(solution))
                if nbr_unique_values < len(solution) / 8:
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
        else:
            self.info['task'] = 'Unknown'
        return self.info['task']

    def get_processed_data(self):
        """ 
            Preprocess data.
			- Missing values inputation
			- +Inf and -Inf replaced by maximum and minimum
			- One hot encoding for categorical variables

            :return: Dictionnary containing the preprocessed data as Pandas DataFrame
            :rtype: Dict
		"""
        processed_data, processed_train_test = dict(), dict()
        data_df = self.get_data_as_df()
        train_test_df = self.get_train_test_as_df()

        for k in list(data_df.keys()):
            processed_data[k] = preprocessing(data_df[k])
        for k in list(train_test_df.keys()):
            processed_train_test[k] = preprocessing(train_test_df[k])

        return processed_data, processed_train_test

    def compute_descriptors(self):
        """ 
            Compute descriptors of the dataset and store them in self.descriptors dictionary.
			- ratio: Dataset ratio
			- skewness_min: Minimum skewness over features 
			- skewness_max: Maximum skewness over features
			- skewness_mean: Average skewness over features
		"""
        self.descriptors['ratio'] = int(self.info['feat_num']) / int(self.info['train_num'])
            
        skewness = self.get_train_test_as_df()['X_train'].skew()
        self.descriptors['skewness_min'] = skewness.min()
        self.descriptors['skewness_max'] = skewness.max()
        self.descriptors['skewness_mean'] = skewness.mean()

    def show_info(self):
        """ 
            Show AutoML info 
        """
        for k in list(self.info.keys()):
            key = k.capitalize().replace('_', ' ')
            value = self.info[k]
            if isinstance(value, str):
                value = value.capitalize().replace('_', ' ').replace('.', ' ')

            print('{}: {}'.format(key, value))

    def show_descriptors(self):
        """ 
            Show descriptors of the dataset 
			- Dataset ratio
			- Scatter plot features matrix
			- Classes distribution
			- Correlation matrix
			- Hierarchical clustering heatmap
			- First two principal components
			- First two LDA components
			- T-SNE plot
		"""

        # Text

        for k in list(self.descriptors.keys()):
            key = k.capitalize().replace('_', ' ')
            value = self.descriptors[k]
            if isinstance(value, str):
                value = value.capitalize().replace('_', ' ').replace('.', ' ')

            print('{}: {}'.format(key, value))

        # Plots
        x_sets = ['X_train']
        y_sets = ['y_train']
        # If there is a test set
        if (len(self.train_test['X_test']) > 0):
            x_sets.append('X_test')
            y_sets.append('y_test')

        train_test = self.get_train_test_as_df()

        print('Scatter plot matrix')
        sns.set(style="ticks")
        for x in x_sets:
            print(x)
            sns.pairplot(train_test[x])
            plt.show()

        print('Classes distribution')
        for y in y_sets:
            print(y)
            show_classes(train_test[y])

        print('Correlation matrix')
        for x in x_sets:
            print(x)
            show_correlation(train_test[x])

        print('Hierarchical clustering heatmap')
        row_method = 'average'
        column_method = 'single'
        row_metric = 'euclidean'  #'cityblock' #cosine
        column_metric = 'euclidean'
        color_gradient = 'coolwarm'  #'red_white_blue
        for x in x_sets:
            print(x)
            heatmap(train_test[x], row_method, column_method, row_metric,
                    column_metric, color_gradient)

        print('Principal components analysis')
        for i in range(len(x_sets)):
            print(x_sets[i])
            print(y_sets[i])
            show_pca(train_test[x_sets[i]], train_test[y_sets[i]])

        # Linear discriminant analysis
        #if int(self.info['target_num']) > 2: # or label_num ?
        if False:  # TODO
            print('Linear discriminant analysis')
            for i in range(len(x_sets)):
                print(x_sets[i])
                print(y_sets[i])
                show_lda(train_test[x_sets[i]], train_test[y_sets[i]])

        print('T-distributed stochastic neighbor embedding')
        for i in range(len(x_sets)):
            print(x_sets[i])
            print(y_sets[i])
            show_tsne(train_test[x_sets[i]], train_test[y_sets[i]])
