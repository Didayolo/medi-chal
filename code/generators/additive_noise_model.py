# Imports
import numpy as np
from sys import path
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
problem_dir = 'code/auto_ml'
path.append(problem_dir)
from auto_ml import AutoML


class RF_generator():
    def __init__(self, ds):
        """ Data generator using multiple imputations with random forest
            Input:
              ds: AutoML object containing data
        """
        # List of Random Forests
        self.models = []

        # Random forest from sklearn
        self.regressor = RandomForestRegressor
        self.classifier = RandomForestClassifier

        # AutoML dataset
        self.ds = ds
        # self.ds.process_data() # todo: optimize

        # Generated DataFrame
        self.gen_data = self.ds.get_data(processed=True).copy()

    def process_data(self, **kwargs):
        """ Apply process_data method on ds
        """
        self.ds.process_data(**kwargs)

    def get_data(self):
        return self.ds.get_data('X', processed=True)

    def fit(self, **kwargs):
        """
            Fit one random forest for each column, given the others
            :param kwargs: Random Forest parameters
        """
        data = self.get_data()
        #print('New one')
        #te = list(data.columns)
        for i in range(len(data.columns)):
            #print(te[i])
            # May bug with duplicate names in columns
            y = data[data.columns[i]]
            X = data.drop(data.columns[i], axis=1)

            # Regressor or classifier
            model = self.regressor(n_estimators=5)
            #else:
            #    model = self.classifier(n_estimators=5)
            model.fit(X, y)
            self.models.append(model)

    def generate(self):
        """
            :return: Generated data
            :rtype: pd.DataFrame
        """
        data = self.get_data()
        predicted_matrix = np.zeros(data.shape)
        residual_matrix = np.zeros(data.shape)
        for x in list(data.index.values):
            for i, y in enumerate(list(data.columns.values)):
                row = data.loc[[x]].drop(y, axis=1)
                predicted_matrix[x, i] = self.models[i].predict(row)
                residual_matrix[x, i] = (predicted_matrix[x,i] - data.loc[x, y])**2
        var_vector = np.mean(residual_matrix, axis=0)
        for i in range(predicted_matrix.shape[0]):
            row = predicted_matrix[i, :]
            for j, y in enumerate(list(data.columns.values)):
                self.gen_data.at[i, y] = row[j] + np.random.normal(loc=0, scale=np.sqrt(var_vector[j]))

        return self.gen_data

    def generate_main(self, input_dir, basename):
        """ Generate synthetic data and returns numpy matrix
            :return: Numpy matrix
        """
        X = self.generate()
        return AutoML.from_df(input_dir, basename, X, y=None)




data = AutoML('./boston_housing_automl/', 'boston_housing')
rf = RF_generator(data)
rf.fit()
rf.generate_main('boston_anm_' + str(20) + '_automl/', 'boston_anm_' + str(20))