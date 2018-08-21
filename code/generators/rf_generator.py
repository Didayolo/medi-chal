# Imports
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
problem_dir = '../auto_ml'
from sys import path
path.append(problem_dir)
from auto_ml import AutoML

class RF_generator():
    def __init__(self, ds, processed=False, **kwargs):
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
        #self.ds.process_data(**kwargs) # todo: optimize
        
        # Generated DataFrame
        self.gen_data = self.ds.get_data(processed=processed).copy()
    
    
    def process_data(self, **kwargs):
        """ Apply process_data method on ds
        """
        self.ds.process_data(**kwargs)
        #self.gen_data = self.ds.get_data(processed=True).copy()
    
    
    def get_data(self, processed=False):
        return self.ds.get_data('X', processed=processed)
    
    
    def fit(self, processed=False, **kwargs):
        """ 
            Fit one random forest for each column, given the others
            :param kwargs: Random Forest parameters
        """ 
        data = self.get_data(processed=processed)

        for i in range(len(data.columns)):
            # May bug with duplicate names in columns
            y = data[data.columns[i]]
            X = data.drop(data.columns[i], axis=1) 
            
            # Regressor or classifier
            if self.ds.feat_type[i] == 'Numerical':
                model = self.regressor(**kwargs)
            else:
                model = self.classifier(**kwargs)
            
            model.fit(X, y)
            self.models.append(model)
      
        
    def generate(self, p=0.8, processed=False):
        """ 
            Generate examples by copying data and then do values imputations
            
            :param p: the probability of changing a value
                        if p=0, the generated dataset will be equals to the original
                        if p=1, the generated dataset will contains only new values
            
            :return: Generated data
            :rtype: pd.DataFrame
        """
        data = self.get_data(processed=processed)
        
        for x in list(data.index.values):
            for i, y in enumerate(list(data.columns.values)):
            
                if np.random.random() < p:
                    row = data.loc[[x]].drop(y, axis=1)
                    # WARNING
                    #self.gen_data.at[x, y] = self.models[i].predict(row)
                    
                    # DEBUG
                    prediction = self.models[i].predict(row)
                    if isinstance(prediction, np.ndarray):
                        self.gen_data.at[x, y] = prediction[0]
                    else:
                        self.gen_data.at[x, y] = prediction
        
        return self.gen_data
    

    def partial_fit_generate(self, p=0.8, processed=False, **kwargs):
        """
            Fit and generate for high dimensional case.
            To avoid memory error, features are trained and generated one by one.
            
            :param p: The probability of changing a value
                        if p=0, the generated dataset will be equals to the original
                        if p=1, the generated dataset will contains only new values
            :param kwargs: Random Forest parameters

            :return: Generated data
            :rtype: pd.DataFrame
        """
        data = self.get_data(processed=processed)
        
        # Features are trained and generated one by one 
        for i in range(len(data.columns)):
            # May bug with duplicate names in columns
            y = data.columns[i] # name
            Y = data[y]         # data
            X = data.drop(data.columns[i], axis=1) 
            
            # Regressor or classifier
            if self.ds.feat_type[i] == 'Numerical':
                model = self.regressor(**kwargs)
            else:
                model = self.classifier(**kwargs)
            
            # FIT    
            model.fit(X, Y)
            
            # GENERATE
            for x in list(data.index.values): # rows
            
                if np.random.random() < p:
                    row = data.loc[[x]].drop(y, axis=1)
                    
                    # DEBUG
                    prediction = model.predict(row)
                    if isinstance(prediction, np.ndarray):
                        self.gen_data.at[x, y] = prediction[0]
                    else:
                        self.gen_data.at[x, y] = prediction
                    
        return self.gen_data
    
    
    def generate_to_automl(self, input_dir, basename, p=0.8, partial=False, processed=False, **kwargs):
        """ Generate a DataFrame and save it in autoML format
            
            :param input_dir: Input directory
            :param basename: AutoML basename
            :param p: Probability of replacement
            :param partial: Normal or partial fit
            :param kwargs: Random forest argument for partial fit case
            :return: AutoML object
        """
        if partial:
            X = self.partial_fit_generate(p=p, processed=processed, **kwargs)
        else:
            X = self.generate(p=p, processed=processed)
        return AutoML.from_df(input_dir, basename, X, y=None)
        
        
    def sample(self, n=1, p=0.8, processed=False):
        """ Generate n rows

            :param n: Number of examples to sample
            :param p: The probability of changing a value
                        if p=0, the generated dataset will be equals to the original
                        if p=1, the generated dataset will contains only new values

            :return: Generated data
            :rtype: pd.DataFrame
        """
        data = self.get_data(processed=processed)
        
        # Sampling with replacement
        data = data.sample(n=n, replace=True)
        gen_data = data.copy()
        
        for x in list(data.index.values):
            for i, y in enumerate(list(data.columns.values)):
            
                if np.random.random() < p:
                    row = data.loc[[x]].drop(y, axis=1)

                    # DEBUG
                    prediction = self.models[i].predict(row)
                    if isinstance(prediction, np.ndarray):
                        gen_data.at[x, y] = prediction[0]
                    else:
                        gen_data.at[x, y] = prediction
        
        return gen_data.reset_index()
        
        
    def sample_to_automl(self, input_dir, basename, n=1, p=0.8, processed=False):
        """ Sample n rows and save it in autoML format
            
            :param input_dir: Input directory
            :param basename: AutoML basename
            :param n: Number of examples to sample
            :param p: Probability of replacement
            :return: AutoML object
        """
        X = self.sample(n=n, p=p, processed=processed)
        return AutoML.from_df(input_dir, basename, X, y=None)
        
