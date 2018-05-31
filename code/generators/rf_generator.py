# Imports
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
problem_dir = 'code/auto_ml'  
from sys import path
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
        self.ds.process_data() # todo: optimize
        
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
      
        
    def generate(self, p=0.8):
        """ 
            Generate examples by copying data and then do values imputations
            
            :param p: the probability of changing a value
                        if p=0, the generated dataset will be equals to the original
                        if p=1, the generated dataset will contains only new values
            
            :return: Generated data
            :rtype: pd.DataFrame
        """
        data = self.get_data()
        
        for x in list(data.index.values):
            for i, y in enumerate(list(data.columns.values)):
            
                if np.random.random() < p:
                    row = data.loc[[x]].drop(y, axis=1)
                    self.gen_data.at[x, y] = self.models[i].predict(row)
        
        return self.gen_data
    

    def partial_fit_generate(self, p=0.8, **kwargs):
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
        data = self.get_data()
        
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
                    self.gen_data.at[x, y] = model.predict(row)
                    
        return self.gen_data
    
    
    def generate_to_automl(self, input_dir, basename, p=0.8, partial=False, **kwargs):
        """ Generate a DataFrame and save it in automl format
            
            :param input_dir: Input directory
            :param basename: AutoML basename
            :param p: Probability of replacement
            :param partial: Normal or partial fit
            :param kwargs: Random forest argument for partial fit case
            :return: AutoML object
        """
        if partial:
            X = self.partial_fit_generate(p=p, **kwargs)
        else:
            X = self.generate(p=p)
        return AutoML.from_df(input_dir, basename, X, y=None)
