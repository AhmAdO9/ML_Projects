import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts/model_trainer', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info('splitting the train and test data')

            x_train, x_test, y_train, y_test = (train_array[:, :-1], 
                                                test_array[:, :-1], 
                                                train_array[:, -1], 
                                                test_array[:, -1])
        
            models = {

                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'CatBoosting Regressor': CatBoostRegressor(verbose=False),
                'AdaBoost Regressor': AdaBoostRegressor()
            }

            Params = { 
                "Random Forest" : 
                {   'n_estimators': [50, 100, 200],  # Number of trees in the forest
                    'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
                    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
                    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
                    'max_features': ['sqrt', 'log2', None],  # Number of features to consider when looking for the best split
                    'bootstrap': [True, False]  # Whether to use bootstrap samples or not
                },

            "Decision Tree" : 
                {
                    'criterion': ['squared_error', 'friedman_mse','poisson'],  # Function to measure the quality of a split
                    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
                    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
                    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
                    'max_features': [None, 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
                    'splitter': ['best', 'random']  # Strategy used to split at each node
                },
            
            "Gradient Boosting" : 
                {
                    'n_estimators': [50, 100, 200],  # Number of boosting stages to be used
                    'learning_rate': [0.01, 0.1, 0.5],  # Step size at each iteration
                    'max_depth': [3, 5, 7],  # Maximum depth of the individual trees
                    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
                    'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for fitting each base learner
                    'loss': ['quantile', 'squared_error', 'huber']  # Loss function to be minimized
                },
            
            "Linear Regression" : 
                {
                    'fit_intercept': [True, False],  # Whether to calculate the intercept for this model
                },

            "K-Neighbors Regressor" : 
                {
                    'n_neighbors': [3, 5, 7, 10],  # Number of neighbors to use
                    'weights': ['uniform', 'distance'],  # Weight function used in prediction
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
                    'leaf_size': [10, 20, 30],  # Leaf size passed to BallTree or KDTree
                    'p': [1, 2]  # Power parameter for the Minkowski distance (1 = Manhattan, 2 = Euclidean)
                },

            "CatBoosting Regressor" : 
                {
                    'iterations': [100, 200, 300],  # Number of boosting iterations
                    'learning_rate': [0.01, 0.1, 0.5],  # Learning rate
                    'depth': [6, 8, 10],  # Depth of the trees
                    'l2_leaf_reg': [1, 3, 5],  # L2 regularization coefficient
                    'border_count': [32, 64],  # Number of splits for numerical features
                    'thread_count': [4, 8],  # Number of threads to be used for training
                },

            "AdaBoost Regressor" : 
                {
                    'n_estimators': [50, 100, 200],  # Number of boosting stages
                    'learning_rate': [0.01, 0.1, 1.0]  # Weight applied to each classifier at each stage
                }
            }
            

            model_report, best_model =evaluate_model(x_train, x_test, y_train, y_test, models, Params)

            logging.info(model_report)

            logging.info('Model training Complete')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info('Model saved Complete')


            
        except Exception as e:
            raise CustomException("Some error occurred in model trainer", sys)
        pass





