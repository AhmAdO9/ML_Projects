import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle



def  save_object(file_path, obj):
                try:
                    dir_path = os.path.dirname(file_path)
                    
                    os.makedirs(dir_path, exist_ok=True)

                    with open(file_path, 'wb') as file:
                        pickle.dump(obj, file)

                except Exception  as e:
                    raise CustomException("Some error occurred in Utils", sys)


def evaluate_model(x_train, x_test, y_train, y_test, models, Params):
    try:
        
       
        report = {}
        for model in list(models):
            grid_search = GridSearchCV(
                            estimator=models[model], 
                            param_grid=Params[model], 
                            cv=5,
                            scoring='neg_mean_squared_error')

            grid_search.fit(x_train, y_train)

            best_model = grid_search.best_estimator_

            y_test_pred = best_model.predict(x_test)
           
            test_model_score = r2_score(y_test, y_test_pred)
           
            report[model] = test_model_score

            logging.info(f"Training for {model} completed")
        
        return report, best_model


    except Exception as e:
        raise CustomException("Some Error occurred in utils", sys)


def load_object(file_path):

    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        raise CustomException(e, sys)