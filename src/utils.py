import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import dill

def  save_object(
                file_path, obj):
                
                try:

                    dir_path = os.path.dirname(file_path)
                    
                    os.makedirs(dir_path)

                    with open(file_path, 'wb') as file:
                        dill.dump(obj, file)


                except Exception  as e:
                    raise CustomException("Some error occurred in Utils", sys)