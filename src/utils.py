import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj) :
    try :
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as fp:
            dill.dump(obj,fp)
        
    except Exception as e :
        raise CustomException(e,sys)
    
def evaluate_model(X_train, X_test, y_train, y_test, models, params):

    try:

        report = {}
        trained_models = {}

        for model_name, model in models.items():

            gs = GridSearchCV(
                estimator=model,
                param_grid=params[model_name],
                cv=3,
                scoring='r2',
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)

            report[model_name] = r2_score(y_test, y_test_pred)

            trained_models[model_name] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as fp:
            return dill.load(fp)
        
    except Exception as e :
        raise CustomException(e,sys)
    