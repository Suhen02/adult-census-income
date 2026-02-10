import os
import sys
from src.logger import logging
from src.exception import CustomException

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import evaluate_model,save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifact','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('split the training and test data')

            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )        

            models={
            'RandomForestClassifier':RandomForestClassifier(),
            'LogisticRegression':LogisticRegression(),
            'AdaBoostClassifier':AdaBoostClassifier(),
            'DecisionTreeClassifier':DecisionTreeClassifier(),
            'CatBoostClassifier':CatBoostClassifier()
            }

            model_report:dict=evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )
            
            logging.info('Models report fetched succesfully!')
            best_model_name,(best_score,best_model)=max(
                model_report.items(),
                key=lambda item:item[1][0]
            )

            print('Best model:',best_model_name)
            print('Best score:',best_score)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )

            return [best_score,best_model_name]

        except Exception as e:
            raise CustomException(e,sys)  
            

