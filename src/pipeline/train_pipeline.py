import sys
from src.exception import CustomException
from src.logger import logging
import os

import pandas as pd
from src.utils import load_object
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion

class TrainePipeline:
    def __init__(self,raw_data_path:str=None):
        self.raw_data_path:str='notebook/data/adult.csv'

    def train_entire_pipeline(self):
        try:
            logging.info('Enter the training pipeline')
            data_ingestion=DataIngestion(self.raw_data_path)
            raw_data_path=data_ingestion.initiate_data_ingestion()

            data_clean=DataCleaning(raw_data_path)
            train_path,test_path=data_clean.initiate_data_cleaning()

            data_tranform=DataTransformation()
            train_arr,test_arr,path=data_tranform.initiate_data_transformation(train_path,test_path)
        
            trainer_=ModelTrainer()
            best_score,best_model_name=trainer_.initiate_model_trainer(train_arr,test_arr)    
            logging.info(
                "The training completed with the best model[{best_model_name}:{best_score}]"
                .format(best_model_name=best_model_name, best_score=best_score)
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    train_pipeline=TrainePipeline()
    train_pipeline.train_entire_pipeline()
           