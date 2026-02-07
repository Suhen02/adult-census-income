import os
import sys
from src.exception import CustomException
from src.logger import logging

import numpy as np 
import pandas as pd
from dataclasses import dataclass
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation import DataTransformation


@dataclass 
class DataIngestionConfig:
    raw_data_path:str=os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entred the data ingestion method')
        try:
            df=pd.read_csv('notebook/data/adult.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Raw data saved!')

            return self.ingestion_config.raw_data_path
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    raw_data_path=obj.initiate_data_ingestion()
    data_clean=DataCleaning(raw_data_path)
    train_path,test_path=data_clean.initiate_data_cleaning()
    data_tranform=DataTransformation()
    train_arr,test_arr,path=data_tranform.initiate_data_transformation(train_path,test_path)
    print(path)
    

            