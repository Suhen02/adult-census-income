import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataCleaningConfig:
    train_data_path:str=os.path.join('artifact','train.csv')
    test_data_path:str=os.path.join('artifact','test.csv')

class DataCleaning:
    def __init__(self,raw_data_path):
        self.data_cleaning_config=DataCleaningConfig()
        self.raw_data_path=raw_data_path

    def initiate_data_cleaning(self):
        logging.info('Enter the data cleaning')
        try:
            logging.info('data loading')
            df=pd.read_csv(self.raw_data_path) 

            df=df[df['occupation'].values!='?'] 
            df.drop_duplicates()
            df.drop(columns=['fnlwgt','race',],inplace=True)

            df['income']=[1 if i=='>50K' else 0 for i in df['income']]

            majority_data=df[df.income==0]
            minority_data=df[df.income==1]
            majority_data=majority_data.sample(n=8000,random_state=42)
            balanced_data=pd.concat([majority_data,minority_data])

            logging.info('Data splitting initiated')

            train_set,test_set=train_test_split(balanced_data,test_size=0.2,random_state=42)

            os.makedirs(os.path.dirname(self.data_cleaning_config.train_data_path),exist_ok=True)
            train_set.to_csv(self.data_cleaning_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_cleaning_config.test_data_path,header=True,index=False)

            logging.info('Data cleaning completed!')

            return (
                self.data_cleaning_config.train_data_path,
                self.data_cleaning_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        



