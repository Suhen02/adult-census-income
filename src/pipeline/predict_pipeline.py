import sys
from src.exception import CustomException
from src.logger import logging
import os

import pandas as pd
from src.utils import load_object

class PredictPipeline:
    def  __init__(self):
        pass
    def predicted(self,features):
        try:
            model_path='artifact\model.pkl'
            preprocessor_path='artifact\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
            age:int,
            education_no:int,
            capital_gain:int, 
            capital_loss:int,
            hours_week:int,
            workclass:str,
            education:str,
            marital_status:str,
            occupation:str,
            relationship:str,
            sex:str,
            native_country:str        
    ):
            
        self.age=age
        self.education_no=education_no
        self.capital_gain=capital_gain
        self.capital_loss=capital_loss
        self.hours_week=hours_week
        self.workclass=workclass
        self.education=education
        self.marital_status=marital_status
        self.occupation=occupation
        self.relationship=relationship
        self.sex=sex
        self.native_country=native_country

    def get_data_as_data_frame(self):
        try:
            custom_data_dict={
                'age':[self.age],
                'education.num':[self.education_no],
                'capital.gain':[self.capital_gain],
                'capital.loss':[self.capital_loss],
                'hours.per.week':[self.hours_week],
                'workclass':[self.workclass],
                'education':[self.education],
                'marital.status':[self.marital_status],
                'occupation':[self.occupation],
                'relationship':[self.relationship],
                'sex':[self.sex],
                'native.country':[self.native_country]
            } 

            return pd.DataFrame(custom_data_dict) 
        except Exception as e:
            raise CustomException(e,sys)  
