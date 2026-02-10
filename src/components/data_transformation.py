import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object
from sklearn.impute import SimpleImputer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info('enterd the data_transfomer object method')

            numerical_columns=[
                'age', 
                'education.num', 
                'capital.gain', 
                'capital.loss', 'hours.per.week'] 
            categorical_columns=[
                'workclass', 
                'education', 
                'marital.status',
                'occupation', 
                'relationship', 
                'sex', 'native.country']       
            
            num_pipeline=Pipeline(
                steps=[
                    ('impute',SimpleImputer()),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('one_hot_encoding',OneHotEncoder(handle_unknown='ignore')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info('Enter the data transformation')  

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            print(f"[dataloaded] shape1:{train_df.shape} and shape2:{test_df.shape}")
            logging.info('Data read successfully')

            preprocessor=self.get_data_transformer_object()

            target_column='income'

            input_features_train_df=train_df.drop(columns=[target_column]) 
            target_feature_train_df=np.array(train_df[target_column]).reshape(-1,1)

            input_features_test_df=test_df.drop(columns=[target_column])
            target_feature_test_df=np.array(test_df[target_column]).reshape(-1,1)

            print(f"[saparate target] shape1:{input_features_train_df.shape} and shape2:{input_features_test_df.shape}")

            train_feature_arr=preprocessor.fit_transform(input_features_train_df)
            test_feature_arr=preprocessor.fit_transform(input_features_test_df)

            logging.info('Preprocessing completed')
            print(f"[after preprocessing] shape1:{train_feature_arr.shape} and shape2:{test_feature_arr.shape}")

            train_feature_arr = train_feature_arr.toarray()
            test_feature_arr=test_feature_arr.toarray()

            #print(f"before shape1:{train_feature_arr.s)} and shape2:{test_feature_arr}")
 
            train_arr=np.concat([train_feature_arr,target_feature_train_df],axis=1)
            test_arr=np.concat([test_feature_arr,target_feature_test_df],axis=1)

            

            logging.info('Data is concated')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                object=preprocessor
            )
            logging.info('object saved')
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)