import os
import sys
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score


def save_object(file_path,object):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            dill.dump(object,file_obj)

    except Exception as e:
        raise CustomException(e,sys)   



def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for model_name,model in models.items():

            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)

            test_model_score=accuracy_score(y_test,y_pred)
            report[model_name]=(test_model_score,model)
        
        logging.info('Evalution completed')
        return report

    except Exception as e:
        raise CustomException(e,sys) 


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return  dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)    
                  



              