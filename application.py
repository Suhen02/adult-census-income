from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
        age=int(request.form.get('age')),
        education_no=int(request.form.get('education.num')),
        capital_gain=int(request.form.get('capital.gain')),
        capital_loss=int(request.form.get('capital.loss')),
        hours_week=int(request.form.get('hours.per.week')),
        workclass=request.form.get('workclass'),
        education=request.form.get('education'),
        marital_status=request.form.get('marital.status'),
        occupation=request.form.get('occupation'),
        relationship=request.form.get('relationship'),  # fixed spelling
        sex=request.form.get('sex'),
        native_country=request.form.get('native.country')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predicted(pred_df)
        print(f'result:{result}')
        return render_template('home.html',results=result[0])
    
if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)
