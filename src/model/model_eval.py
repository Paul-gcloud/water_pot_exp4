import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

#test_data = pd.read_csv('./data/processed/test_processed.csv')
def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data {filepath}:{e} ')


#X_test = test_data.drop(columns=['Potability'])
#y_test = test_data['Potability']
def prep_data(data:pd.DataFrame)->tuple[pd.DataFrame, pd.Series]:
    try:
        X=data.drop(columns=['Potability'])
        y=data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f'Error preparing data : {e}')

#model = pickle.load(open('model.pkl','rb'))
def load_model(filepath:str):
    try:
        with open(filepath, 'rb')as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f'Error loading model from {filepath}:{e}')

def model_eval(model, X_test:pd.DataFrame, y_test:pd.Series)->dict:
    try:
        y_pred = model.predict(X_test)

        acc=accuracy_score(y_test, y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test, y_pred)
        f1score=f1_score(y_test,y_pred)

        metrics_dict ={
            'acc':acc,
            'precison':precision,
            'recall':recall,
            'f1score':f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f'Error evaluating model : {e}')

def save_metrics(metrics_dict:dict, filepath:str)->None:
    try:
        with open('reports/metrics.json', 'w') as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        raise Exception(f'Error saving metrics to {filepath}:{e}')
    
def main():
    try:
        data_path = './data/processed/test_processed_mean.csv'
        model_path = './models/model.pkl'
        metrics_path = 'reports/metrics.json'

        test_data = load_data(data_path)
        X_test, y_test = prep_data(test_data)
        model = load_model(model_path)
        metrics = model_eval(model, X_test, y_test)

        save_metrics(metrics, metrics_path)
    except Exception as e:
        raise Exception(f"error occured")
if __name__=="__main__":
    main()