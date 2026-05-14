import numpy as np 
import pandas as pd
import os

#train_data = pd.read_csv('./data/raw/train_data.csv')
#test_data = pd.read_csv('./data/raw/test_data.csv')
def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath}:{e}')


def fill_missing_with_median(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                median_values=df[column].median()
                df[column]=df[column].fillna(median_values)
        return df
    except Exception as e:
        raise Exception(f'Error filling missing values with median: {e}')

def save_data(df:pd.DataFrame, filepath:str)->None:
    try:
        return df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f'Error loading data : {e}')

def main():
    try:
        data_raw_path = './data/raw'
        processed_data_path = './data/processed'

        os.makedirs(processed_data_path)

        train_data = load_data(os.path.join(data_raw_path, 'train_data.csv'))
        test_data = load_data(os.path.join(data_raw_path, 'test_data.csv'))

        processed_train_data = fill_missing_with_median(train_data)
        processed_test_data = fill_missing_with_median(test_data)

        save_data(processed_train_data, os.path.join(processed_data_path, 'train_processed_median.csv'))
        save_data(processed_test_data, os.path.join(processed_data_path, 'test_processed_median.csv'))
    except Exception as e:
        raise Exception(f'Error occured : {e}')
if __name__=="__main__":
    main()

#processed_train_data = fill_missing_with_median(train_data)
#processed_test_data = fill_missing_with_median(test_data)

#data_path = os.path.join('data','processed')
#os.makedirs(data_path, exist_ok=True)

#processed_train_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index = False)
#processed_test_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
    