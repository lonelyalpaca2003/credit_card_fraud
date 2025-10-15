import pickle
import os

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
import xgboost as xgb
import mlflow



mlflow.set_experiment("fraud-detection-2")

def read_filename(path = '../data/creditcard.csv'):
    df = pd.read_csv(path)
    return df



def split_train_test_datasets(df):
    x = df.drop(['Class'], axis = 1)
    y = df['Class']
    x_train, x_val , y_train, y_val = train_test_split(x,y,train_size = 0.8, random_state = 42, stratify = y)

    return x_train, x_val, y_train, y_val


def train_model(x_train, y_train, x_val, y_val):
    with mlflow.start_run():
        train = xgb.DMatrix(data = x_train, label = y_train)
        valid = xgb.DMatrix(data = x_val, label = y_val)
        best_params = {'reg_lambda' :0.05036863565098369,
                        'gamma' : 0.3391370082694971, 'seed' : 42, 'max_depth' :19,
                        'min_child_weight' : 5,
                        'learning_rate' :0.1286049132579215,
                         'objective' : 'binary:hinge',
                        'colsample_bytree' : 0.6965119068062329, 'reg_alpha': 0.16585451812037608,
                        'subsample' : 0.7561490535729092
                        }
        mlflow.log_params(best_params)
        booster = xgb.train(
            params = best_params, 
            dtrain = train, 
            evals=[(train, "train"), (valid, "validation")],
            num_boosting_rounds = 227,
            early_stopping_rounds = 50,
            verbose_eval = False
        )
        y_pred_probs = booster.predict(valid)
        y_pred = (y_pred_probs > 0.5).astype(int)
        f1 = f1_score(y_val, y_pred)
        mlflow.log_metric('f1', f1)
        mlflow.xgboost.log_model(booster, name = "xgb_mlflow")


def main(path = '../data/creditcard.csv'):
    df = read_filename(path)
    x_train, x_val, y_train, y_val = split_train_test_datasets(df)
    train_model(x_train, x_val, y_train, y_val)


if __name__ == '__main__':
    main()

