import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(dataset='adult', verbose=False):
    if dataset=='adult':
        x, y = load_adult(verbose)
    return x, y

def load_adult(verbose=False):
    df = pd.read_csv("data/adult/adult.data",1,",")
    df.columns = ["age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hpw", "country", "income"]

    #change income to 0/1
    income_map={' <=50K':0,' >50K':1}
    df['income']=df['income'].map(income_map).astype(int)

    #drop capital gain and loss as mostly 0s
    df.drop("capital-gain", axis=1,  inplace=True)
    df.drop("capital-loss", axis=1,  inplace=True)

    #save target variable and drop from dataframe
    y=df['income']
    df.drop("income", axis=1,  inplace=True)

    #convert target variable to dummies
    categorical_columns=["workclass", "education",
    "marital-status", "occupation", "relationship", "race", "gender", "country"]
    df = pd.get_dummies(df, columns=categorical_columns)

    scaler = MinMaxScaler()
    scale_cols=["age", "fnlwgt", "education-num", "hpw"]
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    #print some info
    if verbose==True:
        print('Loaded Adult Dataset')
        print('Datasize:', df.shape[0])
        print('No. Features:', df.shape[1])

    return df, y

def test_split(x, y):

    random_num = np.random.RandomState(1300)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random_num,)

    return x_train, x_test, y_train, y_test
