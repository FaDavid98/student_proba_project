import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def load(file_name):
    df=pd.read_csv(file_name,delimiter=';')
    return df

def convert(df):
    '''Converting nominal and numerical features'''
    le = preprocessing.LabelEncoder()
    df = df.apply(lambda col: le.fit_transform(col), axis=0)
    return df

def descending_class(df):
    df['G3'] = np.where(df['G3'].between(0,7), 0, df['G3'])
    df['G3'] = np.where(df['G3'].between(8,14), 1, df['G3'])
    df['G3'] = np.where(df['G3'].between(15,20), 2, df['G3'])
    

    df['G1'] = np.where(df['G1'].between(0,7), 0, df['G1'])
    df['G1'] = np.where(df['G1'].between(8,14), 1, df['G1'])
    df['G1'] = np.where(df['G1'].between(15,20), 2, df['G1'])
    

    df['G2'] = np.where(df['G2'].between(0,7), 0, df['G2'])
    df['G2'] = np.where(df['G2'].between(8,14), 1, df['G2'])
    df['G2'] = np.where(df['G2'].between(15,20), 2, df['G2'])
    return df

def selection(df):
    x=df.iloc[:,0:32]
    y=df.iloc[:,-1]
    return x,y


def split(x,y,test_size):
    XTraining, XTest, YTraining, YTest = train_test_split(x, y, test_size=test_size)
    # oversample = RandomOverSampler()
    # XTraining, YTraining = oversample.fit_resample(XTraining, YTraining) #cuz of imbalanced classes
    return XTraining, XTest, YTraining, YTest

def normalization(x,XTraining, XTest):
    names=x.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    XTraining = pd.DataFrame(min_max_scaler.fit_transform(XTraining), columns=names)
    XTest = pd.DataFrame(min_max_scaler.fit_transform(XTest), columns=names)
    return XTraining, XTest