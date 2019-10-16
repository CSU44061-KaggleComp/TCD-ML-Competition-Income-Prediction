# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import randint as sp_randint
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


def main():
    dataset = pd.read_csv('/kaggle/input/tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv')
    dataset2 = pd.read_csv('/kaggle/input/tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv')

    dataset = dataset.drop_duplicates()
    #replace NaN values
    dataset = replaceNaN(dataset)
    dataset2 = replaceNaN(dataset2)
   
    x = dataset.drop(columns=['Income in EUR','Instance']) 
    y = dataset['Income in EUR']

    #split data
    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=0)
    
    df_train = x_train.join(y_train)
    df_test =  x_test.join(y_test)
    
    all_data = pd.concat((df_train,df_test,dataset2))

    for column in all_data.select_dtypes(include=[np.object]).columns:
        df_train[column] = df_train[column].astype(CategoricalDtype(categories = all_data[column].unique()))
        df_test[column] = df_test[column].astype(CategoricalDtype(categories = all_data[column].unique()))
        dataset2[column] = dataset2[column].astype(CategoricalDtype(categories = all_data[column].unique()))

    x_train = df_train.drop(["Income in EUR"], axis=1)
    x_test = df_test.drop(["Income in EUR"], axis=1)
    
    #encode categorical variables
    x_train = encodeCat(x_train) 
    x_test = encodeCat(x_test) 
    
    feature_scaler = StandardScaler(with_mean=False)
    x_train = feature_scaler.fit_transform(x_train)
    x_test = feature_scaler.transform(x_test)
    
    #causes memory error
    #poly = PolynomialFeatures(degree=2)
    #x_train = poly.fit_transform(x_train)
    #x_test = poly.transform(x_train)
    
    pipeline = Pipeline([
         ('lasso', Lasso(normalize=True))
    ])
    
    param_dist = {'lasso__alpha':[0.02, 0.05]}
    
    model =  RandomizedSearchCV(pipeline, param_distributions=param_dist 
                              ,verbose=10,cv=2).fit(x_train, y_train).best_estimator_
    
    y_pred = model.predict(x_test)
    
    #I was getting many negative values as predictions to I mapped them to 0, 
    #as log tranformations of predictions was not successfull
    y_pred_len = y_pred.shape[0]
    for i in range(y_pred_len):
        if y_pred[i]<0:
            y_pred[i] = 0
    
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    submission(dataset2,model,feature_scaler)
    
def submission(dataset,model,feature_scaler):
    y = dataset['Income']
    #encode categorical variables
    x = dataset.drop(columns=['Income','Instance'])
    x = encodeCat(x)
    x = feature_scaler.transform(x)

    y_pred = model.predict(x)
    
    y_pred_len = y_pred.shape[0]
    for i in range(y_pred_len):
        if y_pred[i]<0:
            y_pred[i] = 0

    submis = pd.read_csv('/kaggle/input/tcdml1920-income-ind/tcd ml 2019-20 income prediction submission file.csv')
    submis['Income'] = y_pred
    submis.head()

    filename = 'Income Predictions.csv'
    submis.to_csv(filename,index=False)
    print('Saved file: ' + filename)
    

def replaceNaN(dataset):
    dataset['Gender'] = dataset['Gender'].fillna(dataset['Gender'].mode()[0])
    dataset['University Degree'] = dataset['University Degree'].fillna(dataset['University Degree'].mode()[0])
    dataset['Profession'] = dataset['Profession'].fillna(dataset['Profession'].mode()[0])
    dataset['Hair Color'] = dataset['Hair Color'].fillna(dataset['Hair Color'].mode()[0])
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
    dataset['Year of Record'] = dataset['Year of Record'].fillna(dataset['Year of Record'].median())
    return dataset

def encodeCat(X):
    X = pd.get_dummies(X,prefix_sep='_', drop_first=True)
    return X

if __name__ == '__main__':
  main()