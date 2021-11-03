import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def get_df():
    head = ['NAME_CONTRACT_TYPE',
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'CNT_CHILDREN',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'AMT_GOODS_PRICE',
    'NAME_TYPE_SUITE', # we need to add this
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'REGION_POPULATION_RELATIVE',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'OWN_CAR_AGE',
    'FLAG_MOBIL',
    'FLAG_EMP_PHONE',
    'FLAG_WORK_PHONE',
    'FLAG_CONT_MOBILE',
    'FLAG_PHONE',
    'FLAG_EMAIL',
    'TARGET']
    df = pd.read_csv('data/application_data.csv')[head].replace("XNA", np.nan).dropna(subset=['CODE_GENDER'])
    df['CODE_GENDER'] = df['CODE_GENDER'].map( {'M':1, 'F':0} )
    df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map( {'Y':1, 'N':0} )
    df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map( {'Y':1, 'N':0} )
    df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].map( {'Cash loans':1, 'Revolving loans':0} )

    df['OWN_CAR_AGE'].fillna(df['OWN_CAR_AGE'].mean(), inplace=True)
    df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].mean(), inplace=True)
    df['AMT_ANNUITY'].fillna(df['AMT_ANNUITY'].mean(), inplace=True)

    df['NAME_TYPE_SUITE'] = pd.get_dummies(df['NAME_TYPE_SUITE'].fillna(df['NAME_TYPE_SUITE'].mode()[0]))
    df['NAME_INCOME_TYPE'] = pd.get_dummies(df['NAME_INCOME_TYPE'])
    df['NAME_EDUCATION_TYPE'] = pd.get_dummies(df['NAME_EDUCATION_TYPE'])
    df['NAME_FAMILY_STATUS'] = pd.get_dummies(df['NAME_FAMILY_STATUS'])
    df['NAME_HOUSING_TYPE'] = pd.get_dummies(df['NAME_HOUSING_TYPE'])


    return df


def get_info(y_true, y_pred):
    # print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
    print('Confusion Matrix - Training Dataset')
    print(pd.crosstab(y_true, y_pred, rownames = ['True'], colnames = ['Predicted'], margins = True))
    print(classification_report(y_true, y_pred))
    
