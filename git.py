# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 09:08:43 2018

@author: Marco Ambuludi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  matplotlib.pyplot  as plt
import matplotlib

names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","target"]
df = pd.read_csv('D:\TESIS\dataset\marcosinprimera.csv', names=names, delimiter=';')

df.shape #derecho
df.head()# derecho
df.describe()#deercho


#borrando caracteristicas redundates

df['num_outbound_cmds'].value_counts()
df.drop('num_outbound_cmds', axis=1, inplace=True)
df['is_host_login'].value_counts()
df.drop('is_host_login', axis=1, inplace=True)

#trasnformacion a caracteristicas categoricas
df['protocol_type'] = df['protocol_type'].astype('category')
df['service'] = df['service'].astype('category')
df['flag'] = df['flag'].astype('category')
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)


#primer grafico univariate histogramms

params = {'axes.titlesize':'28',
          'xtick.labelsize':'24',
          'ytick.labelsize':'24'}
matplotlib.rcParams.update(params)
df.hist(figsize=(50, 30), bins=20)
plt.show()

#ESTANDARIZACION KDD

data=df.values
X=data[:,0:39]
X
from sklearn.preprocessing import StandardScaler
sScaler = StandardScaler()
rescaleX = sScaler.fit_transform(X)
rescaleX
names_inputed = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files",
    "is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"]
df_rescaled = pd.DataFrame(data=rescaleX, columns=names_inputed)
df_rescaled.hist(figsize=(50, 30), bins=20)
plt.show()

#NORMALIZACION KDD

from sklearn.preprocessing import Normalizer
norm = Normalizer()
xNormalize = norm.fit_transform(X)

xNormalize#DERECHA
df_Normalized = pd.DataFrame(data=xNormalize, columns=names_inputed)

df_Normalized.hist(figsize=(50, 30), bins=20)
plt.show()

#ENCODING
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
df['target'] = df['target'].astype('category')
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

data=df.values
Y = data[:,39]
#RECURSIVE FEATURE SELECION
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe = RFE(LogisticRegression(), n_features_to_select=4)
rfe.fit(X, Y)