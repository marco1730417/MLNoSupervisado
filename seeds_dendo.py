# -*- coding: utf-8 -*-

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Realizado por: Diego Vallejo
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import csv, operator
import numpy as np
import scipy
import pylab
import scipy.cluster.hierarchy as sch
from scipy import spatial
from sklearn import metrics

def open_archivo(nombre):
    entrada = open(nombre) # Abrir archivo csv
    csv_ent = csv.reader(entrada, delimiter=',') # Leer todos los registros
    lista=[] # Leer registro (lista)
    for row in csv_ent: # Leer campos por separado (variables)
        lista.append(row)
    return lista

abrir = open_archivo('D:\TESIS\dataset\NUEVA\KDDTestcsv.csv')
y = []
X = []

for i in range(len(abrir)):
    y.append(abrir[i][42])
    X.append(abrir[i][5:9])

y = np.asarray(y, dtype='i4')
X = np.array(X, dtype='f4')

def cos_v(f1,f2):
    result = spatial.distance.cosine(f1,f2); #calcula la distancia entre dos vectores (1-cos(v1,v2))
    return result

def mat_dist_cos(lista2):
    dist=np.zeros((len(lista2),len(lista2))) #matriz de distancias inicializada a ceros
    for j in range(len(lista2)-1):
        for i in range (j, len(lista2)):
            if i!=j:
                dist[i][j]=cos_v(lista2[j],lista2[i])
                dist[j][i]=dist[i][j]
    return dist #matriz de distancia coseno calculada

mat_fin = mat_dist_cos(X)
D = mat_dist_cos(X)
K = 3

###########################################################################
#Exportar matriz de distancia en CSV
np.savetxt("matriz_distancias_seeds.csv", D, delimiter=",")

