# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:15:09 2018

@author: Marco Ambuludi
"""

# -*- coding: utf-8 -*-

import csv, operator
import numpy as np
#import scipy xc

import pylab
import scipy.cluster.hierarchy as sch
from scipy import spatial
from sklearn import metrics
from timeit import timeit
def open_archivo(nombre):
    entrada = open(nombre) # Abrir archivo csv
    csv_ent = csv.reader(entrada, delimiter=';') # Leer todos los registros
    lista=[] # Leer registro (lista)
    for row in csv_ent: # Leer campos por separado (variables)
        lista.append(row)
    return lista
        
abrir = open_archivo('D:\TESIS\dataset\Datos\small.csv');
y=[]
X=[]
   

for i in range(len(abrir)):
    X.append(abrir[i][4:40])
    y.append(abrir[i][41])
X = np.asarray(X, dtype='f8') #convertimos los valores no categoricos a numericos 
y=np.array(y)
y[y=='normal']=1
y[y=='anomaly']=0
y = np.array(y, dtype='f8') #convertimos los valores no categoricos a numericos 


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
K = 2

# Calcula y grafica el dendrograma
import os  # para el os.path.basename
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import fcluster
plt.figure('Dendogram') 
plt.title('Dendrograma de Clustering Jerarquico NSL-KDD')
plt.xlabel('Instancias-Cluster')
plt.ylabel('Distancia')



Y = sch.linkage(D, method='complete') #clustering jerárquico con distancia max entre clusters (Farthest Point Algorithm)
Z0 = fcluster(Y, K, 'maxclust') # Agrupamiento en K Grupos de acuerdo a los tresholds: 1.40346354 Dieciseis; 1.40684625 Catorce grupos; 1.41475747 Trece; 1.4162164 Doce; 1.42089949 Doce 
label_pred = Z0 # Resultados de los Agrupamientos

# Calculo las etiquetas
labels=list('' for i in range(len(D)))
for i in range(len(D)):
    labels[i]=str(i+1)+ ',' + str(Z0[i])

# Calculo el color_treshold
ct=Y[-(K-1),2]  

# Dendograma de la matriz linkage
#Z1 = sch.dendrogram(Y,labels=labels, color_threshold=ct) 
Z1 = sch.dendrogram(Y, color_threshold=ct) 


# Calculo de el orden de los clústers (cluster, documento)
labels=list('' for i in range(len(D)))
for i in range(len(D)):
    labels[i]= str(Z0[i])+ ',' + str(i+1)
    
print "Los clusters son (cluster, objeto): \n", np.sort(np.array(labels))