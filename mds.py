# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 16:30:40 2018

@author: Marco Ambuludi
"""

# -*- coding: utf-8 -*-

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Realizado por: Diego Vallejo
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import scipy
import pylab
import scipy.cluster.hierarchy as sch
from math import*
from scipy import spatial
from sklearn import metrics
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data
y = iris.target
# debemos poner diff.class


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
np.savetxt("matriz_distancias_iris.csv", D, delimiter=",")

###########################################################################
# Escalamiento Multidimensional
import os  # para el os.path.basename
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
plt.figure('Multidimensional Scaling')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Escalamiento Multidimensional Dataset IRIS')
MDS()
# realiza el escalamiento multidimensional (en un plano de dos dimensiones)
# "precomputed" porque se usa la matriz de distancias
# se especifica `random_state` para que el plot sea reproducible
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(D)  # forma (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]

docs = []
for i in range(len(D)): # lista con indices de los documentos ordenados
    docs.append(i+1)

plt.plot(xs, ys, 'ro')
for Xmds, Ymds, Zmds in zip(xs, ys, docs): # añade los titulos de cada documento 
    plt.annotate('{}'.format(Zmds), xy=(Xmds,Ymds), xytext=(-12, 12), ha='right', va = 'bottom',
                textcoords='offset points', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops=dict(arrowstyle='->', shrinkA=0))
