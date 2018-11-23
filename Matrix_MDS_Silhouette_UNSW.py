import csv, operator
import numpy as np
#import scipy xc

import pylab
import scipy.cluster.hierarchy as sch
from scipy import spatial
from sklearn import metrics

def open_archivo(nombre):
    entrada = open(nombre) # Abrir archivo csv
    csv_ent = csv.reader(entrada, delimiter=';') # Leer todos los registros
    lista=[] # Leer registro (lista)
    for row in csv_ent: # Leer campos por separado (variables)
        lista.append(row)
    return lista
        
abrir = open_archivo('C:\Users\Marco Ambuludi\Dropbox\Tesis Marco\DATASET\UNSWprueba.csv');
y=[]
X=[]
   

for i in range(len(abrir)):
    X.append(abrir[i][4:41])
    y.append(abrir[i][43])
X = np.asarray(X, dtype='f8') #convertimos los valores no categoricos a numericos 
y=np.array(y)
y[y=='Normal']=0
y[y=='anomaly']=1

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
###########################################################################
#Exportar matriz de distancia en CSV
np.savetxt("D:\TESIS\dataset\UNSW\matrizdistanciasunsw.csv", D, delimiter=",")
###########################################################################
# Escalamiento Multidimensional DATASET MARCO
import os  # para el os.path.basename
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import MDS
plt.figure('Multidimensional Scaling')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
fig, ax = plt.subplots()
x=0
ax.plot(x, label="Normal",color='b')
ax.plot(x, label="Anormal",color='r')
ax.legend();
plt.title('Escalamiento Multidimensional Dataset UNSW')
MDS()
# realiza el escalamiento multidimensional (en un plano de dos dimensiones)
# "precomputed" porque se usa la matriz de distancias
# se especifica `random_state` para que el plot sea reproducible
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(D)  # forma (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
docs = []
for i in range(len(y)): # lista con indices de los documentos ordenados
    docs.append(y[i])
plt.scatter(xs,ys,marker='o',s=30,lw=0,alpha=2,c=y,cmap=pylab.cm.bwr)
plt.colorbar();
#plt.show()
# https://matplotlib.org/examples/color/colormaps_reference.html

###########################################################################
# Calcula y grafica el dendrograma
from scipy.cluster.hierarchy import fcluster
plt.figure('Dendogram') 
plt.title('Dendrograma de Clustering Jerarquico UNSW')
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

###########################################################################
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
range_n_clusters = [K]

# Crea un subplot con 1 fila y 2 columnas
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)

# El 1er subplot es el plot de la Silueta
# Coeficientes de Silueta con rango entre -1, 1
ax1.set_xlim([-1, 1])

# El (n_clusters+1)*10 es para insertar un espacio en blanco entre la silueta 
# y los clusters individuales, para demarcarlos claramente
n_clusters = K
ax1.set_ylim([0, len(D) + (n_clusters + 1) * 10])

# Vector de Indices de los documentos y sus grupos
cluster_labels = Z0-1


#############################################################################
# Calculo de la silueta

a=np.zeros(len(D))
for k in range(K):
    indices = [i for i, x in enumerate(list(cluster_labels)) if x == k]
    for j in range(len(indices)):
        a[indices[j]]=(np.average(D[indices,indices[j]]))

b=np.zeros(len(D))
#Calculo de los centroides de los grupos
b1=list()
for k in range(K):
    indices1 = [i for i, x in enumerate(list(cluster_labels)) if x == k]
    A=D[indices1,:].sum(axis=0)
    b1.append(indices1[np.argmin(A[indices1])])

b1=np.array(b1)
#Busqueda del grupo mas cercano
b2=list()
for k in range(K):
    lista=range(K)
    del lista[k]
    if np.argmin(D[b1[k],b1[lista]])>=k:
        b2.append(np.argmin(D[b1[k],b1[lista]])+1)
    else:
        b2.append(np.argmin(D[b1[k],b1[lista]]))
#Promedio de la distancia de cada elemento de un grupo respecto a los elementos
# del grupo mas cercano        
for j in range(K):
    indices1 = [i for i, x in enumerate(list(cluster_labels)) if x == j]
    indices2 = [i for i, x in enumerate(list(cluster_labels)) if x == b2[j]]
    A=D[indices1,:]
    b[indices1]=np.average(A[:, indices2], axis=1)
    
#calculo de la silueta
S=np.zeros(len(D))
for i in range(len(D)):
    S[i]=(b[i]-a[i])/max(a[i],b[i])    
    
########################################################################## 

# El silhouette_score da el valor promedio para todos los documentos
# Esto da una perspectiva dentro de la densidad y la separacion de los clusters formados
#silhouette_avg = silhouette_score(pos, cluster_labels) #OJO: USO LOS PUNTOS DEL MDS (POS)
silhouette_avg = np.average(S)
# Calcula los valores de la silueta para cada documento
#sample_silhouette_values = silhouette_samples(pos, cluster_labels) #OJO: USO LOS PUNTOS DEL MDS (POS)
sample_silhouette_values=S
print("Para n_clusters =", n_clusters, "El Valor Promedio de la Silueta es:", silhouette_avg)

y_lower = 10
for i in range(n_clusters):
    # Se agrega los valores de silueta para los documentos que pertenecen al 
    # cluster i, y se los ordena
    ith_cluster_silhouette_values = \
        sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

#DESDE AQUI ES EL PRIMER PLOT  QUE DICE GRAFICO DE SILUETA
    ##########################################
    ##################################3
    # CAMBIAR COLORES LUEGO DE cm.
    color = plt.cm.brg(float(i) / n_clusters)
#   cm._reverse_cmap_spec(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
 # Etiquetas de los clusters en las siluetas 
    ax1.text(-0.9, y_lower + 0.5 * size_cluster_i, str(i+1))
 # Calculo el nuevo y_lower para el siguiente plot  
    y_lower = y_upper + 10  # 10 para los 0 muestras

ax1.set_title("Grafico de Silueta")
ax1.set_xlabel("Valores de los Coeficientes de Silueta")
ax1.set_ylabel("Clusters")
# Linea vertical para marcar el valor medio de la silueta para todos los valores
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.set_yticks([])  # Limpia los ejes de y
ax1.set_xticks([-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

# El 2do Plot muestran los clusters formados 
#SEGUNDO PLOT QUE DICE OBJETOS AGRUPADOS
#para cambiar el color es donde esta RdLgn
#####################################
#######################################
#################################3
colors = plt.cm.brg(cluster_labels.astype(float) / n_clusters)


ax2.scatter(xs, ys, marker='o', s=30, lw=0, alpha=0.7, c=colors)
ax2.set_title("Objetos Agrupados UNSW-NB15")
ax2.set_xlabel("Dimension 1")
ax2.set_ylabel("Dimension 2")
plt.suptitle(("Analisis de Silueta con n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
plt.show()



###########################################################################
# Graficar la MATRIZ DE DISTANCIAS
fig = pylab.figure('Distances Matrix',figsize=(10,7)) #tamaño del gráfico
axmatrix = fig.add_axes([0.1,0.1,0.76,0.8]) #posiciones de la matriz en el grafico
idx1 = np.sort(Z1['leaves']) #indices de los documentos ordenados (solución IRIS)
D = D[idx1,:]
D = D[:,idx1]
# a continuacion es el grafico Matriz de Distancias (azul) si queremos cambiar es el cmap
########################################
#############################################
im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu) # YlGnBu Escala de Azules; YlOrRd Escala de Rojos; YlGn Escala de Verdes
axmatrix.set_xticks([])
axmatrix.set_yticks([])

# ELIMINAMOS LOS EJES DE ABAJO Y DE LA DERECHA POR QUE SALIAN FEOS COMO UNA BARRA NEGRA
# Graficar los valores de los ejes
#axmatrix.set_xticks(range(len(D)))
#axmatrix.set_xticklabels(idx1+1, minor=False)
#axmatrix.xaxis.set_label_position('bottom')
#axmatrix.xaxis.tick_bottom()
#pylab.xticks(rotation=0, fontsize=8)
#axmatrix.set_yticks(range(len(D)))
#axmatrix.set_yticklabels(idx1+1, minor=False)
#axmatrix.yaxis.set_label_position('left')
#axmatrix.yaxis.tick_left()
## FIN COMENTARIOS

axmatrix.set_title('Matriz de Distancias del Dataset UNSW')

# Grafica la Barra de Color
axcolor = fig.add_axes([0.87,0.1,0.02,0.8])
pylab.colorbar(im, cax=axcolor)

###########################################################################
# Calcula y grafica el primer dendrograma
fig = pylab.figure('Distances Matrix and Dendogram',figsize=(8,8)) #tamaño del gráfico
ax1 = fig.add_axes([0.09,0.1,0.2,0.6]) #posiciones del dendograma en el grafico
Z2 = sch.dendrogram(Y, orientation='right',color_threshold=ct) #dendograma de la matriz linkage
ax1.set_xticks([])
ax1.set_yticks([])

# Calcula y grafica el segundo dendrograma
ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
Z3 = sch.dendrogram(Y,color_threshold=ct)
ax2.set_xticks([])
ax2.set_yticks([])

# Graficar la matriz de distancias
axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
idx1 = np.array(Z2['leaves'])
#D = D[idx1,:]
#D = D[:,idx1]
im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu) # YlGnBu Escala de Azules; YlOrRd Escala de Rojos; YlGn Escala de Verdes
axmatrix.set_xticks([])
axmatrix.set_yticks([])

# Graficar los valores de los ejes
#axmatrix.set_xticks(range(len(D)))
#axmatrix.set_xticklabels(idx1+1, minor=False)
#axmatrix.xaxis.set_label_position('bottom')
#axmatrix.xaxis.tick_bottom()
#pylab.xticks(rotation=-90, fontsize=8)
#axmatrix.set_yticks(range(len(D)))
#axmatrix.set_yticklabels(idx1+1, minor=False)
#axmatrix.yaxis.set_label_position('right')
#axmatrix.yaxis.tick_right()

# Grafica la Barra de Color
axcolor = fig.add_axes([0.92,0.1,0.02,0.6])
pylab.colorbar(im, cax=axcolor)

###########################################################################
# Tamaño de los Grupos
from collections import Counter
size_real = dict(Counter(y+1))
size_calc = dict(Counter(label_pred))
print "Tamano de los Grupos Real: \n", size_real
print "Tamano de los Grupos Calculado: \n", size_calc

###########################################################################
# Normalized Mutual Information(NMI) y Adjusted Mutual Information(AMI). 
labels_true = y+1
labels_pred = label_pred
ARIid = metrics.adjusted_rand_score(labels_true, labels_pred)
AMIid = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
NMIid = metrics.normalized_mutual_info_score(labels_true, labels_pred)  
print "ARI: \n", ARIid
print "AMI: \n", AMIid
print "NMI: \n", NMIid
