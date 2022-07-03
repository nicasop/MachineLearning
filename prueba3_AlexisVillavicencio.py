# ALEXIS VILLAVICENCIO

import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score,adjusted_mutual_info_score,normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import davies_bouldin_score,silhouette_score
#pip install validclust ##instalar la libreria validclust
from validclust import dunn

# FUNCIONES
def importarCSV(nombre_file):
    return pd.read_csv(nombre_file)

# 1. Realizar el agrupamiento mediante dos algoritmos diferentes. Para cada caso indicar la etiqueta que correspondería al grupo respectivo, e.g. (0 = VIP, 2 = VIP Potencial, 1 = Nuevos, 3 = Baja Frecuencia…) (4 puntos).

# DATOS

datos = importarCSV('segmentation_data.csv')
num_grupos = 4
dataset = datos.iloc[:,1:4]
print(dataset)
distancias = euclidean_distances(dataset)#matriz de distancias

# MODELOS

##KMEANS
kmeans = KMeans(n_clusters=num_grupos,
                random_state=0,
                max_iter=500) 

##DHC
hc = AgglomerativeClustering(n_clusters=num_grupos, 
                        affinity = 'precomputed', 
                        linkage = 'complete')

##Predicciones
clusters_k = kmeans.fit_predict(dataset)
clusters_dhc = hc.fit_predict(distancias)

print(clusters_k)
print(clusters_dhc)

num_grupos = np.unique(clusters_k,return_counts=True)
num_grupos1 = np.unique(clusters_dhc,return_counts=True)
print('KMEANS')
for i in range(len(num_grupos[0])):
    print(num_grupos[0][i],'---',num_grupos[1][i])

print('\nDHC')
for i in range(len(num_grupos1[0])):
    print(num_grupos1[0][i],'---',num_grupos1[1][i])

# #Etiquetado de los resultados obtenidos
# #Considerando las caracteristicas descritas de cada grupo acerca de los valores de frequency, recency y monetary. El etiquetado de los clusters se va a distribuir de la siguiente manera
# #El grupo con el menor número de instancias va a ser el grupo de Clientes Vip con el etiqueta CVip ya que considero que la tienda tiene muy pocos clientes vip comparados con el resto de clientes.
# #El grupo que le sigue en el número de instancias va a ser el grupo de Vips Potenciales con la etiqueta PVip ya que considero que este grupo debe tener mas instancias que el grupo vip
# #El grupo que le sigue serian los clientes Nuevos con la etiqueta CNuevos ya que este grupo va a tener mas instancias que los dos anterios pero considero que no tantas como el último grupo ya que el último grupo que es el de clientes con baja frecuencia con la etiqueta BAFrecuencia engloblan clientes que ya no van a ir nunca más o que van muy derepente.

##Tomando en cuenta la explicación anterior y el número de instancias en cada cluster el etiquetado seria el siguiente

## etiquetado KMEANS
clus_K = pd.DataFrame(clusters_k)
clus_K[0][clus_K[0]==0] = 'BAFrecuencia'
clus_K[0][clus_K[0]==1] = 'PVip'
clus_K[0][clus_K[0]==2] = 'CVip'
clus_K[0][clus_K[0]==3] = 'CNuevos'

clus_K = np.array(clus_K[0])
print(clus_K)

##  etiquetado DHC
clus_dhc = pd.DataFrame(clusters_dhc)
clus_dhc[0][clus_dhc[0]==0] = 'CNuevos'
clus_dhc[0][clus_dhc[0]==1] = 'PVip'
clus_dhc[0][clus_dhc[0]==2] = 'CVip'
clus_dhc[0][clus_dhc[0]==3] = 'BAFrecuencia'

clus_dhc = np.array(clus_dhc[0])
print(clus_dhc)

resultadoKMEANS = pd.DataFrame({'clientes':datos['client'],'grupo':clus_K})
resultadoDHC = pd.DataFrame({'clientes':datos['client'],'grupo':clus_dhc})

print('KMEANS')
print(resultadoKMEANS)
print('DHC')
print(resultadoDHC)

# 2. Comparar el rendimiento de los resultados del literal anterior mediante TODAS las métricas que conoce (2 puntos).

print('-------- RENDIMIENTO DEL ALGORITMO DE KMEANS -------')
#KMEANS
dunn_k = dunn(distancias,clus_K)
print('Indice de DUNN KMEANS: ',dunn_k)
si1_dhc = silhouette_score(distancias, clus_K, metric='precomputed')#coefiente de silueta a partir de la matriz de distancias
print('Coeficiente de Siluheta de KMEANS:',si1_dhc)

print('\n-------- RENDIMIENTO DEL ALGORITMO DE DHC -------')
#DHC
dunn_dhc = dunn(distancias,clus_dhc)
print('Indice de DUNN DHC: ',dunn_dhc)
si1_dhc = silhouette_score(distancias, clus_dhc, metric='precomputed')#coefiente de silueta a partir de la matriz de distancias
print('Coeficiente de Siluheta de DHC:',si1_dhc)

## No se pudo realizar la validación externa de cada uno de los algoritmos (kmeans y dhc) ya que no se disponia del ground truth

# 3. Tomando en cuenta como “ground truth” el resultado del primer algoritmo, indique qué tan bueno es el rendimiento del segundo algoritmo (2 puntos).

## el resultado del algoritmo de kmeans se tomó como ground truth y el de dhc como la predicción
print('-------- RENDIMIENTO DE LA PREDICCIÓN ---------')
groundTruth = clusters_k
prediction = clusters_dhc
##ARI
ari_dhc = adjusted_rand_score(groundTruth,prediction)
print("ARI DHC: ",ari_dhc)

##AMI
ami_dhc = adjusted_mutual_info_score(groundTruth,prediction)
print("AMI DHC: ",ami_dhc)

##NMI
nmi_dhc = normalized_mutual_info_score(groundTruth,prediction)
print("NMI DHC: ",nmi_dhc)

##Como se puede observar los valores obtenidos estan cercanos a cero por tanto se concluye que el algoritmo de predicción( en este caso DHC) no dio un buen rendimiento


