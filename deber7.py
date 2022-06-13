#### Integrantes:Diego Bedoya - Jhon Granda - Anthony Grijalva - Sebastian Sandoval - Alexis Villaviencio ####

import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from yellowbrick.cluster import KElbowVisualizer

iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
Y = iris.target
num_grupos = len(np.unique(Y))
distancias = euclidean_distances(X)#matriz de distancias
# print(distancias)
#MDS
mds = MDS(n_components=2,
            dissimilarity='precomputed',
            metric=True,
            max_iter=500,
            random_state=0) #normal con 10
x_trans = mds.fit_transform(distancias)

#KMEANS
kmeans = KMeans(n_clusters=num_grupos,
                random_state=0,
                max_iter=500) 
x_kmeans = kmeans.fit(X)
clusters_K = x_kmeans.labels_

##DHC
lin = sch.linkage(distancias, method='complete')#matriz
hc = AgglomerativeClustering(n_clusters=num_grupos, 
                        affinity = 'euclidean', 
                        linkage = 'complete')
clusters_dhc = hc.fit_predict(X)

#ELBOW
kModel = KMeans(random_state=0,max_iter=500) 
elbow = KElbowVisualizer(kModel,k=(1,10),timings=False)
elbow.fit(X)

##Grafico
#MDS
plt.figure(figsize=(17,6)) #tamanio de la figura
plt.subplot(1,3,1)
sns.scatterplot(x=x_trans[:,0],y=x_trans[:,1],hue=Y,palette=["green","red","blue"])#graficoMDS
plt.title('Grafica de los grupos con MDS')

#KMEANS
plt.subplot(1,3,2)
sns.scatterplot(x=x_trans[:,0],y=x_trans[:,1],hue=clusters_K,palette=["green","red","blue"])#graficoKMEANS
plt.title('Grafica de los grupos con el algoritmo de KMEANS')

#DHC
plt.subplot(1,3,3)
sch.dendrogram(lin)
plt.title('Dendograma')
plt.show()

#ELBOW
elbow.show()