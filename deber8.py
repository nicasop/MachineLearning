#### Integrantes:Diego Bedoya - Jhon Granda - Anthony Grijalva - Sebastian Sandoval - Alexis Villaviencio ####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score,adjusted_mutual_info_score,normalized_mutual_info_score
from sklearn.metrics import davies_bouldin_score,silhouette_score,silhouette_samples
#pip install validclust ##instalar la libreria validclust
from validclust import dunn

iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
Y = iris.target
num_grupos = len(np.unique(Y))
distancias = euclidean_distances(X)#matriz de distancias

#KMEANS
kmeans = KMeans(n_clusters=num_grupos,
                random_state=0,
                max_iter=500) 
x_kmeans = kmeans.fit(X)
clusters_K = x_kmeans.labels_

##DHC
hc = AgglomerativeClustering(n_clusters=num_grupos, 
                        affinity = 'euclidean', 
                        linkage = 'complete')
clusters_dhc = hc.fit_predict(X)

####Validacion Interna
print('----------------INDICES DE VALIDACIÓN INTERNA----------------')
##DUNN
#KMEANS
dunn_k = dunn(distancias,clusters_K)
print('Indice de DUNN KMEANS: ',dunn_k)
#DHC
dunn_dhc = dunn(distancias,clusters_dhc)
print('Indice de DUNN DHC: ',dunn_dhc,'\n')

##Silhouette
#KMEANS
si_k = silhouette_score(distancias, clusters_K, metric='precomputed')#coefiente de silueta a partir de la matriz de distancias
print('Coeficiente de Siluheta de KMEANS',si_k)
si2_k=silhouette_samples(distancias,clusters_K,metric='precomputed')

fig,ax = plt.subplots(1)
fig.set_size_inches(8.8, 5)
ax.set_xlim([-0.1, 1])
ax.set_ylim([0, len(X) + (num_grupos + 1) * 10])

y_lower = 10
for i in range(num_grupos):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
  ith_cluster_silhouette_values = si2_k[clusters_K == i]      
  ith_cluster_silhouette_values.sort()

  size_cluster_i = ith_cluster_silhouette_values.shape[0]
  y_upper = y_lower + size_cluster_i

  color = cm.nipy_spectral(float(i) / num_grupos)
  ax.fill_betweenx(
      np.arange(y_lower, y_upper),
      0,
      ith_cluster_silhouette_values,
      facecolor=color,
      edgecolor=color,
      alpha=0.7,
      )

        # Label the silhouette plots with their cluster numbers at the middle
  ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
  y_lower = y_upper + 10  # 10 for the 0 samples

  ax.set_title("The silhouette plot for KMEANS.")
  ax.set_xlabel("The silhouette coefficient values")
  ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
  ax.axvline(x=si_k, color="red", linestyle="--")
  ax.set_yticks([])  # Clear the yaxis labels / ticks
  ax.set_xticks([-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
ax.legend(("Average Silhouette: "+str(round(si_k,3)),),bbox_to_anchor=(1.1,1.1))

#DHC
si_dhc = silhouette_score(distancias, clusters_dhc, metric='precomputed')#coefiente de silueta a partir de la matriz de distancias
print('Coeficiente de Siluheta de DHC',si_dhc,'\n')
si2_dhc=silhouette_samples(distancias,clusters_dhc,metric='precomputed')

fig,ax = plt.subplots(1)
fig.set_size_inches(8.8, 5)
ax.set_xlim([-0.1, 1])
ax.set_ylim([0, len(X) + (num_grupos + 1) * 10])

y_lower = 10
for i in range(num_grupos):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
  ith_cluster_silhouette_values = si2_dhc[clusters_dhc == i]      
  ith_cluster_silhouette_values.sort()

  size_cluster_i = ith_cluster_silhouette_values.shape[0]
  y_upper = y_lower + size_cluster_i

  color = cm.nipy_spectral(float(i) / num_grupos)
  ax.fill_betweenx(
      np.arange(y_lower, y_upper),
      0,
      ith_cluster_silhouette_values,
      facecolor=color,
      edgecolor=color,
      alpha=0.7,
      )

        # Label the silhouette plots with their cluster numbers at the middle
  ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
  y_lower = y_upper + 10  # 10 for the 0 samples

  ax.set_title("The silhouette plot for DHC.")
  ax.set_xlabel("The silhouette coefficient values")
  ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
  ax.axvline(x=si_dhc, color="red", linestyle="--")
  ax.set_yticks([])  # Clear the yaxis labels / ticks
  ax.set_xticks([-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
ax.legend(("Average Silhouette: "+str(round(si_dhc,3)),),bbox_to_anchor=(1.1,1.1))
plt.show()

####Validacion Externa
print('----------------INDICES DE VALIDACIÓN EXTERNA----------------')
##ARI
#Kmeans
ari_k = adjusted_rand_score(Y,clusters_K)
print("ARI KMEANS: ",ari_k)
#DHC
ari_dhc = adjusted_rand_score(Y,clusters_dhc)
print("ARI DHC: ",ari_dhc,'\n')

##AMI
ami_k = adjusted_mutual_info_score(Y,clusters_K)
print("AMI KMEANS: ",ami_k)
#DHC
ami_dhc = adjusted_mutual_info_score(Y,clusters_dhc)
print("AMI DHC: ",ami_dhc,'\n')

##NMI
nmi_k = normalized_mutual_info_score(Y,clusters_K)
print("NMI KMEANS: ",nmi_k)
#DHC
nmi_dhc = normalized_mutual_info_score(Y,clusters_dhc)
print("NMI DHC: ",nmi_dhc,'\n')