### Alexis Villavicencio ###
##solo guiate si esta medio diferente

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score,adjusted_mutual_info_score,normalized_mutual_info_score
from sklearn.metrics import davies_bouldin_score,silhouette_score,silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#pip install validclust ##instalar la libreria validclust
from validclust import dunn

k = 3
matriz_D = np.array([[1,0.23,0.78,0.04,0.69,0.15,0.56],
                     [0.23,1,0.11,0.98,0.35,0.22,0.45],
                     [0.78,0.11,1,0.77,0.63,0.35,0.01],
                     [0.04,0.98,0.77,1,0.12,0.85,0.89],
                     [0.69,0.35,0.63,0.12,1,0.08,0.96],
                     [0.15,0.22,0.35,0.85,0.08,1,0.56],
                     [0.56,0.45,0.01,0.89,0.96,0.56,1]])
dist = 1 - matriz_D

print('MATRIZ DE DISTANCIAS\n',matriz_D)

# a) Ingrese dicha matriz en un algoritmo de aprendizaje no supervisado e indique los resultados de las instancias etiquetadas 
# cuando k=3. (1 Pto.)

hc = AgglomerativeClustering(n_clusters=k, 
                        affinity = 'precomputed', 
                        linkage = 'complete')

clusters_dhc = hc.fit_predict(dist)
print('\nRESULTADO DEL ETIQUETADO\n',clusters_dhc)

# b) Determine la cantidad ideal de clusters que deben formarse (justifique su respuesta numérica y gráficamente). (1 Pto.).

# c) Tomando en cuenta, los resultados del literal a), calcule el Índice de Dunn (1 Pto.)
dunn = dunn(dist,clusters_dhc)
print('\nINDICE DE DUNN\n',dunn)

# d) Tomando en cuenta, los resultados del literal a), calcule el coeficiente de 
# silueta de cada grupo (1 Pto.)
si_dhc = silhouette_score(dist, clusters_dhc, metric='precomputed')#coefiente de silueta a partir de la matriz de distancias
print('\nCOEFICIENTE DE SILUETA\n',si_dhc)

si1_dhc=silhouette_samples(dist,clusters_dhc,metric='precomputed')

fig,ax = plt.subplots(1)
fig.set_size_inches(8.8, 5)
ax.set_xlim([-0.1, 1])
ax.set_ylim([0, len(matriz_D) + (k + 1) * 10])

y_lower = 10
for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
  ith_cluster_silhouette_values = si1_dhc[clusters_dhc == i]      
  ith_cluster_silhouette_values.sort()

  size_cluster_i = ith_cluster_silhouette_values.shape[0]
  y_upper = y_lower + size_cluster_i

  color = cm.nipy_spectral(float(i) / k)
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

  ax.set_title("The silhouette plot.")
  ax.set_xlabel("The silhouette coefficient values")
  ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
  ax.axvline(x=si_dhc, color="red", linestyle="--")
  ax.axvline(x=0, color="black", linestyle="-")
  ax.set_yticks([])  # Clear the yaxis labels / ticks
  ax.set_xticks([-0.2, 0, 0.2, 0.4,0.6,0.8,1])
ax.legend(("Average Silhouette: "+str(round(si_dhc,3)),),bbox_to_anchor=(1.1,1.1))
plt.show()

# e) Considerando que el ground truth del dataset es el siguiente: (1, 1, 2, 3, 1, 3, 2). Calcular tres índices de validación 
# externa. (1.5 Ptos.)
print('\nVALIDACION EXTERNA')
ground_truth = [1, 1, 2, 3, 1, 3, 2]
##ARI
ari_dhc = adjusted_rand_score(ground_truth,clusters_dhc)
print("ARI DHC: ",ari_dhc)
##AMI
ami_dhc = adjusted_mutual_info_score(ground_truth,clusters_dhc)
print("AMI DHC: ",ami_dhc)
##NMI
nmi_dhc = normalized_mutual_info_score(ground_truth,clusters_dhc)
print("NMI DHC: ",nmi_dhc)