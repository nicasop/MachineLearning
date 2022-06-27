#### Integrantes:Diego Bedoya - Jhon Granda - Anthony Grijalva - Sebastian Sandoval - Alexis Villaviencio ####
#### LIBRERIAS ####
import numpy as np
import nlp
import jaccard
import tdIdf as tdf
import cosenoVectorial as cosV
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score,silhouette_samples
from validclust import dunn

#FUNCIONES
def plot_dendrogram(model, **kwargs):
    '''
    Esta función extrae la información de un modelo AgglomerativeClustering
    y representa su dendograma con la función dendogram de scipy.cluster.hierarchy
    '''
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot
    sch.dendrogram(linkage_matrix, **kwargs)

##### MATRIZ DE DISTANCIAS #####
##IMPORTAR DATOS
papers = nlp.importarCSV('papers.csv')
titulos = papers['title'].tolist()
keywords = papers['keywords'].tolist()
abstracts = papers['abstract'].tolist()
##NLP
titulosL = nlp.limpiarDocumento(titulos,'en')
keywordsL = nlp.limpiarDocumento(keywords,'en')
abstractsL = nlp.limpiarDocumento(abstracts,'en')
##JACCARD
#TITULOS
matrizTit = jaccard.matrizJaccard(titulosL)
# print('Matriz de distancias de los Títulos\n',matrizTit)
#KEYWORDS
matrizKey = jaccard.matrizJaccard(keywordsL)
# print('Matriz de distancias de las Keywords\n',matrizKey)
##FULL INVERTED INDEX -- TF-IDF -- COSENO VECTORIAL
#FULL INVERTED INDEX
diccionario={'tokens':[],'ocurrencias':[]}
diccionario['tokens']= nlp.indexacionToken(abstractsL)
diccionario['ocurrencias'] = nlp.ocurrencias(diccionario['tokens'],abstractsL)
#TF-IDF
matriz = tdf.bagWords(diccionario,abstractsL)
wtf = tdf.matrizPTF(matriz)
dF = tdf.documentF(wtf)
idf = tdf.IDF(dF,len(abstractsL))
tf_idf = tdf.TFIDF(wtf,idf)
#Coseno Vectorial
matrizN = cosV.matrizNormal(tf_idf)
matrizAbs = cosV.matrizDistacias(matrizN)
# print('Matriz de distancias de los Abstracts\n',matrizAbs)
#MATRIZ DE DISTACIAS
ponderacion = [0.5,0.3,0.2]
Distancias = cosV.matrizDistanciaPonderada(matrizAbs,matrizKey,matrizTit,ponderacion)
# dis1 = 1-Distancias
# print('Matriz de distancias\n',Distancias)

#DHC
num_grupos = 5
hc = AgglomerativeClustering(n_clusters=num_grupos, 
                        affinity = 'precomputed', 
                        linkage = 'complete',
                        compute_distances=True)
clusters_dhc = hc.fit(Distancias)

##Graficar Dendograma
# altura_corte = 0.2
# plot_dendrogram(clusters_dhc,color_threshold=altura_corte)
# plt.title("Dendograma")
# plt.axhline(y=altura_corte, c = 'black', linestyle='--')
# plt.show()

# np.fill_diagonal(Distancias,0)
# print(Distancias)
dist = 1 - Distancias
# print(dist)

##Validación Interna
print('----------------INDICES DE VALIDACIÓN INTERNA----------------')
##DUNN
dunn_dhc = dunn(dist,clusters_dhc.labels_)
print('Indice de DUNN DHC: ',dunn_dhc,'\n')

#Coeficiente de Siluheta
si1_dhc = silhouette_score(dist, clusters_dhc.labels_, metric='precomputed')#coefiente de silueta a partir de la matriz de distancias
print('Coeficiente de Siluheta de: ',si1_dhc)

# si2_dhc=silhouette_samples(Distancias,clusters_dhc.labels_,metric='precomputed')
# fig,ax = plt.subplots(1)
# fig.set_size_inches(8.8, 5)
# ax.set_xlim([-0.1, 1])
# ax.set_ylim([0, len(abstractsL) + (num_grupos + 1) * 10])

# y_lower = 10
# for i in range(num_grupos):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#   ith_cluster_silhouette_values = si2_dhc[clusters_dhc.labels_ == i]      
#   ith_cluster_silhouette_values.sort()

#   size_cluster_i = ith_cluster_silhouette_values.shape[0]
#   y_upper = y_lower + size_cluster_i

#   color = cm.nipy_spectral(float(i) / num_grupos)
#   ax.fill_betweenx(
#       np.arange(y_lower, y_upper),
#       0,
#       ith_cluster_silhouette_values,
#       facecolor=color,
#       edgecolor=color,
#       alpha=0.7,
#       )

#         # Label the silhouette plots with their cluster numbers at the middle
#   ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#         # Compute the new y_lower for next plot
#   y_lower = y_upper + 10  # 10 for the 0 samples

#   ax.set_title("The silhouette plot for the various clusters.")
#   ax.set_xlabel("The silhouette coefficient values")
#   ax.set_ylabel("Cluster label")

#     # The vertical line for average silhouette score of all the values
#   ax.axvline(x=si1_dhc, color="red", linestyle="--")
#   ax.set_yticks([])  # Clear the yaxis labels / ticks
#   ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
# ax.legend(("Average Silhouette: "+str(round(si1_dhc,3)),),bbox_to_anchor=(1.1,1.07))
# plt.show()