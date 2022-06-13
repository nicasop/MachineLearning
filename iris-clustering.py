import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

def importarCSV(nombre_file):
    return pd.read_csv(nombre_file)

iris = datasets.load_iris()
X = iris.data[:,0:4]
Y = iris.target
# print(X)
# print(Y)
# iris = importarCSV('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
# iris = importarCSV('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
distancias = euclidean_distances(X)

#MDS
mds = MDS(n_components=2,
            dissimilarity='precomputed',
            metric=True,
            max_iter=500,
            random_state=0) #normal con 10
x_trans = mds.fit_transform(distancias)

#KMEANS
kmeans = KMeans(n_clusters=3,
                random_state=0,
                max_iter=500) 

x_kmeans = kmeans.fit(X)
clusters = x_kmeans.labels_
#Grafico
plt.figure(figsize=(15,15)) #tamanio de la figura
plt.subplot(2,1,1)
sns.scatterplot(x=x_trans[:,0],y=x_trans[:,1],hue=Y,palette=["green","red","blue"])
plt.title('Grafica de los grupos con MDS')

plt.subplot(2,1,2)
sns.scatterplot(x=x_trans[:,0],y=x_trans[:,1],hue=clusters,palette=["green","red","blue"])
plt.title('Grafica de los grupos con el algoritmo de KMEANS')
plt.show()

#DHC
hc = AgglomerativeClustering(n_clusters = 3, 
                        affinity = 'euclidean', 
                        linkage = 'complete')
y_hc = hc.fit_predict(distancias)

lin = sch.linkage(distancias, method='complete')
plt.figure(figsize=(15,15)) #tamanio de la figura
sch.dendrogram(lin)
plt.show()


# print(iris.iloc[:,4])
# fig = iris[iris.variety == 'Setosa'].plot(kind='scatter', x='sepal.length', y='sepal.width', color='blue', label='Setosa')
# iris[iris.variety == 'Versicolor'].plot(kind='scatter', x='sepal.length', y='sepal.width', color='green', label='Versicolor', ax=fig)
# iris[iris.variety == 'Virginica'].plot(kind='scatter', x='sepal.length', y='sepal.width', color='red', label='Virginica', ax=fig)
# fig.set_xlabel('Pétalo - Longitud')
# fig.set_ylabel('Pétalo - Ancho')
# fig.set_title('Pétalo Longitud vs Ancho')
# plt.show()