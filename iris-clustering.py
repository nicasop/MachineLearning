import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

def importarCSV(nombre_file):
    return pd.read_csv(nombre_file)

iris = importarCSV('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
# iris = importarCSV('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
print(iris.head()) 
# print(iris.groupby('variety').size())
# mds = MDS(dissimilarity='precomputed', random_state=0)
# x_trans = mds.fit_transform(matriz)
# plt.figure(figsize=(27,10))
# figure = sns.scatterplot(x=x_trans[:,0],y=x_trans[:,1],hue=datos['tema'])
print(iris.iloc[:,4])
fig = iris[iris.variety == 'Setosa'].plot(kind='scatter', x='sepal.length', y='sepal.width', color='blue', label='Setosa')
iris[iris.variety == 'Versicolor'].plot(kind='scatter', x='sepal.length', y='seal.width', color='green', label='Versicolor', ax=fig)
iris[iris.variety == 'Virginica'].plot(kind='scatter', x='sepal.length', y='sepal.width', color='red', label='Virginica', ax=fig)
fig.set_xlabel('Pétalo - Longitud')
fig.set_ylabel('Pétalo - Ancho')
fig.set_title('Pétalo Longitud vs Ancho')
plt.show()