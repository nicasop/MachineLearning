# Deber 11 Cross Validation
# Integrantes: Diego Bedoya - Jhon Granda - Anthony Grijalva - Sebastian Sandoval - Alexis Villaviencio

# Librerias
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold,cross_validate,LeaveOneOut
import pandas as pd
import numpy as np

# Captura y Separación de los datos del dataset de Iris
iris = datasets.load_iris()
cant_datos = 100
X = iris.data[0:cant_datos,]
Y = iris.target[0:cant_datos]

# Modelo de Regresión Logística binomial
modelo = LogisticRegression(penalty='l1',solver='liblinear',multi_class='ovr',random_state=1)

# Selección del tamanio de la particion de train y test
train = 0.70
cant_train = int(train*cant_datos)
cant_test = cant_datos-cant_train
print('Para el metodo de holdout y el de random cross validation se utilizo:')
print('Training {}% de los datos'.format(round(train*100)))
print('Test {}% de los datos\n'.format(round((1 - train)*100)))

##### Método de HoldOut #####
# Entrenamiento
x_train = X[0:cant_train,:]
y_train = Y[0:cant_train]
modelo.fit(x_train,y_train)

# Test
x_test = X[cant_train:cant_datos,]
y_test = Y[cant_train:cant_datos]
yest = modelo.predict(x_test)

# Validacion
print('------- METODO HOLDOUT --------')
print("Accuracy del metodo de holdout:",accuracy_score(yest,y_test))

##### Random Cross Validation ####
# Randomizacion de los datos
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=train,random_state=1)

# Entrenamiento
modelo.fit(x_train,y_train)

# Test
yest=modelo.predict(x_test)

# Validacion
print('\n------- METODO RANDOM CROSS VALIDATION --------')
print("Accuracy del metodo de Random Cross Validation:",accuracy_score(yest,y_test))

#### K-Fold Cross Validation ####
# Entrenamiento
num_particiones = 10
kf=KFold(n_splits=num_particiones)

# Test
score=cross_val_score(modelo,X,Y,cv=kf)

# Validacion
print('\n------- METODO K-FOLD CROSS VALIDATION --------')
print('Se utilizo un k=',num_particiones)
print("Accuracy de cada iteracion del metodo K-Fold Cross Validation:",score)

# Leave One Out Cross Validation
# Entrenamiento
lO = LeaveOneOut()

# Test
score=cross_val_score(modelo,X,Y,cv=lO)

# Validacion
print('\n------- METODO LEAVE ONE OUT CROSS VALIDATION --------')
print("Accuracy de cada iteracion del metodo Leave One Out Cross Validation:",score)


