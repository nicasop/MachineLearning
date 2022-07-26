## PRUEBA 4 ALEXIS VILLAVICENCIO

import numpy as np
import pandas as pd

## FUNCIONES
def accuracy(matrix):
    sum = 0
    for i in range(len(matrix)):
        sum += matrix[i][i]
    return round(sum/np.sum(matrix),3)

def precision(matrix,pos_clase):
    mT = np.transpose(matrix)
    return round(mT[pos_clase-1][pos_clase-1]/np.sum(mT[pos_clase-1]),3)

def recall (matrix,pos_clase):
    return round(matrix[pos_clase-1][pos_clase-1]/np.sum(matrix[pos_clase-1]),3)

def F1_score(recall,precision):
    return round(2*(recall*precision/(recall+precision)),3)

def FPR(matrix,pos_class):
    total_clase_original = np.sum(matrix, axis=1)
    mdf = pd.DataFrame(matrix)
    sumN = sumD = 0
    for i in range(len(mdf[pos_class-1])):
        if i != (pos_class-1):
            sumN += mdf[pos_class-1][i]
            sumD += total_clase_original[i]
    return round(sumN/sumD,3)

## Matriz de confusion
matriz_confusion = np.array([[137,13,3,0,0,1],
                            [1,55,1,0,0,1],
                            [2,4,84,0,0,0],
                            [3,0,1,153,5,2],
                            [0,0,3,0,44,2],
                            [0,0,2,1,4,35]])
print('------------ Matriz de Confusion ------------')
print(matriz_confusion)

# a) Determinar el precision para cada clase (0.5 Ptos.)
pre = []
for i in range(1,len(matriz_confusion)+1):
    pre.append(precision(matriz_confusion,i))

print('\n--------- PRECISION DE CADA CLASE ----------')
print(pre)

# b) Determinar el recall para cada clase (0.5 Ptos.)
rec = []
for i in range(1,len(matriz_confusion)+1):
    rec.append(recall(matriz_confusion,i))

print('\n--------- RECALL DE CADA CLASE ----------')
print(rec)

# c) Determinar el accuracy (0.5 Ptos.)
print('\n--------- ACCURACY DEL MODELO ----------')
print(accuracy(matriz_confusion))

# d) Determinar la sensitividad (0.5 Ptos.)
sen = []
for i in range(1,len(matriz_confusion)+1):
    sen.append(recall(matriz_confusion,i))

print('\n--------- SENSITIVIDAD DE CADA CLASE ----------')
print(sen)

# e) Determinar la especificidad (0.5 Ptos.)
## El TNR = 1 - FPR 
fpr = []
for i in range(1,len(matriz_confusion)+1):
    fpr.append(FPR(matriz_confusion,i))

print('\n--------- ESPECIFICIDAD DE CADA CLASE ----------')
print(1-np.array(fpr))

# f) Determinar el False Negative Rate (0.5 Ptos.)
## El FNR = 1 - TPR pero el TPR es lo mismo que el recall/sencitividad
print('\n--------- FNR DE CADA CLASE ----------')
print(1-np.array(sen))

# g) Determinar el False Positive Rate (0.5 Ptos.)
fpr = []
for i in range(1,len(matriz_confusion)+1):
    fpr.append(FPR(matriz_confusion,i))

print('\n--------- FPR DE CADA CLASE ----------')
print(fpr)


# h) Determinar el número de aciertos (0.5 Ptos.)
print('\n--------- Numero de aciertos ----------')
print(np.sum(np.diag(matriz_confusion)))

# i) Determinar el tamaño del dataset (0.5 Ptos.)
print('\n--------- Cantidad de en el dataset ----------')
print(np.sum(matriz_confusion))

# j) Indicar la cardinalidad de la clase real (1 Pto.)
print('\n---------- Cardinalidad de cada clase ----------')
car_real = np.sum(matriz_confusion, axis=1)
print(car_real)
for i in range(len(car_real)):
    print('Clase',i+1,':',car_real[i])

# k) Indicar la cantidad de instancias clasificadas como clase 3 que eran de la clase 5 (0.5 Ptos.)
print('\n--------- CLASIFICADAS CLASE 3 QUE ORIGINALMENTE ERAN CLASE 5 ----------')
print(matriz_confusion[4][2])

# l) Determinar el TPR (1 Pto.)
tpr = []
for i in range(1,len(matriz_confusion)+1):
    tpr.append(recall(matriz_confusion,i))

print('\n--------- TPR DE CADA CLASE ----------')
print(tpr)