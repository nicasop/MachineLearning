# Deber 12 Evaluación Rendimiento Clasificadores 
# Integrantes: Diego Bedoya - Jhon Granda - Anthony Grijalva - Sebastian Sandoval - Alexis Villaviencio

# Librerias
import numpy as np

# Matriz de confusión a utilizar
matriz = np.array([[944,0,5,3,0,12,14,1,1,0],
                   [0,1100,4,4,1,0,2,2,22,0],
                   [20,18,873,22,18,0,20,18,40,3],
                   [10,2,29,888,2,34,2,18,19,6],
                   [1,1,5,0,893,1,20,3,6,52],
                   [21,6,3,47,11,721,26,9,41,7],
                   [20,2,12,1,18,15,883,0,7,0],
                   [5,20,27,2,8,1,0,935,4,26],
                   [7,19,11,24,11,46,29,14,792,21],
                   [10,2,2,12,53,14,1,35,14,866]])

# matriz_P = np.array([[130,74,2,6],
#                      [96,99,6,16],
#                      [3,4,207,4],
#                      [6,12,4,177]])

print('--------- Matriz de Confusión ----------\n',matriz,'\n')
# print(matriz_P)

# Funciones
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

# Codigo
precision_values = []
recall_values = []
f1_score_values = []
num_clases = len(matriz)

for i in range(1,num_clases+1):
    pc = precision(matriz,i)
    rc = recall(matriz,i)
    f1s = F1_score(rc,pc)
    precision_values.append(pc)
    recall_values.append(rc)
    f1_score_values.append(f1s)

print('--------- Resumen de las Metricas ---------')
print('Acurracy:',accuracy(matriz),'\n')
for i in range(1,num_clases+1):
    print('Clase',i)
    print('Precision:',precision_values[i-1])
    print('Recall:',recall_values[i-1])
    print('F1 Score:',f1_score_values[i-1],'\n')