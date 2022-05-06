import numpy as np

def vectorModulos(matriz):
    i = j = 0
    modulos = []
    coordenadas = []
    while True:
        coordenadas.append(matriz[i][j]) 
        i += 1
        if i == len(matriz):
           i = 0
           j += 1
           modulos.append(np.linalg.norm(coordenadas))
           coordenadas = []
        if j == len(matriz[0]):
           break 
    return modulos

def matrizNormal(matriz):
    matrizNormal = np.zeros((len(matriz),len(matriz[0])))
    modulos = vectorModulos(matriz)
    # print('Modulos: ',modulos)
    i = j = 0
    while True:
        matrizNormal[i][j] = matriz[i][j]/modulos[j]
        i += 1
        if i == len(matrizNormal):
            i = 0
            j += 1
        if j == len(matrizNormal[0]):
            break
    return matrizNormal

def distancias(v,w):
    sum = 0
    for i in range(len(v)):
        sum += v[i]*w[i]
    return sum

def matrizDistacias(matriz):
    transpuesta = np.transpose(matriz)
    matrizD = np.zeros((len(transpuesta),len(transpuesta)))
    i = j = 0
    while True:
        if matrizD[i][j] == 0:
            matrizD[i][j] = matrizD[j][i] = distancias(transpuesta[i],transpuesta[j])
        j += 1
        if j == len(transpuesta):
            j = 0
            i += 1
        if i == len(transpuesta):
            break
    return matrizD