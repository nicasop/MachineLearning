#### JACCARD ####

#### LIBRERIAS ####
import numpy as np

def indexarDoc(a):
    aL = []
    for element in a:
        if element not in aL:
            aL.append(element)
    return aL

def interseccion(a,b):
    aL = indexarDoc(a)
    bL = indexarDoc(b)
    cont=0
    for token in aL:
        for token1 in bL:
            if token == token1:
                cont += 1
    return cont

def union(a,b):
    uni = []
    for token in a:
        if token not in uni:
            uni.append(token)
    for token in b:
        if token not in uni:
            uni.append(token)
    return len(uni)

def jaccard(a,b):
    intersec = len(np.intersect1d(a,b))   
    uni = len(np.union1d(a,b)) 
    # intersec = interseccion(a,b)
    # uni = union(a,b)
    return round(intersec/uni,2)

# def matrizJaccard(coleccion):
#     matriz = np.zeros((len(coleccion),len(coleccion)))
#     for i in range(len(coleccion)):
#         for j in range(len(coleccion)):
#             if matriz[i][j] == 0:
#                 matriz[i][j] = matriz[j][i] = jaccard(coleccion[i],coleccion[j])
#     return matriz

def matrizJaccard(coleccion):
    matriz = np.zeros((len(coleccion),len(coleccion)))
    i = j = 0
    while True:
        if matriz[i][j] == 0:
            matriz[i][j] = matriz[j][i] = jaccard(coleccion[i],coleccion[j])
        j += 1
        if j == len(coleccion):
            j = 0 
            i += 1
        if i == len(coleccion):
            break
    return matriz