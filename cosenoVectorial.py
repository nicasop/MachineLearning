import numpy as np

# def vectorModulos(matriz):
#     # matrizTran = np.transpose(matriz)
#     modulos = []
#     for fila in matriz:
#         modulos.append(np.linalg.norm(fila))
#     # i = j = 0
#     # coordenadas = []
#     # while True:
#     #     coordenadas.append(matriz[i][j]) 
#     #     i += 1
#     #     if i == len(matriz):
#     #        i = 0
#     #        j += 1
#     #        modulos.append(np.linalg.norm(coordenadas))
#     #        coordenadas = []
#     #     if j == len(matriz[0]):
#     #        break 
#     return modulos

def matrizNormal(matriz):
    matrizTran = np.transpose(matriz)
    matrizNormal = np.zeros((len(matrizTran),len(matrizTran[0])))
    # modulos = vectorModulos(matrizTran)
    i = j = 0
    modulo = np.linalg.norm(matrizTran[i])
    while True:
        matrizNormal[i][j] = matrizTran[i][j]/modulo
        j += 1
        if j == len(matrizNormal[0]):
            j = 0
            i += 1
        if i == len(matrizNormal):
            break
        modulo = np.linalg.norm(matrizTran[i])
    return matrizNormal

# def distancias(v,w):
#     sum = 0
#     for i in range(len(v)):
#         sum += v[i]*w[i]
#     return sum
def matrizDistacias1(transpuesta):
    # transpuesta = np.transpose(matriz)
    matrizD = np.zeros((len(transpuesta),len(transpuesta)))
    for i in range(len(matrizD)):
        for j in range(len(matrizD[0])):
            if matrizD[i][j] == 0:
                matrizD[i][j] = matrizD[j][i] = round(np.dot(transpuesta[i],transpuesta[j]),2)
    return matrizD

def matrizDistacias(transpuesta):
    # transpuesta = np.transpose(matriz)
    matrizD = np.zeros((len(transpuesta),len(transpuesta)))
    i = j = 0
    while True:
        if matrizD[i][j] == 0:
            matrizD[i][j] = matrizD[j][i] = round(np.dot(transpuesta[i],transpuesta[j]),2)
        j += 1
        if j == len(transpuesta):
            j = 0
            i += 1
        if i == len(transpuesta):
            break
    return matrizD