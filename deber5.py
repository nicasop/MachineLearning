#### LIBRERIAS ####
import math
import numpy as np
import nlp
import jaccard

# d1 = 'dias de lluvia'
# d2 = 'resbalo en un dia de lluvia'
# coleccion = [d1.split(),d2.split()]

# print(coleccion)
# print(jaccard.matrizJaccard(coleccion))

#### TF-IDF ####

def bagWords(dic,coleccion):
    bWords = np.zeros((len(dic['tokens']),len(coleccion)))
    for i in range (len(dic['tokens'])):
        ocurrecia = dic['ocurrencias'][i]
        for ocu in ocurrecia:
            bWords[i][ocu[0]-1] = ocu[1]
    return bWords

def pesadoTF(term):
    if term != 0:
        return 1 + math.log(term,10)
    else:
        return 0

def matrizPTF(matriz):
    for i in range(len(matriz)):
        for j in range(len(matriz[0])):
            matriz[i][j] = pesadoTF(matriz[i][j])
    return matriz

d1 = 'LA CASA DEL ARBOL, DENTRO DEL ARBOL'
d2 = 'ARBOLES RURALES, PARROQUIA RURAL'
d3 = 'FLORA Y FAUNA, DE ARBOLES ECUATORIANOS DE ARBOLES ENDEMICOS'
coleccion = [d1,d2,d3]
coleccionLim = nlp.limpiarDocumento(coleccion)
print(coleccion)
print(coleccionLim)

diccionario={'tokens':[],'ocurrencias':[]}
diccionario['tokens']= nlp.indexacionToken(coleccionLim)
diccionario['ocurrencias'] = nlp.ocurrencias(diccionario['tokens'],coleccionLim)
print(diccionario['ocurrencias'])
nlp.imprimirFII(diccionario['tokens'],diccionario['ocurrencias'])

matriz = bagWords(diccionario,coleccion)
print(matriz)

matrizP = matrizPTF(matriz)
print(matrizP)





