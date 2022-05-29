##Alexis Villavicencio
##librerias
import re
from operator import index
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import math
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

##funciones
##nlp
def limpiarDocumento1 (cole):
    colecciontok=[]
    for documento in range (len(cole)):
        documentoaux = re.sub('[^A-Za-z0-9]+',' ', cole[documento])#eliminar caracteres especiales    
        documentoaux = documentoaux.lower()# poner todo en minúsculas
        colecciontok.append(documentoaux)
    return colecciontok

def limpiarDocumento2 (cole):    
    colecciontok=[]
    for documento in range (len(cole)):
        documentoaux = cole[documento].split()# tokenización
        documentoaux = quitarStopwords('en',documentoaux)# quitar stopwords
        documentoaux = stemming(documentoaux)# stemming
        colecciontok.append(documentoaux)
    return colecciontok

def quitarStopwords(tipo,documento):
    if tipo == 'en':
        n = stopwords.words("english")
    elif tipo == 'es':
        n = stopwords.words("spanish")
    for token in documento:
        if token in n:
            documento.remove(token)
    return documento

def stemming(documento):
    stemmer = PorterStemmer()
    documentoS = []
    for token in documento:
        documentoS.append(stemmer.stem(token))
    return documentoS

##FII
def indexacionToken(coleccion):
    palabras=[]    
    for documento in coleccion:
        for token in documento:
            if token not in palabras:
                palabras.append(token)
    return palabras

def obtenPos(tok,lista):
    vpos=[]
    for pos in range (len(lista)):
        if tok == lista[pos]:
            vpos.append(pos+1)
    return vpos

def tokenDoc(tok,colDoc):
    vaux=[]
    for doc in range (len(colDoc)):
        vaux1=[]
        if tok in colDoc[doc]:
            vaux1.append(doc+1)
            vaux1.append(len(obtenPos(tok,colDoc[doc])))
            vaux1.append(obtenPos(tok,colDoc[doc]))
            vaux.append(vaux1)
    return vaux

def ocurrencias (dic,colDoc):
    vec=[]
    for token in dic:
        vec.append(tokenDoc(token,colDoc))
    return vec

def imprimirFII(vecT,vecO):
    print('\nFull Inverted Index')
    for j in range (len(vecT)):
        print(vecT[j],'-->',vecO[j])

##tf-idf
def bagWords(dic,coleccion):
    bWords = np.zeros((len(dic['tokens']),len(coleccion)))
    for i in range (len(dic['tokens'])):
        ocurrencia = dic['ocurrencias'][i]
        for ocu in ocurrencia:
            bWords[i][ocu[0]-1] = ocu[1]
    return bWords

def pesadoTF(term):
    if term != 0:
        return 1 + math.log10(term)
    else:
        return 0

def documentF(matriz):
    doFrecuen = []
    for lista in matriz:
        doFrecuen.append(conteo(lista))
    return doFrecuen


def conteo(lista):
    cont = 0
    for elemento in lista:
        if elemento != 0:
            cont += 1
    return cont

def matrizPTF(matriz):
    mPTF = np.zeros((len(matriz),len(matriz[0]))) 
    for i in range(len(matriz)):
        for j in range(len(matriz[0])):
            mPTF[i][j] = pesadoTF(matriz[i][j])
    return mPTF

def IDF(df,N):
    idf = []
    for elemento in df:
        idf.append(math.log10(N/elemento))
    return idf

def TFIDF(wtf,idf):
    matriz = np.zeros((len(wtf),len(wtf[0])))
    i = j = 0
    while True:
        matriz[i][j] = wtf[i][j]*idf[i]
        j += 1
        if j == len(wtf[0]):
            j = 0
            i += 1
        if i == len(wtf):
            break
    return matriz

##Coseno Vectorial
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

##puntuacion
def puntuacion(matriz):
    pD1 = pD2 = pD3 = 0
    i = 0
    while True:
        if matriz[i][0] != 0:
            pD1 += matriz[i][len(matriz[0])-3]
            pD2 += matriz[i][len(matriz[0])-2]
            pD3 += matriz[i][len(matriz[0])-1]
        i += 1
        if i == len(matriz):
            break
    vector = [pD1,pD2,pD3]
    return vector


Q1 = "Hash table in Machine Vision Learning"
Q2 = "Probabilisitic model in cancer experiments"
Q3 = "Learning in tasks with small data and Classifications models"

D1 = "Hashing is a popular approximate nearest neighbor search approach in large-scale image retrieval. Supervised hashing, which incorporates similarity/dissimilarity information on entity pairs to improve the quality of hashing function learning, has recently received increasing attention. However, in the existing supervised hashing methods for images, an input image is usually encoded by a vector of hand-crafted visual features. Such hand-crafted feature vectors do not necessary preserve the accurate semantic similarities of images pairs, which may often degrade the performance of hashing function learning. In this paper, we propose a supervised hashing method for image search, in which we automatically learn a good image representation tailored to hashing as well as a set of hash functions."
D2 = "Our two main contributions are: (i) two novel probabilistic models for binary and multiclass classification, and (ii) very efficient variational approximation procedures for these models. We illustrate the generalization performance of our algorithms on two different applications. In computer vision experiments, our method outperforms the state-of-the-art algorithms on nine out of 12 benchmark supervised domain adaptation experiments defined on two object recognition data sets."
D3 = "In cancer biology experiments, we use our algorithm to predict mutation status of important cancer genes from gene expression profiles using two distinct cancer populations, namely, patient-derived primary tumor data and in-vitro-derived cancer cell line data. We show that we can increase our generalization performance on primary tumors using cell lines as an auxiliary data source."

consultas = [Q1,Q2,Q3]
documentos = [D1,D2,D3]
print('Consultas\n',consultas)
print('Documentos\n',documentos)

# 1. Realizar un proceso de limpieza eliminación de caracteres especiales y conversión a letras 
# minúsculas, tanto para consultas como para documentos. (1 Pto.)
print('Pregunta 1')
consultasL = limpiarDocumento1(consultas)
print('\nConsultas sin caracteres especiales y en minusculas\n',consultasL)
documentosL = limpiarDocumento1(documentos)
print('\nDocumentos sin caracteres especiales y en minusculas\n',documentosL)

# 2. Eliminar las stopwords sobre los datos curados, realizar stemming (algoritmo de Porter) y tokenizar 
# (tanto consultas como documentos). (1 Pto.).
print('Pregunta 2')
consultasL = limpiarDocumento2(consultasL)
print('\nConsultas sin stopwords, realizado stemming y tokenizacion\n',consultasL)
documentosL = limpiarDocumento2(documentosL)
print('\nDocumentos sin stopwords, realizado stemming y tokenizacion\n',documentosL)

# 3. Crear una sola bolsa de palabras unificada (tanto de consultas como documentos) 
# e indicar las dimensiones de la matriz (1.5 Ptos.)
print('Pregunta 3')
coleccion = consultasL + documentosL
diccionario={'tokens':[],'ocurrencias':[]}
diccionario['tokens']= indexacionToken(coleccion)
diccionario['ocurrencias'] = ocurrencias(diccionario['tokens'],coleccion)
# imprimirFII(diccionario['tokens'],diccionario['ocurrencias'])
bagW = bagWords(diccionario,coleccion)
print('\nBolsa de palabras de toda la coleccion que incluye consultas y documentos\n',bagW)
print('\nLas dimensiones de la matriz son\nFilas:',len(bagW),' Columnas:',len(bagW[0]))

# 4.Aplicar el pesado TF, de la bolsa de palabras del literal anterior (1 Ptos.)
print('Pregunta 4')
wtf = matrizPTF(bagW)
print('\nLa matriz TF o pesado del TF de toda la coleccion que incluye consultas y documentos\n',wtf)

# 5.Calcular el IDF, únicamente de las tres consultas (no documentos) (1.5 Ptos.)
print('Pregunta 5')
diccionarioConsultas={'tokens':[],'ocurrencias':[]}
diccionarioConsultas['tokens']= indexacionToken(consultasL)
diccionarioConsultas['ocurrencias'] = ocurrencias(diccionario['tokens'],consultasL)
# imprimirFII(diccionarioConsultas['tokens'],diccionarioConsultas['ocurrencias'])
bagWC = bagWords(diccionarioConsultas,consultasL)
print('\nBolsa de palabras de las consultas\n',bagWC)
dFC = documentF(bagWC)
idfC = IDF(dFC,len(consultasL))
print('\nIDF de las consultas\n',idfC)

# 6.Obtener la matriz TF-IDF, de toda la bolsa total de palabras, del literal 3 (1.5 Ptos.)
print('Pregunta 6')
dF = documentF(wtf)
idf = IDF(dF,len(coleccion))
tf_idf = TFIDF(wtf,idf)
print('\nMatriz TF-IDF de toda la coleccion que incluye consultas y documentos\n',tf_idf)

# 7.Qué tan similares son Q1 y D1. Exprese su respuesta como una probabilidad (1 Pto.)
print('Pregunta 7')
transpuesta = np.transpose(tf_idf)
coleccion1 = [transpuesta[0],transpuesta[3]]
tfIdf_q1_d1 = np.transpose(coleccion1)
matrizN_q1_d1 = matrizNormal(tfIdf_q1_d1)
matrizSi_q1_d1 = matrizDistacias(matrizN_q1_d1)
print('\nMatriz de similitud entre Q1 y D1\n',matrizSi_q1_d1)
print('EL nivel de similitud entre Q1 y D1 es de:',matrizSi_q1_d1[0][1]*100,'%')

# 8.Considerando un sistema de RI. Si un usuario realiza la primera consulta, cuál debería 
# ser el orden en el que se desplieguen los tres documentos en función de su importancia? 
# (Justifique su respuesta mediante un sistema de puntuación) (1.5 Ptos.)
print('Pregunta 8')
coleccion2 = [transpuesta[0],transpuesta[3],transpuesta[4],transpuesta[5]]
tfIdf_q1_docs = np.transpose(coleccion2)
print('\nMatriz TFIDF de la Q1 con los Documentos\n',tfIdf_q1_docs)
vectorPuntuacion = puntuacion(tfIdf_q1_docs)
print('\nVector de puntuaciones',vectorPuntuacion)

print('La puntuacion del documento 1 con respecto a la consulta 1 es:',vectorPuntuacion[0])
print('La puntuacion del documento 2 con respecto a la consulta 1 es:',vectorPuntuacion[1])
print('La puntuacion del documento 3 con respecto a la consulta 1 es:',vectorPuntuacion[2])
###Respuesta
# d1 -->1.3526624873414566
# d2 -->0.47712125471966244
# d3 --> 0.0
##El orden de aparicion de los documentos tomando en cuenta su puntuacion seria D1,D2,D3
# pero considerando que el ultimo obtuvo una puntuacion de cero 0 talvez no se muestre