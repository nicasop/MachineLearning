import pandas as pd
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
##importar documento
def importarCSV(nombre_file):
    return pd.read_csv(nombre_file)

##nlp
def normalizarDocumento (cole):
    colecciontok=[]
    for documento in range (len(cole)):
        documentoaux = re.sub('[^A-Za-z0-9]+',' ', cole[documento])#eliminar caracteres especiales    
        documentoaux = documentoaux.lower()# poner todo en minúsculas
        colecciontok.append(documentoaux)
    return colecciontok

def limpiarDocumento (cole,idioma):    
    colecciontok=[]
    for documento in range (len(cole)):
        documentoaux = cole[documento].split()# tokenización
        documentoaux = quitarStopwords(idioma,documentoaux)# quitar stopwords
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

##Codificacion
def clavePalabra(palabra,base):
    palabra = palabra.lower()
    codigo = 0
    tamPalabra = len(palabra)-1
    for i in range(len(palabra)): 
        codigo += ord(palabra[i])*base**(tamPalabra - i)
    return codigo

##tabla hash
def hashFunction(dividendo,divisor):
   return dividendo%divisor

def kvalue(x,B):
    return hashFunction(x,B-1) + 1

def rehashingFunction(posHashAn,k,B):
    return hashFunction(posHashAn + k,B)

def printHashTable(hashTable):
    for i in range(len(hashTable)):
        print(i,' -> ',hashTable[i])


# 1. Consumir en Python, directamente desde la URL, el siguiente dataset: 
# https://archive.ics.uci.edu/ml/datasets/AAAI+2013+Accepted+Papers (Enlaces a un sitio externo.) . 
# En el caso de no lograr hacer este paso ir al literal 2, caso contrario avanzar al literal 3 (1 Pto.)
datosCSV = importarCSV('https://archive.ics.uci.edu/ml/machine-learning-databases/00314/%5bUCI%5d%20AAAI-13%20Accepted%20Papers%20-%20Papers.csv')

# 3.Realizar el proceso NLP de normalización de los abstracts, de los primeros seis documentos (0.5 Ptos.).
datosImp = datosCSV['Abstract'].tolist()
coleccionAbs = datosImp[0:6]
coleccionAbsL = normalizarDocumento(coleccionAbs)
print('\nPregunta3\n')
print('\nColeccion normalizada\n',coleccionAbsL)

# 4.Realizar eliminación de stopwords, stemming (algoritmo de Porter) y tokenización de los abstracts, 
# de los primeros seis documentos (0.5 Ptos.).
coleccionAbsL = limpiarDocumento(coleccionAbsL,'en')
print('\nPregunta4\n')
print('\nColeccion Limpia\n',coleccionAbsL)

# 5.Obtener el full inverted index, del literal anterior (1 Pto.)
diccionario={'tokens':[],'ocurrencias':[]}
diccionario['tokens']= indexacionToken(coleccionAbsL)
diccionario['ocurrencias'] = ocurrencias(diccionario['tokens'],coleccionAbsL)
print('\nPregunta5\n')
imprimirFII(diccionario['tokens'],diccionario['ocurrencias'])

# Colocar en un tabla hash de tamaño 10, los primeros 5 tokens del literal anterior. Usar el sistema de codificación ASCII (1 Pto.)

palabra = 'HOLA'
base = 2

tamTablaHash = 11
hashTable = [None]*tamTablaHash
#elementos = [23,14,9,6,30,12,18] # tamaño tabla hash 7 usado para pruebas
elementos = [51,14,3,7,18,30] # tamaño tabla hash 11 usado para pruebas

posicionElemento = 0
numColisiones = 0
rehas = False
if ( len(elementos) <= tamTablaHash):
    while posicionElemento < len(elementos):
        if not rehas:
            posicion = hashFunction(elementos[posicionElemento],tamTablaHash)

        if (hashTable[posicion] is None):
            hashTable[posicion] = elementos[posicionElemento]
            rehas = False
            posicionElemento += 1
        else:
            numColisiones += 1
            rehas = True
            k = kvalue(elementos[posicionElemento],tamTablaHash)
            posicion = rehashingFunction(posicion,k,tamTablaHash)

    print('Número de colisiones: ',numColisiones)
    printHashTable(hashTable)
else:
    print('El número de elemtos es mayor al número de cubetas disponibles en la tabla hash')
# Obtener la matriz de incidencia binaria, del literal 4, e indicar las dimensiones de la matriz resultante (1 Pto.)

# Obtener la matriz TF-IDF, del literal 4, e indicar las dimensiones de la matriz resultante (1 Pto.)

# Mediante el coeficiente de Jaccard, en función de los abstracts que han pasado todo el proceso de NLP, indique que tan similares son los documentos D1 y D2 (1 Pto.)