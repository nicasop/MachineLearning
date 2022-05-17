##librerias
import re
from operator import index
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import math
nltk.download('stopwords')

def importarCSV(nombre_file):
    return pd.read_csv(nombre_file)

def limpiarDocumento1 (cole):    
    colecciontok=[]
    for documento in range (len(cole)):
        documentoaux = re.sub('[^A-Za-z0-9]+',' ', cole[documento])#eliminar caracteres especiales    
        documentoaux = documentoaux.lower()# poner todo en minúsculas
        colecciontok.append(documentoaux)
    return colecciontok

def limpiarDocumento2 (cole,idioma):    
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

##literal 3
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
    print('Full Inverted Index')
    for j in range (len(vecT)):
        print(vecT[j],'-->',vecO[j])

##literal 4
def hashFunction(dividendo,divisor):
   return dividendo%divisor

def kvalue(x,B):
    return hashFunction(x,B-1) + 1

def rehashingFunction(posHashAn,k,B):
    return hashFunction(posHashAn + k,B)

def printHashTable(hashTable):
    for i in range(len(hashTable)):
        print(i,' -> ',hashTable[i])

def clavePalabra(palabra,base):
    palabra = palabra.lower()
    codigo = 0
    tamPalabra = len(palabra)-1
    for i in range(len(palabra)): 
        codigo += ord(palabra[i])*base**(tamPalabra - i) #funcion polinomial puede ser cambiada
    return codigo

##literal 5 
def mIncidenciaB(dic,coleccion):
    bWords = np.zeros((len(dic['tokens']),len(coleccion)))
    for i in range (len(dic['tokens'])):
        ocurrencia = dic['ocurrencias'][i]
        for ocu in ocurrencia:
            bWords[i][ocu[0]-1] = 1
    return bWords

##literal 6
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
    i = j = 0
    while True:
        mPTF[i][j] = pesadoTF(matriz[i][j])
        j += 1
        if j == len(matriz[0]):
            j = 0
            i += 1
        if i == len(matriz):
            break
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

coleccion = importarCSV("papers.csv")
Abst = coleccion['abstract'].tolist()
Abst = Abst[0:6]

# Realizar el proceso NLP de normalización de los abstracts, de los primeros seis documentos
AbstL = limpiarDocumento1(Abst)

# Realizar eliminación de stopwords, stemming (algoritmo de Porter) y tokenización de los 
# abstracts, de los primeros seis documentos
AbstL = limpiarDocumento2(AbstL,'en')

# Obtener el full inverted index, del literal anterior
diccionario={'tokens':[],'ocurrencias':[]}
diccionario['tokens']= indexacionToken(AbstL)
diccionario['ocurrencias'] = ocurrencias(diccionario['tokens'],AbstL)
imprimirFII(diccionario['tokens'],diccionario['ocurrencias'])

# Colocar en un tabla hash de tamaño 10, los primeros 5 tokens del literal anterior. Usar el 
# sistema de codificación ASCII, con base a=3 y función polinomial

tokens = diccionario['tokens'][0:5]
base = 3
elementos = []
for token in tokens:
    elementos.append(clavePalabra(token,base))

print(elementos)

tamTablaHash = 10
hashTable = [None]*tamTablaHash
#elementos = [23,14,9,6,30,12,18] # tamaño tabla hash 7 usado para pruebas
# elementos = [51,14,3,7,18,30] # tamaño tabla hash 11 usado para pruebas
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

# Obtener la matriz de incidencia binaria, del literal 4, e indicar las dimensiones 
# de la matriz resultante
matrizIB = mIncidenciaB(diccionario,AbstL)
print(matrizIB)
print('filas:',len(matrizIB))
print('columnas:',len(matrizIB[0]))


# Obtener la matriz TF-IDF, del literal 4, e indicar las dimensiones de la matriz resultante
matriz = bagWords(diccionario,AbstL)
wtf = matrizPTF(matriz)
dF = documentF(wtf)
idf = IDF(dF,len(AbstL))
tf_idf = TFIDF(wtf,idf)
print(tf_idf)
print('filas:',len(tf_idf))
print('columnas:',len(tf_idf[0]))