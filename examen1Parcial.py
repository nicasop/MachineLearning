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
    documentoL = []
    if tipo == 'en':
        n = stopwords.words("english")
    elif tipo == 'es':
        n = stopwords.words("spanish")
    for token in documento:
        if token not in n:
            documentoL.append(token)
    return documentoL

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
    
##MATRIZ DE iNCIDENCIA BINARIA
def mInBinaria(dic,coleccion):
    iBinaria = np.zeros((len(dic['tokens']),len(coleccion)))
    for i in range (len(dic['tokens'])):
        ocurrencia = dic['ocurrencias'][i]
        for ocu in ocurrencia:
            iBinaria[i][ocu[0]-1] = 1
    return iBinaria

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

##Jaccardd
def interseccion(a,b):
    cont=0
    for token in a:
        if token in b:
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
    print(uni)
    return len(uni)

def jaccard(a,b):
    intersec = interseccion(a,b)  
    uni = union(a,b) 
    return round(intersec/uni,2)




# 1. Consumir en Python, directamente desde la URL, el siguiente dataset: 
# https://archive.ics.uci.edu/ml/datasets/AAAI+2013+Accepted+Papers (Enlaces a un sitio externo.) . 
# En el caso de no lograr hacer este paso ir al literal 2, caso contrario avanzar al literal 3 (1 Pto.)
datosCSV = importarCSV('https://archive.ics.uci.edu/ml/machine-learning-databases/00314/%5bUCI%5d%20AAAI-13%20Accepted%20Papers%20-%20Papers.csv')
print('\nPregunta 1\n')
print('datos Obtenidos desde le archivo csv',datosCSV)

# 3.Realizar el proceso NLP de normalización de los abstracts, de los primeros seis documentos (0.5 Ptos.).
datosImp = datosCSV['Abstract'].tolist()
coleccionAbs = datosImp[0:6]
coleccionAbsL = normalizarDocumento(coleccionAbs)
print('\nPregunta 3\n')
print('\nColeccion normalizada\n',coleccionAbsL)

# 4.Realizar eliminación de stopwords, stemming (algoritmo de Porter) y tokenización de los abstracts, 
# de los primeros seis documentos (0.5 Ptos.).
coleccionAbsL = limpiarDocumento(coleccionAbsL,'en')
print('\nPregunta 4\n')
print('\nColeccion Limpia\n',coleccionAbsL)

# 5.Obtener el full inverted index, del literal anterior (1 Pto.)
diccionario={'tokens':[],'ocurrencias':[]}
diccionario['tokens']= indexacionToken(coleccionAbsL)
diccionario['ocurrencias'] = ocurrencias(diccionario['tokens'],coleccionAbsL)
print('\nPregunta 5\n')
imprimirFII(diccionario['tokens'],diccionario['ocurrencias'])

# 6.Colocar en un tabla hash de tamaño 10, los primeros 5 tokens del literal anterior. Usar el sistema de codificación ASCII (1 Pto.)
print('\nPregunta 6\n')
base = 2
tokensE = diccionario['tokens'][0:5]
print('\n5 Primeros tokens\n',tokensE)
tokensECod = []
for token in tokensE:
    tokensECod.append(clavePalabra(token,base))
print('\n5 Primeros tokens codificados\n',tokensECod)

tamTablaHash = 10
hashTable = [None]*tamTablaHash

posicionElemento = 0
numColisiones = 0
rehas = False
if ( len(tokensECod) <= tamTablaHash):
    while posicionElemento < len(tokensECod):
        if not rehas:
            posicion = hashFunction(tokensECod[posicionElemento],tamTablaHash)

        if (hashTable[posicion] is None):
            hashTable[posicion] = tokensECod[posicionElemento]
            rehas = False
            posicionElemento += 1
        else:
            numColisiones += 1
            rehas = True
            k = kvalue(tokensECod[posicionElemento],tamTablaHash)
            posicion = rehashingFunction(posicion,k,tamTablaHash)

    print('Número de colisiones: ',numColisiones)
    printHashTable(hashTable)
else:
    print('El número de elemtos es mayor al número de cubetas disponibles en la tabla hash')

# 7.Obtener la matriz de incidencia binaria, del literal 4, e indicar las dimensiones de 
# la matriz resultante (1 Pto.)
matrizInBinaria = mInBinaria(diccionario,coleccionAbsL)
print('\nPregunta 7\n')
print('\nMatriz de incidendia binaria\n',matrizInBinaria)
print('Dimension: Filas-->',len(matrizInBinaria),'Columnas-->',len(matrizInBinaria[0]))

# 8.Obtener la matriz TF-IDF, del literal 4, e indicar las dimensiones de la matriz resultante (1 Pto.)
matriz = bagWords(diccionario,coleccionAbsL)
wtf = matrizPTF(matriz)
dF = documentF(wtf)
idf = IDF(dF,len(coleccionAbsL))
tf_idf = TFIDF(wtf,idf)
print('\nPregunta 8\n')
print('\nMatriz TF-IDF\n',tf_idf)
print('Dimension: Filas-->',len(tf_idf),'Columnas-->',len(tf_idf[0]))

# 9.Mediante el coeficiente de Jaccard, en función de los abstracts que han pasado todo el proceso de NLP, 
# indique que tan similares son los documentos D1 y D2 (1 Pto.)
D1 = coleccionAbsL[0]
D2 = coleccionAbsL[1]
print('\nPregunta 9\n')
print('D1\n',D1)
print('D2\n',D2)
print('\nLa similitud entre los documentos D1 y D2 es:',jaccard(D1,D2)*100,'%')