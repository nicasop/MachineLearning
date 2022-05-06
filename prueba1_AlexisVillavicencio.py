#####Alexis Villavicencio Prueba 1

##librerias
from pickle import TRUE
import re
import math
from operator import index
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

##funciones
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

def recuperarDatosHTML(enlace,etiqueta):
    file = urlopen(enlace)
    html = file.read()
    # file.close() #solo si la pagina web es dinamica
    soup = BeautifulSoup(html,features='html.parser')
    tit = []
    for p in soup.find_all(etiqueta):
        tit.append(p.get_text())
    titulos = tit[1:7]
    return titulos

def imprimirFII(vecT,vecO):
    print('\nFull Inverted Index')
    for j in range (len(vecT)):
        print(vecT[j],'-->',vecO[j])

def codificarToken(token):
    tam = len(token)
    x = math.ceil(tam/2)
    return 3*math.pow(x,4) + 2*math.pow(x,2) + 1

def hashFunction(dividendo,divisor):
   return dividendo%divisor

def kvalue(x,B):
    return hashFunction(x,B-1) + 1

def rehashingFunction(posHashAn,k,B):
    return hashFunction(posHashAn + k,B)

def printHashTable(hashTable):
    for i in range(len(hashTable)):
        print(i,' -> ',hashTable[i])

def elementosTablaHash(diccionario):
    elementos = []
    pos = 0
    while len(elementos) < 6:
        if diccionario[pos] not in elementos:
            elementos.append(int(diccionario[pos]))
        pos += 1
    return elementos

def cubetaPalabraHashTable(hashTable):
    palabras=[]
    cubeta=[]
    for i in range(len(hashTable)):
        if hashTable[i] is not None:
            palabras.append(hashTable[i])
            cubeta.append(i)
    return palabras,cubeta

def recuperarPalabras(codificacion,dicCodi,dic):
    palabrasC=[]
    for num in codificacion:
        posPalabras = []
        for i in range(len(dicCodi)):
            if num == dicCodi[i]:
                posPalabras.append(i)
        palabras = []
        for pos in posPalabras:
            palabras.append(dic[pos])
        palabrasC.append(palabras)
    return palabrasC

def imprimirPalabrasRecu(dic):
    print('\nPalabras guardadas en cada cubeta')
    for i in range(len(dic['codificacion'])):
        print('Cubeta: ',dic['cubeta'][i],'--> Codificación: ',dic['codificacion'][i],'--> Palabras: ',dic['palabras'][i])

## programa ##
##Pregunta 1 Extraer el párrafo (i.e.: 1. Classical IR Model.... ) y el párrafo (i.e.: 1. Boolean Model... ) de la siguiente URL. Cada párrafo es un documento diferente
print('\nPregunta 1')
sc = ['1. Classical IR Model — It is designed upon basic mathematical concepts and is the most widely-used of IR models. Classic Information Retrieval models can be implemented with ease. Its examples include Vector-space, Boolean and Probabilistic IR models. In this system, the retrieval of information depends on documents containing the defined set of queries. There is no ranking or grading of any kind. The different classical IR models take Document Representation, Query representation, and Retrieval/Matching function into account in their modelling.','1. Boolean Model — This model required information to be translated into a Boolean expression and Boolean queries. The latter is used to determine the information needed to be able to provide the right match when the Boolean expression is found to be true. It uses Boolean operations AND, OR, NOT to create a combination of multiple terms based on what the user asks.']
print("\nweb scrapping",sc)

##Pregunta 2 Realizar un proceso de limpieza eliminación de caracteres especiales y conversión a letras minúsculas (1 Pto.)
coleccionLim = limpiarDocumento1(sc)
print('\nPregunta 2')
print("\nLimpieza de caracteres especiales y convertido a minúsculas: ",coleccionLim)

##Pregunta 3 Eliminar las stopwords sobre los datos curados, realizar stemming (algoritmo de Porter) y tokenizar (1 Pto.).
coleccionLim = limpiarDocumento2(coleccionLim) 
print('\nPregunta 3')
print("\nTokenizacion, eliminar stopwords y stemming(Porter)",coleccionLim)

##Pregunta 4 Crear un diccionario de los tokens con su respectivo full inverted index (1 Pto.).
diccionario={'tokens':[],'ocurrencias':[]}
diccionario['tokens']= indexacionToken(coleccionLim)
diccionario['ocurrencias'] = ocurrencias(diccionario['tokens'],coleccionLim)
print('\nPregunta 4')
imprimirFII(diccionario['tokens'],diccionario['ocurrencias'])

##Pregunta 5 Transformar los tokens del diccionario en valores numéricos. Para esto usar la siguiente función no lineal 
# y = 3x^4+2x^2-1. Donde x es la longitud de dos caracteres del token que pasó por el stemmer (si el resultado de longitud 
# es impar redondear al inmediato superior, e.g.: cazar, x=3). (1.5 Ptos.)
dicCodificado = []
dicOrdenado = []
for token in diccionario['tokens']:
    dicCodificado.append(codificarToken(token))
    dicOrdenado.append(codificarToken(token))
dicOrdenado.sort(reverse=True)
print('\nPregunta 5')
print("\nDiccionario Codificado: ",dicCodificado)
print("\nDiccionario Codificado Ordenado: ",dicOrdenado)

##Pregunta 6 Cargar el diccionario numérico en una tabla hash que realice direccionamiento abierto con re-hashing y estrategia 
# de 2da función hash. La tabla hash tiene que tener 10 cubetas y se deben colocar solamente los 6 valores (diferentes) más altos 
# obtenidos en el literal anterior, el resto de cubetas debe tener el valor None. (2 Ptos.).
tamTablaHash = 10
hashTable = [None]*tamTablaHash
print(len(dicOrdenado))
elementos = elementosTablaHash(dicOrdenado)
print('\nElementos tabla hash: ',elementos)
print('\nPregunta 6')

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

##Pregunta 7 Indique la palabra o palabras que han sido colocadas en las cubetas llenas (1.5 Ptos).
palabrasCubetas = {'codificacion':[],'cubeta':[],'palabras':[]}
palabrasCubetas['codificacion'],palabrasCubetas['cubeta'] = cubetaPalabraHashTable(hashTable)
palabrasCubetas['palabras'] = recuperarPalabras(palabrasCubetas['codificacion'],dicCodificado,diccionario['tokens'])
print('\nPregunta 7')
imprimirPalabrasRecu(palabrasCubetas)