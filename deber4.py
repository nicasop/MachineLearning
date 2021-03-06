# Integrantes: Diego Bedoya - John Granda - Anthony Grijalva - Sebastian Sandoval - Alexis Villavicencio

##librerias
import re
from operator import index
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

##funciones
def limpiarDocumento (cole):    
    colecciontok=[]
    for documento in range (len(cole)):
        documentoaux = re.sub('[^A-Za-z0-9]+',' ', cole[documento])#eliminar caracteres especiales    
        documentoaux = documentoaux.lower()# poner todo en minúsculas
        documentoaux = documentoaux.split()# tokenización
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

def exportarCSV(nombre_file,coleccion):
    dic = {'coleccion': coleccion} 
    df = pd.DataFrame(dic) 
    df.to_csv(nombre_file,index=False,encoding='utf-8')

def importarCSV(nombre_file):
    return pd.read_csv(nombre_file)

def imprimirFII(vecT,vecO):
    print('Full Inverted Index')
    for j in range (len(vecT)):
        print(vecT[j],'-->',vecO[j])


## programa ##
coleccion = recuperarDatosHTML("https://openai.com/dall-e-2/","p")
#print('Datos recuperados de la página web',coleccion)
exportarCSV("deber4.csv",coleccion)
coleccionImp = importarCSV('deber4.csv')
#print('Datos recuperados desde el archivo CSV',coleccionImp['coleccion'].tolist())
datosImp = coleccionImp['coleccion'].tolist()
coleccionLim = limpiarDocumento(datosImp)
diccionario={'tokens':[],'ocurrencias':[]}
diccionario['tokens']= indexacionToken(coleccionLim)
diccionario['ocurrencias'] = ocurrencias(diccionario['tokens'],coleccionLim)
imprimirFII(diccionario['tokens'],diccionario['ocurrencias'])