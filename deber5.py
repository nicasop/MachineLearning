#### LIBRERIAS ####
import math
import numpy as np
import nlp
import jaccard
import tdIdf as tdf
import cosenoVectorial as cosV
import time

#FUNCIONES
def matrizDistanciaPonderada(matriz1,matriz2,matriz3,ponderacines):
    matrizDistancias = np.zeros((len(matriz1),len(matriz1)))
    i = j = 0
    while True:
        matrizDistancias[i][j] = matriz1[i][j]*ponderacines[0] + matriz2[i][j]*ponderacines[1] + matriz3[i][j]*ponderacines[2]
        j += 1
        if j ==  len(matriz1):
            j = 0
            i += 1
        if i == len(matriz1):
            break
    return matrizDistancias

##IMPORTAR DATOS
start = time.process_time()
papers = nlp.importarCSV('papers.csv')
titulos = papers['title'].tolist()
keywords = papers['keywords'].tolist()
abstracts = papers['abstract'].tolist()
##NLP
titulosL = nlp.limpiarDocumento(titulos,'en')
keywordsL = nlp.limpiarDocumento(keywords,'en')
abstractsL = nlp.limpiarDocumento(abstracts,'en')
##JACCARD
#TITULOS
matrizTit = jaccard.matrizJaccard(titulosL)
print('Matriz de distancias de los Títulos\n',matrizTit)
#KEYWORDS
matrizKey = jaccard.matrizJaccard(keywordsL)
print('Matriz de distancias de las Keywords\n',matrizKey)
##FULL INVERTED INDEX -- TF-IDF -- COSENO VECTORIAL
#FULL INVERTED INDEX
diccionario={'tokens':[],'ocurrencias':[]}
diccionario['tokens']= nlp.indexacionToken(abstractsL)
diccionario['ocurrencias'] = nlp.ocurrencias(diccionario['tokens'],abstractsL)
#TF-IDF
matriz = tdf.bagWords(diccionario,abstractsL)
wtf = tdf.matrizPTF(matriz)
dF = tdf.documentF(wtf)
idf = tdf.IDF(dF,len(abstractsL))
tf_idf = tdf.TFIDF(wtf,idf)
#Coseno Vectorial
matrizN = cosV.matrizNormal(tf_idf)
matrizAbs = cosV.matrizDistacias(matrizN)
print('Matriz de distancias de los Abstracts\n',matrizAbs)
#MATRIZ DE DISTACIAS
ponderacion = [0.5,0.3,0.2]
Distancias = matrizDistanciaPonderada(matrizAbs,matrizKey,matrizTit,ponderacion)
print('Matriz de distancias\n',Distancias)
end = time.process_time()
print('El tiempo de ejecución del sistema fue:',end-start)