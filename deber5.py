#### LIBRERIAS ####
import math
import numpy as np
import nlp
import jaccard
import tdIdf as tdf
import cosenoVectorial as cosV

#FUNCIONES
# def matrizDistanciaPonderada(matriz1,matriz2,matriz3,ponderacines):
#     i = j = 0
#     while True:




##IMPORTAR DATOS
papers = nlp.importarCSV('papers.csv')
# titulos = papers['title'].tolist()
keywords = papers['keywords'].tolist()
# abstracts = papers['abstract'].tolist()

##NLP
# titulosL = nlp.limpiarDocumento(titulos,'en')
keywordsL = nlp.limpiarDocumento(keywords,'en')
print(keywordsL[len(keywordsL)-1])
print(jaccard.interseccion(keywordsL[len(keywordsL)-1],keywordsL[len(keywordsL)-1]))
print(jaccard.union(keywordsL[len(keywordsL)-1],keywordsL[len(keywordsL)-1]))
valor = jaccard.jaccard(keywordsL[len(keywordsL)-1],keywordsL[len(keywordsL)-1])
print(valor)
# abstractsL = nlp.limpiarDocumento(abstracts,'en')

##JACCARD
#TITULOS
# matrizTit = jaccard.matrizJaccard(titulosL)

#KEYWORDS
matrizKey = jaccard.matrizJaccard(keywordsL)
print(matrizKey[len(matrizKey)-1][len(matrizKey)-1])

##FULL INVERTED INDEX -- TF-IDF -- COSENO VECTORIAL
# #FULL INVERTED INDEX
# diccionario={'tokens':[],'ocurrencias':[]}
# diccionario['tokens']= nlp.indexacionToken(abstractsL)
# diccionario['ocurrencias'] = nlp.ocurrencias(diccionario['tokens'],abstractsL)
# # nlp.imprimirFII(diccionario['tokens'],diccionario['ocurrencias'])

# #TF-IDF
# matriz = tdf.bagWords(diccionario,abstractsL)
# # print(matriz)
# wtf = tdf.matrizPTF(matriz)
# # print(wtf)
# dF = tdf.documentF(wtf)
# # print(dF)
# idf = tdf.IDF(dF,len(abstractsL))
# # print(idf)
# tf_idf = tdf.TFIDF(wtf,idf)
# # print(tf_idf)

# #Coseno Vectorial
# matrizN = cosV.matrizNormal(tf_idf)
# # print(matrizN)
# # matrizDistanciaAbs = cosV.matrizDistacias(cosV.matrizNormal(tf_idf))
# matrizAbs = cosV.matrizDistacias(matrizN)
# print(matrizAbs)
# ponderacion = [0.5,0.3,0.2]