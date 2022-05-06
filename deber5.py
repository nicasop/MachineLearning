#### LIBRERIAS ####
import math
import numpy as np
import nlp
import jaccard
import tdIdf as tdf

# d1 = 'dias de lluvia'
# d2 = 'resbalo en un dia de lluvia'
# coleccion = [d1.split(),d2.split()]

# print(coleccion)
# print(jaccard.matrizJaccard(coleccion))

d1 = 'LA CASA DEL ARBOL, DENTRO DEL ARBOL'
d2 = 'ARBOLES RURALES, PARROQUIA RURAL'
d3 = 'FLORA Y FAUNA, DE ARBOLES ECUATORIANOS DE ARBOLES ENDEMICOS'
coleccion = [d1,d2,d3]
coleccionLim = nlp.limpiarDocumento(coleccion)
# print(coleccion)
# print(coleccionLim)

diccionario={'tokens':[],'ocurrencias':[]}
diccionario['tokens']= nlp.indexacionToken(coleccionLim)
diccionario['ocurrencias'] = nlp.ocurrencias(diccionario['tokens'],coleccionLim)
print(diccionario['ocurrencias'])
nlp.imprimirFII(diccionario['tokens'],diccionario['ocurrencias'])

matriz = tdf.bagWords(diccionario,coleccion)
# print(matriz)
wtf = tdf.matrizPTF(matriz)
print(wtf)

dF = tdf.documentF(wtf)
# print(dF)

idf = tdf.IDF(dF,len(coleccionLim))
print(idf)

tf_idf = tdf.TFIDF(wtf,idf)
print(tf_idf)