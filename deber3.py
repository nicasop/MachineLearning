import re

def limpiarDocumento (cole):    
    colecciontok=[]
    for documento in range (len(coleccion)):
        documentoaux = re.sub('[^A-Za-z0-9]+',' ', coleccion[documento])    
        documentoaux = documentoaux.lower()
        documentoaux = documentoaux.split()
        colecciontok.append(documentoaux)
    return colecciontok

def indexacionToken(coleccion):
    palabras=[]    
    for documento in colecciontok:
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
            vpos=[] 
            vaux1.append(doc+1)
            vaux1.append(len(obtenPos(tok,colDoc[doc])))
            vaux1.append(obtenPos(tok,colDoc[doc]))
            vaux.append(vaux1)
    return vaux

def ocurrencias (dic):
    vec=[]
    for token in dic:
        vec.append(tokenDoc(token,colecciontok))
    return vec

def imprimirFII(vecT,vecO):
    for j in range (len(vecT)):
        print(vecT[j],'-->',vecO[j])




coleccion= ["To do is to be. To be is to do.",
            "To be or not to be. I am what I am.",
            "I think therefore I am. Do be do be do.",
            "Do do do, da da da. Let it be, let it be."]

colecciontok=limpiarDocumento(coleccion)
diccionario={'tokens':[],'ocurrencias':[]}
diccionario['tokens']= indexacionToken(colecciontok)
diccionario['ocurrencias']= ocurrencias(diccionario['tokens'])
imprimirFII(diccionario['tokens'],diccionario['ocurrencias'])
