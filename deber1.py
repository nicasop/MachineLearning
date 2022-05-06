# Integrantes: 
# Diego Bedoya - Anthony Grijalva - Jhon Granda - Sebastian Sandoval - Alexis Villavicencio

def hashFunction(dividendo,divisor):
   return dividendo%divisor

def kvalue(x,B):
    return hashFunction(x,B-1) + 1

def rehashingFunction(posHashAn,k,B):
    return hashFunction(posHashAn + k,B)

def printHashTable(hashTable):
    for i in range(len(hashTable)):
        print(i,' -> ',hashTable[i])

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