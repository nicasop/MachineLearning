def clavePalabra(palabra,base):
    palabra = palabra.lower()
    codigo = 0
    tamPalabra = len(palabra)-1
    for i in range(len(palabra)): 
        codigo += ord(palabra[i])*base**(tamPalabra - i)
    return codigo

palabra = 'HOLA'
base = 2

if (base > 1):
    print(palabra,' -----> ',clavePalabra(palabra,base))
else:
    print('La base no puede ser inferior a 1')