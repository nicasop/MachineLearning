# Deber 13 Evaluación Rendimiento Regresion 
# Integrantes: Diego Bedoya - Jhon Granda - Anthony Grijalva - Sebastian Sandoval - Alexis Villaviencio

# Librerias
import numpy as np
from sklearn.metrics import precision_score
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split

# Carga del dataset de iris
datos = datasets.load_iris()
datosInde = datos.data[:,0:3]
datosDepen = datos.data[:,3]
cant_datos = len(datosInde)

# Division entre training y test
train = 0.70
cant_train = int(train*cant_datos)
cant_test = cant_datos-cant_train

# Random Cross Validation
x_train,x_test,y_train,y_test=train_test_split(datosInde,datosDepen,train_size=train,random_state=12)

# Modelo de regresión: configuración y entrenamiento del modelo
reg = LinearRegression()
reg.fit(x_train,y_train)

# Prediccion
y_pred = reg.predict(x_test)

# Calculo de las metricas de rendimiento del modelo de regresion
print('------------- METRICAS PARA EVALUAR EL RENDIMIENTO DEL MODELO DE REGRESION -------------')
# MAE MEAN ABSOLUTE ERROR
print('MAE:',mean_absolute_error(y_test,y_pred))

#MSE MEAN SQUARE ERROR
print('MSE',mean_squared_error(y_test, y_pred,squared=True))

#RMSE ROOT MEAN SQUARE ERROR
print("RMSE:",mean_squared_error(y_test, y_pred,squared=False))

# R2 Coeficiente de determinación
r2 = r2_score(y_test, y_pred)
print('R2:',r2)

# R2 AJUSTADO
r2_adj = 1 - (1 - r2)*((len(y_test)-1)/(len(y_test)-len(reg.coef_)-1))
print('R2 Ajustado:',r2_adj)
