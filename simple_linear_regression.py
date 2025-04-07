# Regresión Lineal Simple

# Explicaciones clave:
# - En este ejemplo, usamos la columna '2000' como variable independiente (X)
#   y la columna '2020' como variable dependiente (y).
# - Se asume que en tu CSV (internet_usage.csv) las columnas de los años (2000, 2020, etc.)
#   contienen datos numéricos sobre el uso de internet para cada país.
# - Ajusta los nombres de las columnas según tu CSV real.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Importar el dataset
dataset = pd.read_csv('internet_usage.csv')

# 2. Seleccionar la(s) columna(s) independiente(s) y dependiente
#    Suponemos que la columna '2000' existe y es numérica, y queremos predecir la columna '2020'.
X = dataset[['2000']].values   # Variable independiente
y = dataset['2020'].values     # Variable dependiente

# 3. Dividir el dataset en conjunto de entrenamiento y de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=1/3, 
    random_state=0
)

# 4. Entrenar el modelo de Regresión Lineal Simple
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 5. Predecir los resultados para el conjunto de prueba
y_pred = regressor.predict(X_test)

# 6. Visualizar los resultados en el conjunto de entrenamiento
plt.scatter(X_train, y_train, color='red')  
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Uso de Internet (2020) vs (2000) [Entrenamiento]')
plt.xlabel('Uso de Internet en 2000')
plt.ylabel('Uso de Internet en 2020')
plt.show()

# 7. Visualizar los resultados en el conjunto de prueba
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Uso de Internet (2020) vs (2000) [Prueba]')
plt.xlabel('Uso de Internet en 2000')
plt.ylabel('Uso de Internet en 2020')
plt.show()

