#!/usr/bin/env python
# coding: utf-8

# # TFM 2

# In[ ]:


Intentaremos predecir Total_Sales en función de datos de las diferentes tiendas


# ## IMPORTACION DE DATOS, TRATAMIENTO DE NA'S Y OUTLIERS

# In[1]:


#Importamos librerías:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyodbc


# In[2]:


#Cargamos los csv

# Conexión con MSSQL Server
server = 'DESKTOP-VUJ9ETK\SQLEXPRESS'
database = 'TFM_EnriqueRocho'

conn = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';Trusted_Connection=yes;')

# Query para seleccionar los datos de las tablas
query1 = 'SELECT * FROM all_tables'
query2 = 'SELECT * FROM Delivery_Route'
query3 = 'SELECT * FROM Sales_Rotura'

# Pasamos los resultados SQL a DataFrames
df_all_tables = pd.read_sql(query1, conn)
df_Delivery_Route = pd.read_sql(query2, conn)
df_Sales_Rotura = pd.read_sql(query3, conn)

# Cerramos la conexión
conn.close()

# Revisamos los datos de los Dataframes
print(df_all_tables.head()) 
print(df_Delivery_Route.head()) 
print(df_Sales_Rotura.head()) 


# In[3]:


df_all_tables['provincia'].fillna('Desconocido', inplace=True)
df_all_tables['poblacion'].fillna('Desconocido', inplace=True)

df_Delivery_Route['provincia'].fillna('Desconocido', inplace=True)
df_Delivery_Route['poblacion'].fillna('Desconocido', inplace=True)

df_Delivery_Route['Delivery_DAY'] = pd.to_datetime(df_Delivery_Route['Delivery_DAY'])
df_Delivery_Route['Route_DAY'] = pd.to_datetime(df_Delivery_Route['Route_DAY'])

df_Sales_Rotura['Sales_DAY'] = pd.to_datetime(df_Sales_Rotura['Sales_DAY'])
df_Sales_Rotura['OoS_DAY'] = pd.to_datetime(df_Sales_Rotura['OoS_DAY'])

df_Sales_Rotura['provincia'].fillna('Desconocido', inplace=True)
df_Sales_Rotura['poblacion'].fillna('Desconocido', inplace=True)


# In[4]:


#Preparamos outliers

from scipy import stats

threshold = 2
outliers2 = df_all_tables[np.abs(stats.zscore(df_all_tables['Total_Delivered'])) > threshold]


# In[5]:


threshold = 2
outliers4 = df_all_tables[np.abs(stats.zscore(df_all_tables['Total_Sales'])) > threshold]


# In[6]:


outliers_indices = outliers2.index.union(outliers4.index)
df_all_tables_no_outliers = df_all_tables.drop(outliers_indices)


# In[7]:


df_all_tables_no_outliers  = df_all_tables_no_outliers[(df_all_tables_no_outliers ['Total_Delivered'] >= 0)
                                                       & (df_all_tables_no_outliers ['Total_Sales'] >= 0)]


# In[8]:


threshold = 3
outliers6 = df_Sales_Rotura[np.abs(stats.zscore(df_Sales_Rotura['Sales_Uds'])) > threshold]

outlier_indices6 = outliers6.index

df_Sales_Rotura_no_outliers = df_Sales_Rotura.drop(outlier_indices6)
df_Sales_Rotura_no_outliers  = df_Sales_Rotura_no_outliers[(df_Sales_Rotura_no_outliers['Sales_Uds']>= 0)]


# In[9]:


df_all_tables_no_outliers.head()


# In[10]:


df_all_tables_no_outliers.columns


# ## PREPARACIÓN DE DATOS PARA LOS ALGORITMOS

# Para poder usar los algoritmos de Data Science, vamos a realizar varios cambios en los dataframes:
# 
# Elección de las columnas a tratar.
# 
# Data preprocessing: Realizaremos one-hot encoding para convertir las variables numéricas en formato numerico.
# 
# Scaling/Normalizar: Antes de realizar PCA escalaremos las columnas numéricas, incluidas las generadas con one-hot encoding usando técnicas como StandardScaler.
# 
# PCA: usaremos PCA a los datos numéricos escalados para reducir su dimensionalidad.

# #### ELECCIÓN DE VARIABLES

# Debido a que con todas las columnas que hay podríamos encontrar resultados pobres, vamos a trabajar en este caso solo con algunas columnas.
# 
# En este caso vamos a intentar obtener las ventas totales (Total_Sales) en función de características de las diferentes tiendas, por lo que haremos drop de las columnas que no hacen referencia a eso y algunas más para evitar problemas de rendimiento en la computación de los modelos.

# In[11]:


#Copiamos el dataframe para crear un df nuevo y haremos drop de columnas que no usaremos.

df_affiliated_sales = df_all_tables_no_outliers.copy()

df_affiliated_sales = df_affiliated_sales.drop(['Month','Affiliated_Code','Affiliated_NAME','POSTALCODE', 'poblacion'
                                                    ,'Product_Code','SIZE', 'Format','Cost_price',
                                                    'Sell_price','Margin','Total_Delivered','Total_OoS','Total_Sales'], axis=1)

df_affiliated_sales.head()


# #### DUMMY VARIABLES
Crearemos un dataframe categórico a partir del df que contiene las columnas con las que ibamos a trabajar: df_affiliated_sales. Seleccionaremos las columnas categóricas para convertir a numéricas bivariables (0 y 1) mediante one-hot encoding para que sea más fácil trabajar con ellas.
# In[12]:


categorical_columns = ['Engage', 'Management_Cluster', 'provincia','Location', 'Tam_m2']

df_affiliated_sales_cat = pd.get_dummies(df_affiliated_sales, columns=categorical_columns)

df_affiliated_sales_cat.head()


# In[13]:


df_affiliated_sales_cat.columns


# #### PCA

# PCA es principal component analysis, que permite realizar una reducción de dimensionalidad lineal de las variables y facilitar así la aplicación de algoritmos.
# 
# Con el proceso de conversión de columnas categóricas a numéricas se ha generado demasiadas columnas por lo que vamos a hacer un PCA para elegir qué variables pueden ser más representativas.

# In[62]:


from sklearn.decomposition import PCA

# Applicamos PCA inicialmente con un número de componentes muy alto
pca = PCA(n_components=66)
pca.fit(df_affiliated_sales_cat)
transformed_data = pca.transform(df_affiliated_sales_cat)


# Ahora revisamos los componentes óptimos de PCA:

# In[63]:


import matplotlib.pyplot as plt

pca = PCA()
pca.fit(transformed_data)

# Calculamos la cumulative explained variance.
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()


# Vemos que 20 componentes explican más del 80% de la varianza, por lo que podemos reducir los componentes a ese número. A partir de ahí, el resto de componentes contribuyen menos a capturar la estructura de los datos.

# Aplicamos PCA de nuevo con 20 componentes.

# In[64]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

pca = PCA(n_components=20)
pca.fit(df_affiliated_sales_cat)
pca_features = pca.transform(df_affiliated_sales_cat)


# ## REGRESIÓN LINEAL

# Aplicaremos un modelo simple, una regresión lineal usando el array de PCA. Intentaremos predecir valores de Total_Sales.

# In[65]:


# Dividimos los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(pca_features, df_all_tables_no_outliers['Total_Sales'], test_size=0.2, random_state=42)

# Iniciamos el modelo
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predecimos en el set de test
y_pred = regressor.predict(X_test)

# Calculamos Mean Squared Error (MSE) y R2 para medir la precisión del modelo.
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2): {r2}")


# El MSE mide la diferencia media cuadrada entre los valores reales y los predichos. En este caso la diferencia entre los valores reales y los predichos es de 135.5. Valores bajos representarían mayor precisión del modelo.
# 
# R-squared (R2) representa la proporción de varianza de la variable objetivo (Total_Sales) que es explicada por el resto de las variables del modelo. Un valor 0.031 sugiere que solo el 3% de la varianza es explicada por las features del modelo. Este valor debería más cercano a 1 (100%).
# 
# Parece por tanto, que la regresión lineal no está capturando la relación de la variable objetivo con el resto de variables. 

# Hacemos un plot de los valores reales de Total_Sales vs los predichos por este modelo

# In[66]:


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Total Sales')
plt.ylabel('Predicted Total Sales')
plt.title('Actual vs Predicted Total Sales')
plt.show()


# El plot debería dibujar una diagonal donde los valores de ambos ejes coincidieran, pero aquí tenemos muchos valores fuera de esa posible diagonal, lo que indica sobreestimación o infraestimación.

# ## CROSS-VALIDATION

# Vamos a usar cross-validation ahora para intentar mejorar el desempeño del modelo.
# 
# Los resultados de Cross Validation se derivan de un modelo de evaluación donde el dataset se divide en subsets para entrenamiento y testing multiples veces para revisar la precisión del modelo.  

# In[67]:


from sklearn.model_selection import cross_val_score, KFold

X = transformed_data
y = df_all_tables_no_outliers['Total_Sales']

# Inicializamos el modelo
model = LinearRegression()

# Seleccionamos el num de k-fold y aplicamos
k_folds = 5 
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Calculamos las métricas para este método:
mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

# Calculamos la media y desviación standard de MSE y R2.
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

print(f"Cross-Validation - Mean Squared Error (MSE): {mean_mse} ± {std_mse}")
print(f"Cross-Validation - R-squared (R2): {mean_r2} ± {std_r2}")


# Los resultados mediante cross-validation obtuvieron resultados ligeramente mejores que el modelo de regresión lineal en MSE (135.5 vs 134.498) y en R2 (0.039 vs. 0.031).
# En el caso de MSE y R2 para Cross-Validation la pequeña desviación indica consistencia entre los diferentes folds empleados (5).
# 
# Cross-Validation tampoco supone una mejora sensible.

# ## BASELINE

# Para mejorar la precisión del modelo, podemos llevar a cabo más acciones.
# Por ejemplo realizar una comparación con el Baseline, donde intentaremos predecir la media de la variable target (Total_Sales) para todas las muestras:

# In[68]:


# Calculamos la media de la variable objetivo:
mean_sales = np.mean(y)

# Creamos array de la media de ventas para conseguir la misma longitud de y_test
mean_sales_predictions = np.full_like(y_test, mean_sales)

# Calculamos MSE para el Baseline (prediciendo la media)
baseline_mse = mean_squared_error(y_test, mean_sales_predictions)
print("Baseline MSE (Predicting Mean):", baseline_mse)


# Hemos obtenido por tanto estos resultados:
#     
# MSE del modelo de regresión lineal: 135.56
# Baseline MSE (Predicción de media): 140.04
# 
# Esto implica que el modelo de regresión MSE funciona un poco mejor que una baseline que simplemente predice los valores de la media para todas las muestras.
# 
# A pesar de que la diferencia no es substancial, indica que el modelo proporciona algo de valor con respecto a una predicción básica de media. 
# 
# Sin embargo parece que hay espacio de mejora, ya que la diferencia no es muy grande.

# ## FEATURE ENGINEERING: POLYNOMIAL FEATURES

# Hay varias maneras de mejorar los resultados, una de ellas es mediante feature engineering.
# 
# En este caso usaremos una Polynomial Regression, que captura relaciones no lineales usando término polinomicos en las variables de predicción. 

# In[69]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Definimos los grados de polynomial features
degree = 2 

# Creación de pipeline con PolynomialFeatures y LinearRegression
poly_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Ajuste del modelo a los datos de train:
poly_reg.fit(X_train, y_train)

# Predicciones:
predictions = poly_reg.predict(X_test)

# Mean Squared Error (MSE) y R-squared
poly_mse = mean_squared_error(y_test, predictions)
poly_r2 = r2_score(y_test, predictions)

print(f"Polynomial Regression - Mean Squared Error (MSE): {poly_mse}")
print(f"Polynomial Regression - R-squared (R2): {poly_r2}")


# La regresión polinomica tiene un desempeño un poco mejor que la regresión lineal (134.27 vs 135.5).
# Los valores de R2 también han mejorado (0.04 vs 0.031), por lo que la regresión polinomica explica un poco más de varianza de la variable objetivo Total_Sales.
# 
# A pesar de la mejora aun siguen siendo resultados muy pobres.

# # Random Forest

# Intentaremos ver si un modelo Random Forest puede dar mejores resultados para la predicción de Total_Sales.
# Este modelo permite manejar relaciones e interacciones entre features más complejas.

# In[70]:


from sklearn.ensemble import RandomForestRegressor

# Creamos el modelo
rf_model = RandomForestRegressor(n_estimators=25, random_state=42) 

# Hacemos ajuste del modelo sobre los sets de entrenamiento
rf_model.fit(X_train, y_train)

# Predecimos sobre el set de test
rf_predictions = rf_model.predict(X_test)

# MSE y R2 para Random Forest
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print(f"Random Forest - Mean Squared Error (MSE): {rf_mse}")
print(f"Random Forest - R-squared (R2): {rf_r2}")


# El MSE ha bajado a 129.70, y R2 ha subido a 0.073 mejorando al resto de modelos.
# 
# Hemos dejado el n_estimators en 25 porque un num mayor incrementa el tiempo de computación pero no aporta una gran mejora en MSE y R2.

# Vamos a revisar un poco más en profundidad este modelo con algunas otras métricas.

# In[71]:


from sklearn.metrics import mean_absolute_error, median_absolute_error, explained_variance_score

#  Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, rf_predictions)
print(f"Mean Absolute Error (MAE): {mae}")

#  Median Absolute Error (MedAE)
medae = median_absolute_error(y_test, rf_predictions)
print(f"Median Absolute Error (MedAE): {medae}")

#  Explained Variance Score
explained_variance = explained_variance_score(y_test, rf_predictions)
print(f"Explained Variance Score: {explained_variance}")


# Mean Absolute Error (MAE): Mide la media absoluta de diferencia entre los valores predecidos y los reales. Un MAE de 8.68 indica de media que las predicciones de este modelo están ±8.61 unidades separadas de los valores reales.
# 
# Median Absolute Error (MedAE): MedAE calcula la mediana absoluta de esa diferencia. El valor 7.09 indica que la mitad de las predicciones tienen un error absoluto de menos de 7.09, lo que puede ser más robusto a outliers que a MAE.
# 
# Explained Variance Score: Nos indica la proporción de varianza de la variable objetivo que el modelo captura. Un 0.07 indica que el modelo explica sobre el 7% de varianza de la variable objetivo.
# 
# Parece que este modelo tampoco explica una parte substancial de la variabilidad del objetivo.

# Visualizaremos los valores obtenidos en RF

# In[24]:


import matplotlib.pyplot as plt

# Valores actuales vs valores de predicción
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_predictions, alpha=0.5)
plt.xlabel('Actual Total Sales')
plt.ylabel('Predicted Total Sales')
plt.title('Actual vs Predicted Total Sales (Random Forest)')
plt.show()


# De nuevo lo ideal seria tener una diagonal en la que los valores de predicción serían los mismos que los reales, lo que no sucede (aunque se ha mejorado la tendencia) por lo que se están sobreestimando o infraestimando los valores de Total_Sales.
# 
# Por tanto, parece que no podemos mejorar los resultados sensiblemente con los features que hemos seleccionado.
# Tal vez una mejor opción sea revisar qué features pueden ser más adecuados para predecir Total_Sales.

# ## REVISIÓN DE FEATURES

# Intentaremos acotar los features que nos pueden proporcionar una predicción para Total_Sales más robusta.

# In[73]:


#Vamos a crear un nuevo df:

df_affiliated_sales2 = df_all_tables_no_outliers.copy()

df_affiliated_sales2 = df_affiliated_sales2.drop(['Affiliated_NAME','POSTALCODE', 'poblacion', 'Product_Code','SIZE', 'Format'
                                                    ], axis=1)

df_affiliated_sales2.columns


# In[74]:


#Creación de df categórico de df_affiliated_sales_cat2.

# Trabajaremos con las siguientes columnas categóricas, las mencionamos explícitamente para que se incluya 
#también Month y Engage como categórica:

categorical_columns2 = ['Month', 'Affiliated_Code', 'provincia', 'Engage', 'Management_Cluster',
       'Location', 'Tam_m2']
df_affiliated_sales_cat2 = pd.get_dummies(df_affiliated_sales2, columns=categorical_columns2)


# In[75]:


df_affiliated_sales_cat2.columns


# #### SelectKBest

# Ahora vamos a utilizar SelectKBest para seleccionar las columnas que más influyen en Total_Sales.
# En este caso seleccionaremos k=10, que nos dará las mejores 10 columnas:

# In[76]:


from sklearn.feature_selection import SelectKBest, f_regression

# Separamos los datos en features (X) y la variable objetivo ('Total_Sales')
X2 = df_affiliated_sales_cat2.drop(columns=['Total_Sales'])  # Features
y2 = df_affiliated_sales_cat2['Total_Sales']  # Variable Objetivo

# Inicializamos SelectKBest con la scoring function (f_regression para problemas de regresion)
selector = SelectKBest(score_func=f_regression, k=10) 

# Ajustamos el selector a los datos
X_new = selector.fit_transform(X2, y2)

# Obtenemos las columnas seleccionadas
selected_columns = X2.columns[selector.get_support()]

# Creamos un df con las columnas seleccionadas
selected_features_df = pd.DataFrame(X_new, columns=selected_columns)

# Mostramos las features más importantes
print("Selected Features:")
print(selected_features_df.head())


# #### RANDOM FOREST

# Además un modelo Random Forest nos va a permitir realizar una segunda revisión de las features más importantes sobre Total_Sales

# In[77]:


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Inicializamos RF
rf = RandomForestClassifier(n_estimators=10, random_state=42)

# Ajuste de modelo
rf.fit(X2, y2)

# Features por importancia
feature_importances = pd.Series(rf.feature_importances_, index=X2.columns)
feature_importances = feature_importances.sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 8))
feature_importances.head(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.show()


# Vemos un resultado similar al obtenido en SelectKBest aunque con más importancia en los meses y columnas numéricas en el caso de RF y Engage en el caso de SelectKBest.
# 
# En todo caso, Total_Delivered, Total_OoS, Month_5, Month_10, provincia_Madrid, Engage_1, 2, 3, Location_City son algunas de las columnas que más impactan a Total_Sales.
# 
# Vamos a intentar por tanto mejorar los resultados del modelo lo máximo posible seleccionando solo algunas columnas.
# 
# Nos centraremos en: 'Total_Delivered','Total_OoS', 'Month_3','Month_4','Month_5','Month_6','Month_7', 'Month_10', 'Engage_1', 
#                      'Engage_2','Engage_3','provincia_Madrid','Location_CITY'  
# 
# Hemos descartado Sell_Price y Cost_Price ya que las sustituiremos por Margin con la que tienen un alto grado de relación.

# #### SELECCIÓN DE NUEVOS FEATURES

# In[105]:


# Especificamos las columnas
specified_columns = ['Margin','Total_Delivered','Total_OoS','Month_5','Month_10','Engage_1', 
                     'Engage_2','Engage_3','provincia_Madrid','Location_CITY',    
                     ]
# Subset del dataframe
df_affiliated_sales_selected = df_affiliated_sales_cat2[specified_columns].copy()          


# In[99]:


df_affiliated_sales_selected.columns


# In[106]:


#Normalizamos las columnas numéricas que utilizaremos que en este caso solo será Total_Delivered:

from sklearn.preprocessing import MinMaxScaler

# Iniciamos MinMaxScaler
scaler = MinMaxScaler()

# Normalizamos
df_affiliated_sales_selected[['Margin','Total_Delivered','Total_OoS']] = scaler.fit_transform(df_affiliated_sales_selected[['Margin','Total_Delivered','Total_OoS']])


# In[107]:


#Aplicamos un PCA inicial

from sklearn.decomposition import PCA

pca2 = PCA(n_components=10) 

pca_result2 = pca2.fit_transform(df_affiliated_sales_selected)


# In[102]:


#Obtenemos el PCA óptimo

import matplotlib.pyplot as plt

pca2 = PCA()
pca2.fit(pca_result2)

# Calculamos la varianza acumulada
cumulative_variance = np.cumsum(pca2.explained_variance_ratio_)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()


# In[110]:


#Aplicamos PCA para 10, ya que se ha comprobado una bajada de precisión muy acusada con unos 7.

pca2 = PCA(n_components=10) 

pca_result2 = pca2.fit_transform(df_affiliated_sales_selected)


# #### NUEVA REGRESIÓN LINEAL

# In[111]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

y2 = df_all_tables_no_outliers['Total_Sales']

# Split de los datos del PCA y la variable objetivo en training y testing sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(pca_result2, y2, test_size=0.2, random_state=42)

# Inicializamos la regresión lineal
model = LinearRegression()

# Ajuste del modelo en datos de training
model.fit(X_train2, y_train2)

# Predicción en el test set
predictions = model.predict(X_test2)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test2, predictions)
print(f"Mean Squared Error (MSE): {mse}")

# R-squared (R2)
r2 = r2_score(y_test2, predictions)
print(f"R-squared (R2): {r2}")


# Como vemos la diferencia con los anteriores modelo es muy grande al usar las columnas más adecuadas. Hemos obtenido un MSE más pequeño 40.79 lo que indica que de media la diferencia entre los valores predichos y los reales es más pequeña.
# 
# R2 da un valor 0.70 lo que indica que este modelo explica el 70% de la varianza de Total_Sales.
# 
# Es posible que hayamos hecho un overfit del modelo y que solo vaya a funcionar bien con los datos de entrenamiento, por lo que tendremos que valorar otras métricas.

# In[112]:


import matplotlib.pyplot as plt

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test2, predictions, color='blue', alpha=0.5)  # Scatter plot
plt.plot([min(y_test2), max(y_test2)], [min(y_test2), max(y_test2)], color='red')  # Diagonal line for reference
plt.title('Actual vs Predicted Total_Sales')
plt.xlabel('Actual Total_Sales')
plt.ylabel('Predicted Total_Sales')
plt.show()


# Vemos en esta gráfica que los valores reales se ajustan más con los predichos.

# Utilizaremos un Cross-Validation para asegurarnos si el modelo es adecuado:

# In[113]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Inicializamos la Regresión Lineal
model = LinearRegression()

# 3 fold cross validation para evitar problemas de tiempo de procesado
scores = cross_val_score(model, pca_result2, y2, cv=3, scoring='r2')

# Imprimimos los resultados
print("R-squared scores for each fold:", scores)

# Calculamos la media y desviación standard de los valores R2 obtenidos
mean_r2 = np.mean(scores)
std_r2 = np.std(scores)
print(f"\nMean R-squared (R2) across folds: {mean_r2}")
print(f"Standard deviation of R-squared (R2) across folds: {std_r2}")


# Los valores entre los folds se mantienen en 0.70 con poca variación lo que indica que este modelo se podría extrapolar a otros datasets no utilizados.
# 
# La media de R2 es 0.70 también, por lo que el modelo explica un 70% de la varianza de Total_sales de media.
# 
# La desviación standard 0.0019 indica que la precisión del modelo se mantiene estable entre diferentes subsets.

# #### NUEVA POLYNOMIAL REGRESSION

# Podríamos aplicar también polynomial features para hacer el mismo proceso que hemos hecho antes, aunque los resultados no cambiarán demasiado:

# In[114]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Inicializamos PolynomialFeatures y LinearRegression
degree = 2  
poly = PolynomialFeatures(degree=degree)
model2 = make_pipeline(poly, LinearRegression())

# Ajustamos el modelo a los datos de train
model2.fit(X_train2, y_train2)

# Predecimos en el set de test
predictions2 = model2.predict(X_test2)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test2, predictions2)
print(f"Mean Squared Error (MSE): {mse}")

#  R-squared (R2)
r2 = r2_score(y_test2, predictions2)
print(f"R-squared (R2): {r2}")


# Con polynomial features con un grado de  (que no podemos aumentar por problemas de tiempo de computación) obtenemos un MSE de 35.82, con R2 en 0.74.
# 
# Por lo tanto obtenemos unos resultados algo mejores.

# #### NUEVO RANDOM FOREST

# Por seguir comparando con los resultados obtenidos con los features anteriores, seguimos el mismo proceso aplicando igualmente un RandomForestRegressor

# In[115]:


from sklearn.ensemble import RandomForestRegressor

# Inicializamos el RF
rf_model2 = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42)

# Ajustamos el modelo a los datos de train
rf_model2.fit(X_train2, y_train2)

# Predecimos en el set de test
rf_predictions2 = rf_model2.predict(X_test2)

# Mean Squared Error (MSE)
rf_mse = mean_squared_error(y_test2, rf_predictions2)
print(f"Random Forest Mean Squared Error (MSE): {rf_mse}")

# R-squared (R2)
rf_r2 = r2_score(y_test2, rf_predictions2)
print(f"Random Forest R-squared (R2): {rf_r2}")


# Como vemos este modelo parece ser aun más adecuado, con un MSE que baja a 32.88 y un R2 0.76, por lo que se explicaría un 76% de la variabilidad de Total_Sales con el modelo.

# In[116]:


import matplotlib.pyplot as plt

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test2, rf_predictions2, color='blue', alpha=0.5)  # Scatter plot
plt.plot([min(y_test2), max(y_test2)], [min(y_test2), max(y_test2)], color='red')  # Diagonal line for reference
plt.title('Actual vs Predicted Total_Sales')
plt.xlabel('Actual Total_Sales')
plt.ylabel('Predicted Total_Sales')
plt.show()


# Revisamos también un CrossFold para el modelo RF:

# In[117]:


rf_cv_scores2 = cross_val_score(rf_model2, pca_result2, y, cv=3, scoring='r2')

print("Random Forest R-squared scores for each fold:", rf_cv_scores2)

rf_mean_r2 = np.mean(rf_cv_scores2)
rf_std_r2 = np.std(rf_cv_scores2)
print(f"\nRandom Forest Mean R-squared (R2) across folds: {rf_mean_r2}")
print(f"Random Forest Standard deviation of R-squared (R2) across folds: {rf_std_r2}")


# Vemos que los valores de Cross Validation (cv=3) para RF se mantienen entre 0.75 y 0.76 con una variación entre folds de 0.0039

# Como conclusión podemos ver que en el caso de este dataset es mucho más determinante para obtener buenos resultados el uso de los componentes adecuados en lugar de pasar de un modelo de regresión lineal a uno de RF o intentar mejorarlo con polynomial features.
# 
# Esto puede suponer que tengamos que ceñirnos a determinadas columnas en favor de obtener una mayor precisión. Tal vez el punto intermedio entre columnas que nos resulten más interesantes y las columnas óptimas nos aporte unos resultados también adecuados en término medio.

# ## PREDICCIONES

# Creamos un dataframe para usar en predicciones. Se crea una columna nueva Month_11 y se hace una regresión lineal para añadir datos a esa nueva columna

# In[119]:


from sklearn.linear_model import LogisticRegression

# Listamos las columnas para la predicción (excluyendo un posible Month_11)
columns_for_prediction = [col for col in df_affiliated_sales_selected.columns if col != 'Month_11']

# Separamos features para training
X_train3 = df_affiliated_sales_selected[columns_for_prediction]

# Creamos un dataframe vacío para almacenar el Month_11
df_with_predicted_month_11 = df_affiliated_sales_selected.copy()

# Hacemos ajuste del modelo de regresión lineal usando Month_10 para la predicción
model = LogisticRegression()
model.fit(X_train3, df_affiliated_sales_selected['Month_10'])

# Predecimos los valores de 'Month_11'
predicted_month_11 = model.predict(X_train3)

# Asignamos los valores predichos al nuevo df
df_with_predicted_month_11['Month_11'] = predicted_month_11

print(df_with_predicted_month_11)


# In[45]:


df_with_predicted_month_11.columns


# Ahora predeciremos los valores de la columna Total_Sales usando el modelo RF entrenado.

# Aplicamos el mismo PCA al nuevo df_with_predicted_month_11:

# In[120]:


#Aplicamos PCA para 10 componentes

pca3 = PCA(n_components=10) 

pca_result_predicted_month_11 = pca3.fit_transform(df_with_predicted_month_11)


# In[121]:


pca_result_predicted_month_11


# In[122]:


# Predecimos 'Total_Sales' para el nuevo df
predicted_total_sales = rf_model2.predict(pca_result_predicted_month_11)

# Asignamos los valores al df
df_with_predicted_month_11['Predicted_Total_Sales'] = predicted_total_sales

print(df_with_predicted_month_11)


# In[123]:


df_with_predicted_month_11.head()


# In[124]:


df_with_predicted_month_11.columns


# In[51]:


#Filtramos los valores para Month_11 para obtener solo las filas que conntienen este mes

df_with_predicted_month_11 = df_with_predicted_month_11[df_with_predicted_month_11['Month_11'] == 1]


# In[52]:


df_with_predicted_month_11.head(20)


# Month_11 coincide con Month_10 porque es el mes que hemos usado para conseguir datos de Month_11 en la reg lineal.
# Nos quedaremos solo con los datos de Month_11 igualmente.

# In[125]:


df_with_predicted_month_11.drop(['Month_5', 'Month_10'], axis=1, inplace=True)


# In[126]:


df_with_predicted_month_11.head()


# In[ ]:


Por último cargaremos este nuevo df en la misma base de datos de donde obtuvimos los dataframes originales


# In[55]:


from sqlalchemy import create_engine
import pyodbc

# Detalles de conexión
server = 'DESKTOP-VUJ9ETK\\SQLEXPRESS' 
database = 'TFM_EnriqueRocho'
driver = '{SQL Server}'
trusted_connection = 'yes'

# String de conexión
conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection};'

# Conexión con pyodbc
conn = pyodbc.connect(conn_str)

# Creción de SQLAlchemy engine usando la conexión
engine = create_engine(f'mssql+pyodbc://', creator=lambda: conn)

# Cambiamos el nombre de la tabla
table_name = 'Prediccion_Sales'

# Subimos el DataFrame a SQL Server
df_with_predicted_month_11.to_sql(table_name, engine, index=False, if_exists='replace')

print("DataFrame uploaded to SQL Server successfully!")


# In[ ]:




