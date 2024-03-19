#!/usr/bin/env python
# coding: utf-8

# # TFM 3

# En este caso vamos a intentar predecir Total_OoS (uds de rotura de stock) en función de características de los productos.

# ## IMPORTACION DE DATOS, TRATAMIENTO DE NA'S Y OUTLIERS

# In[213]:


#Importamos librerías:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyodbc


# In[214]:


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


# In[215]:


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


# In[216]:


#Preparamos outliers

from scipy import stats

threshold = 2
outliers2 = df_all_tables[np.abs(stats.zscore(df_all_tables['Total_Delivered'])) > threshold]


# In[217]:


threshold = 2
outliers4 = df_all_tables[np.abs(stats.zscore(df_all_tables['Total_Sales'])) > threshold]


# In[218]:


outliers_indices = outliers2.index.union(outliers4.index)
df_all_tables_no_outliers = df_all_tables.drop(outliers_indices)


# In[219]:


df_all_tables_no_outliers  = df_all_tables_no_outliers[(df_all_tables_no_outliers ['Total_Delivered'] >= 0)
                                                       & (df_all_tables_no_outliers ['Total_Sales'] >= 0)]


# In[220]:


df_all_tables_no_outliers.columns


# In[221]:


threshold = 3
outliers6 = df_Sales_Rotura[np.abs(stats.zscore(df_Sales_Rotura['Sales_Uds'])) > threshold]

outlier_indices6 = outliers6.index

df_Sales_Rotura_no_outliers = df_Sales_Rotura.drop(outlier_indices6)
df_Sales_Rotura_no_outliers  = df_Sales_Rotura_no_outliers[(df_Sales_Rotura_no_outliers['Sales_Uds']>= 0)]


# ## PREPARACIÓN DE DATOS PARA LOS ALGORITMOS

# En este caso como vamos a intentar obtener Total_OoS (uds de rotura de stock) en función de características de los productos, por lo que haremos drop de las columnas o variables que no hacen referencia a ese ámbito:

# #### ELECCIÓN DE VARIABLES

# In[222]:


#Copiamos el dataframe para crear un df nuevo y haremos drop de columnas que no usaremos a priori.

df_product = df_all_tables_no_outliers.copy()

df_product = df_product.drop(['Affiliated_Code', 'Affiliated_NAME', 'POSTALCODE',
       'poblacion', 'provincia', 'Engage', 'Management_Cluster', 'Location',
       'Tam_m2'], axis=1)

df_product.head()


# #### DUMMY VARIABLES

# Creamos el df categórico, indicando las columnas que queremos transformar:

# In[223]:


# Trabajaremos con las siguientes columnas categóricas:
categorical_columns = ['Month','Product_Code', 'SIZE', 'Format']

df_product_cat = pd.get_dummies(df_product, columns=categorical_columns)


# In[224]:


df_product_cat.head()


# #### SelectKBest

# Usamos SelectKBest para revisar entre todas las nuevas obtenidas, cuales son las columnas que más impactan en Total_OoS

# In[225]:


#Revisamos las columnas que más impactan en Total_OoS:

from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

# Separamos los datos en features (X) y la variable objetivo ('Total_OoS')
X = df_product_cat.drop(columns=['Total_OoS'])  # Features
y = df_product_cat['Total_OoS']  # Variable Objetivo

# Inicializamos SelectKBest con la scoring function (f_regression para problemas de regresion)
selector = SelectKBest(score_func=f_regression, k=10) 

# Ajustamos el selector a los datos
X_new = selector.fit_transform(X, y)

# Obtenemos las columnas seleccionadas
selected_columns = X.columns[selector.get_support()]

# Creamos un df con las columnas seleccionadas
selected_features_df = pd.DataFrame(X_new, columns=selected_columns)

# Mostramos las features más importantes
print("Selected Features:")
print(selected_features_df.head())


# Primero usaremos algunas de estas columnas, además de algunas más relacionadas con Producto en un nuevo subset para ver qué resultados obtenemos.

# In[226]:


#Creamos el nuevo subset:

specified_columns2 = ['Total_OoS','Cost_price', 'Sell_price','Month_3', 'Month_4', 'Month_5', 'Month_6',
                    'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7',
       'Month_8', 'Month_9', 'Month_10', 
       'Product_Code_Natu408','Product_Code_Dome206','Product_Code_Dome427','Product_Code_Natu079','Product_Code_Brit627',
       'Product_Code_Natu969','Product_Code_Dome164', 'Product_Code_Natu122','Product_Code_Brit700', 'Product_Code_Dome213',
       'Product_Code_Inte404','Product_Code_Dome363','Product_Code_Dome762','Product_Code_Dome459','Product_Code_Natu723',
       'Product_Code_Inte327', 'Product_Code_Dome615','Product_Code_Dome797','Product_Code_Dome527','Product_Code_Brit555',  
       'Product_Code_Natu461','Product_Code_Dome104',
       'SIZE_85', 'SIZE_142', 'SIZE_190','SIZE_317','SIZE_283','SIZE_125','SIZE_114', 'SIZE_481',     
       'Format_ASL', 'Format_ATA',
       'Format_ETO']

df_product_oos = df_product_cat[specified_columns2].copy()

df_product_oos.head()


# In[227]:


df_product_oos.columns


# La idea inicial era realizar un pca para reducir la dimensionalidad, hacer clustering y aplicar una regresión lineal una vez optimizado todo.

# Estamos excluyendo la target variable Total_OoS ya que si la incluimos con el resto de features puede condicionar el peso del clustering.

# Revisamos de nuevo la cantidad óptima de clusters

# Con este resultado tal vez es un poco complicado discernir la relación entre clustering, componentes pca y features.
# No parece haber una separación clara entre clusters ni que los puntos del mismo cluster estén próximos.
# 
# Es posible además que sea mucho más valioso aplicar clustering sobre columnas más concretas.
# 
# En este caso vamos a intentar también ceñirnos un poco más a las columnas que más impactaban a Total_OoS, aunque en el caso de Product_Code y Size nos limitaremos solo a las columnas que más peso tenían en el df como vimos en las gráficas.
# 
# Aunque este nuevo análisis no nos dará información de qué productos estarán fuera de stock en un determinado mes,
# sí que nos servirá para saber qué grupos de productos se comportan de manera similar en cuanto a rotura para determinados meses.

# #### StandardScaler, Clustering y PCA

# In[228]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

# Separamos 'Total_OoS' como variable objetivo
y = df_product_oos['Total_OoS'] 

# Datos que usaremos en clustering y PCA
X_for_clustering = df_product_oos.drop(columns=['Total_OoS'])  # Data for clustering

# Escalamos las variables numéricas
numerical_columns = ['Cost_price', 'Sell_price']
scaler_for_pca = StandardScaler()


# In[229]:


X_for_pca_scaled = X_for_clustering.copy()
X_for_pca_scaled[numerical_columns] = scaler_for_pca.fit_transform(X_for_clustering[numerical_columns])


# Realizamos K-means clustering para 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_for_clustering)

# Add cluster labels to the data
X_for_clustering['Cluster'] = clusters


# Revisamos el num óptimo de Clusters:

# In[230]:


wcss = []

#Num máximo de clusters a testear:
max_clusters = 10
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_for_clustering)
    wcss.append(kmeans.inertia_)

# Plot de inertia:
plt.figure(figsize=(8, 6))
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()


# El num óptimo de clusters parece ser 2-3 según lo que veremos en el siguiente plot de inertia. Mantendremos 3 sobretodo para tener una visualización más segmentada.

# Además revisamos el num adecuado de componentes PCA:

# In[231]:


# Ajuste del num máximo de componentes
optimal_components =  15
pca = PCA(n_components=optimal_components)
pca_result = pca.fit_transform(X_for_pca_scaled)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance Ratio')
plt.show()


# Con 11 componentes se explica más del 80% de la varianza, por lo que usaremos 11 como componentes óptimos.

# In[232]:


from sklearn.decomposition import PCA

# Applicamos de nuevo PCA con los componentes óptimos
optimal_components = 11
pca = PCA(n_components=optimal_components)
pca_result = pca.fit_transform(X_for_pca_scaled)

# DF para los componentes PCACreate a DataFrame with PCA components
df_pca = pd.DataFrame(data=pca_result, columns=[f'PCA_{i+1}' for i in range(optimal_components)], index=X_for_clustering.index)

# Concatenamos 'df_pca' con los datos de cluster
X_clustered = pd.concat([df_pca, X_for_clustering['Cluster']], axis=1)

# Concatenamos con 'Total_OoS'
df_processed = pd.concat([X_clustered, y], axis=1)


# In[233]:


df_processed.columns


# In[234]:


#Revisamos Nan por posibles errores de concat en la creación de df_processed

df_processed.isnull().sum()


# ## APLICACIÓN DE ALGORITMOS

# #### Regresión Lineal

# Aplicaremos primero una regresión lineal para ver el comportamiento del dataset obtenido después de clustering, PCA y StandardScaler con la variable objetivo Total_OoS.

# In[235]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Separamos las features de la variable objetivo
X2 = df_processed.drop(columns=['Total_OoS']) 
y2 = df_processed['Total_OoS']

# Obtenemos los sets testing y training
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Inicializamos el modelo
reg = LinearRegression()
reg.fit(X_train2, y_train2)

# Predecimos sobre el set de testing
y_pred2 = reg.predict(X_test2)

# Evaluamos el modelo
mse = mean_squared_error(y_test2, y_pred2)
r2 = r2_score(y_test2, y_pred2)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")


# Como vemos el MSE es bastante alto y el R2 explica una porción muy pequeña de la varianza de Total_OoS (2,9%).Podemos hacer visualizaciones para obtener más información.

# Realizamos primero una visualización 3D de clustering de los 3 primeros componentes

# In[236]:


# Visualización 3 primeros componentes.
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca['PCA_1'], df_pca['PCA_2'], df_pca['PCA_3'], c=X_for_clustering['Cluster'], cmap='viridis', alpha=0.5)
ax.set_xlabel('PCA_1')
ax.set_ylabel('PCA_2')
ax.set_zlabel('PCA_3')
ax.set_title('PCA Visualization with Cluster Assignments')
plt.show()


# Visualizamos los resultados de Predicción Total_OoS contra los valores predichos:

# In[237]:


plt.scatter(y_test2, y_pred2)
plt.title('Linear Regression: Predicted vs. Actual Total_OoS')
plt.xlabel('Actual Total_OoS')
plt.ylabel('Predicted Total_OoS')
plt.show()


# En general se están prediciendo valores muy bajos, y esto puede ser debido al gran peso del valor 0 para Total_OoS, ya que hemos visto que hay muchísimos valores 0 y pocos que no son 0. Esto puede condicionar las predicciones de los modelos, como puede ser este caso.
# 
# Igualmente vamos a intentar aplicar otros modelos.

# #### Random Forest

# Aplicaremos un random forest haciendo una selección de las 10 features con más peso sobre la variable objetivo Total_OoS para intentar mejorar la precisión del modelo.

# In[238]:


from sklearn.ensemble import RandomForestRegressor

# Inicializamos el modelo
forest = RandomForestRegressor(random_state=42)

# Ajustamos el modelo
forest.fit(X_train2, y_train2)

# Revisamos la importancia de las features
feature_importances = forest.feature_importances_

# Creamos un df con la importancia de las features y las columnas correspondientes
feature_importance_df = pd.DataFrame({'Feature': X2.columns, 'Importance': feature_importances})

# Ordenamos por importancia
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Seleccionamos las 10 features más importantes
top_features = feature_importance_df.head(10)['Feature'].tolist()

# Vovemos a entrenar el modelo con los features seleccionados
reg = LinearRegression()
reg.fit(X_train2[top_features], y_train2)

# Predecimos sobre el test set con los features seleccionados
y_pred3 = reg.predict(X_test2[top_features])

# Evaluamos el modelo
mse = mean_squared_error(y_test2, y_pred3)
r2 = r2_score(y_test2, y_pred3)

print(f"Mean Squared Error with selected features: {mse}")
print(f"R-squared Score with selected features: {r2}")


# No obtenemos una mejora con este método.

# #### GradientBoostingRegressor

# Antes de realizar más cambios, vamos a intentarlo con otro modelom Gradient Boosting Regressor.

# In[239]:


from sklearn.ensemble import GradientBoostingRegressor

# Inicializamos el modelo
grad_boost = GradientBoostingRegressor(random_state=42)

# Ajustamos el modelo
grad_boost.fit(X_train2, y_train2)

# Predecimos sobre el test set
y_pred4 = grad_boost.predict(X_test2)

# Evaluamos el modelo
mse = mean_squared_error(y_test2, y_pred4)
r2 = r2_score(y_test2, y_pred4)

print(f"Mean Squared Error with Gradient Boosting: {mse}")
print(f"R-squared Score with Gradient Boosting: {r2}")


# Como vemos en este caso se han mejorado los resultados, pero no es una mejora sensible que nos permita decidirnos este modelo. 

# #### Mejoras en PCA

# Vamos a revisar los diferentes componentes PCA y su peso con respecto a la variable objetivo:

# In[240]:


from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(df_pca, y2)

# Conseguimos puntuaciones y p-values para cada feature
feature_scores = pd.DataFrame({'Feature': df_pca.columns, 'Score': selector.scores_, 'p-value': selector.pvalues_})
feature_scores.sort_values(by='Score', ascending=False, inplace=True)
print(feature_scores)


# Vamos a intentar aplicar una regresión lineal a partir de los componentes PCA con mayor peso sobre la variable objetivo.

# In[241]:


# Usamos los componentes top 5 según los Score obtenidos
k = 5 
top_components = feature_scores['Feature'][:k].tolist()

# Seleccionamos los top componentes
selected_pca_components = df_pca[top_components]

# Dividimos los datos en training y tests
X_train3, X_test3, y_train3, y_test3 = train_test_split(selected_pca_components, y2, test_size=0.2, random_state=42)

# Inicialiamos la regresión lineal
regression_model = LinearRegression()

# Ajustamos el modelo
regression_model.fit(X_train3, y_train3)

# Hacemos predicciones
predictions3 = regression_model.predict(X_test3)

# Valoramos con MSE y R2
mse = mean_squared_error(y_test3, predictions3)
r2 = r2_score(y_test3, predictions3)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")


# Los resultados con mejores componentes PCA son bajos también, por lo que parece que más allá del algoritmo escogido parece un problema de elección de features, o tal vez de peso del valor 0 en Total_OoS.

# #### Elección de nuevas variables/features

# Revisamos de nuevo la importancia de los features con respecto a Total_OoS:

# In[242]:


from sklearn.feature_selection import SelectKBest, f_regression

# Separamos los datos en features (X) y la variable objetivo ('Total_OoS')
X3 = df_product_cat.drop(columns=['Total_OoS'])  # Features
y3 = df_product_cat['Total_OoS']  # Variable Objetivo

# Inicializamos SelectKBest con la scoring function (f_regression para problemas de regresion)
selector = SelectKBest(score_func=f_regression, k=10) 

# Ajustamos el selector a los datos
X_new = selector.fit_transform(X3, y3)

# Obtenemos las columnas seleccionadas
selected_columns = X3.columns[selector.get_support()]

# Creamos un df con las columnas seleccionadas
selected_features_df = pd.DataFrame(X_new, columns=selected_columns)

# Mostramos las features más importantes
print("Selected Features:")
print(selected_features_df.head())


# También revisamos las variables que más impactan en Total_OoS usando un random forest classifier:

# In[243]:


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Inicializamos RF
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Ajuste de modelo
rf.fit(X3, y3)

# Features por importancia
feature_importances = pd.Series(rf.feature_importances_, index=X3.columns)
feature_importances = feature_importances.sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 8))
feature_importances.head(20).plot(kind='barh')
plt.title('Top 20 Feature Importances')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.show()


# Vemos que Total_Delivered y Total_Sales tienen un gran impacto en Total_OoS, lo que es lógico ya que más ventas pueden llevar a roturas de stock, y más entregas hemos visto que podían suponer más ventas.
# 
# Se consideró quitar Total_Sales o Total_Delivered de los posteriores estudios de modelos, pero la precisión en los mismos caía drásticamente para cualquiera de esas dos columnas (entorno a un 10-25% de precisión MSE como máximo).
# 
# Hemos dejado otras features como algunos Product_Code importantes, los meses, Cost_price, Sell_Price, Margin y Size para poder ampliar las predicciones a más campos, y también porque perdíamos algo de precisión sin alguno de ellos.

# In[244]:


duplicate_columns = X3.columns[X3.columns.duplicated()]
print(duplicate_columns)


# In[245]:


X4 = df_product_cat[['Total_Sales', 'Total_Delivered', 'Cost_price', 'Sell_price', 'Margin',
                              'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
                              'Product_Code_Natu408', 'Product_Code_Dome206', 'Product_Code_Dome427',
                              'Product_Code_Natu079', 'Product_Code_Brit627', 'Product_Code_Natu969',
                              'Product_Code_Dome164', 'Product_Code_Natu122', 'Product_Code_Brit700',
                              'Product_Code_Dome213', 'Product_Code_Inte404', 'Product_Code_Dome363',
                              'Product_Code_Dome762', 'Product_Code_Dome459', 'Product_Code_Natu723',
                              'Product_Code_Inte327', 'Product_Code_Dome615', 'Product_Code_Dome797',
                              'Product_Code_Dome527', 'Product_Code_Brit555', 'Product_Code_Natu461',
                              'Product_Code_Dome104',
                              'SIZE_85', 'SIZE_142', 'SIZE_190', 'SIZE_317', 'SIZE_283', 'SIZE_125', 'SIZE_114', 'SIZE_481']].copy()

y4 = df_product_cat[['Total_OoS']].copy()


# In[246]:


duplicate_columns = X4.columns[X4.columns.duplicated()]
print(duplicate_columns)


# #### Decission Tree

# Iniciamos un modelo de árbol de decisión, escalaremos las variables numéricas.

# In[247]:


from sklearn.tree import DecisionTreeRegressor

X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.2, random_state=42)

# Extraemos las columnas numéricas
numerical_columns = ['Total_Sales','Cost_price', 'Sell_price','Margin']

# Escalamos las variables numéricas
scaler = StandardScaler()
X_train4[numerical_columns] = scaler.fit_transform(X_train4[numerical_columns])
X_test4[numerical_columns] = scaler.transform(X_test4[numerical_columns])

# Iniciamos el modelo
decision_tree = DecisionTreeRegressor(random_state=42)

# Ajustamos el modelo a los training sets
decision_tree.fit(X_train4, y_train4)

# Hacemos predicciones
tree_preds = decision_tree.predict(X_test4)

# Evaluamos el modelo
mse = mean_squared_error(y_test4, tree_preds)
print(f"Decision Tree MSE: {mse}")

r2 = r2_score(y_test4, tree_preds)
print(f"Decision Tree R-squared Score {r2}")


# MSE es bastante alto 5.3, R2 da información de un 43,7% de la varianza del objetivo. No es un mal resultado pero podría ser mejor, por lo que aplicaremos algunos cambios.

# #### Resampling y nuevo Decision Tree

# Al haber un gran peso de los valores 0 en Total_OoS esto puede estar impactando en la predicción, por lo que vamos a intentar realizar un resampling para corregir esta tendencia y aplicar un modelo decision tree.
# 
# Este resampling intenta compensar el peso de los valores para que el modelo no se vea condicionado por el valor 0.

# In[248]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Definimos la estrategia de resampling con oversampling para valores que no son 0 y undersample para valores 0
resampling_strategy = [('over', RandomOverSampler()), ('under', RandomUnderSampler())]

# Creamos un pipeline de resampling
resampling_pipeline = Pipeline(steps=resampling_strategy)

# Ajustamos y aplicamos la estrategia de resampling en los datos de entrenamiento
X_train_resampled4, y_train_resampled4 = resampling_pipeline.fit_resample(X_train4, y_train4)


# En este caso aplicaremos un clasificador Decisión Tree usando los sets sobre los que hemos aplicado el resampling.

# In[249]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score

# Iniciamos el Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Ajustamos
clf.fit(X_train_resampled4, y_train_resampled4)

# Hacemos las predicciones en el test set
y_pred4 = clf.predict(X_test4)

# Revisamos un classification_report
print(classification_report(y_test4, y_pred4))

# Calculams la precision
precision = precision_score(y_test4, y_pred4, average='weighted')
print(f"Precision: {precision}")


# Para los modelos de clasificación tenemos unas métricas distintas a los modelos de regresión.
# 
# Por ejemplo hemos utilizado classification_report por un lado, que incluye las siguientes métricas:
# 
# Precisión: Indica la proporción de instancias predichas correctamente entre las instancias predichas para cada clase.
# En este caso la precisión es baja, por lo que el modelo tiene a generar muchos falsos positivos.
# 
# Recall: proporción de las instancias predichas correctamente entre las instancias reales de cada clase.
# El Recall is bajo también lo que indica que el modelo tiene problemas para identificar correctamente las instancias de esas clases.
# 
# F1-score: Meda harmónica de precisión y recall, proporcionando equilibrio entre las dos métricas. 
# Es bajo para casi todas las clases, indicando un funcionamiento pobre en general para precisión y recall.
# 
# Support: Indica el número de samples para cada clase. La clase '0' tiene muchos más samples que el resto de clases.
# 
# Accuracy: La precisión del modelo en general es del 16%, que puede engañoso dado el deseqquilibrio entre clases.
# El modelo consigue precisión alta en la clase mayoritaría pero muy baja en las clases minoritarias.
# 
# En conclusión, el modelo parece predecir bastante bien la clase mayoritaria 0, pero no para el resto dado el gran desequilibrio entre clases.

# Hemos utilizado además una métrica de precisión que nos ha dado un 84% de precisión, lo que no se corresponde con lo obtenido con classification_report.
# 
# La aparente discrepancia entre la puntuación de precisión y el report de clasificación puede deberse a:
# 
# Metodología de Weighting: la media weighting en precision_score y en classification_report puede usar diferentes métodos para calcular la media weighted, posiblemente considerando la distribución de clases o de pesos de samples de manera diferente.
# 
# Normalización: el calculo puede realizarse mediante el número de samples entre otros factores, llevando a diferencias entre las medias de weight.

# Revisamos además una gráfica para ver la correspondencia entre valores reales y predichos.

# In[250]:


plt.scatter(y_test4, y_pred4)
plt.title('Linear Regression: Predicted vs. Actual Total_OoS')
plt.xlabel('Actual Total_OoS')
plt.ylabel('Predicted Total_OoS')
plt.show()


# Vemos que se están prediciendo valores más altos, lo que quiere decir que el resampling en ese sentido está funcionando y teniendo en cuenta valores más altos. Sin embargo los valores predichos más altos no se corresponden con los reales.
# 
# Aunque se acumulan mucho más los valores entorno a la diagonal donde deberían coincidir Actual y Predicted, vemos que la precisión no es muy alta sobretodo a partir de valores predicted más altos de 35, por lo que parece que al modelo le cuesta más predecir valores en general altos.

# Utilizamos diferentes modelos para intentar obtener mejores resultados.

# #### GaussianNB

# In[251]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Iniciamos el modelo
naive_bayes = GaussianNB()

# Ajustamos el modelo
naive_bayes.fit(X_train_resampled4, y_train_resampled4)

# Hacemos predicciones
nb_preds = naive_bayes.predict(X_test4)

# Evaluamos el modelo
accuracy = accuracy_score(y_test4, nb_preds)
print(f"Gaussian Naive Bayes Accuracy: {accuracy}")
print(classification_report(y_test4, nb_preds))


# Con GaussianNB la precisión es también baja.

# #### SVM

# Se entrena también un modelo más complejo como SVM, donde se ha tenido que entrenar el modelo en base a una pequeña porción de los datos para poder implementarlo sin incidencias en su ejecución (solo un 0,03 del total de datos).

# In[252]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Seleccionamos las columnas
specified_columns3 = df_product_cat[['Total_OoS','Total_Sales', 'Total_Delivered', 'Cost_price', 'Sell_price', 'Margin',
                              'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
                              'Product_Code_Natu408', 'Product_Code_Dome206', 'Product_Code_Dome427',
                              'Product_Code_Natu079', 'Product_Code_Brit627', 'Product_Code_Natu969',
                              'Product_Code_Dome164', 'Product_Code_Natu122', 'Product_Code_Brit700',
                              'Product_Code_Dome213', 'Product_Code_Inte404', 'Product_Code_Dome363',
                              'Product_Code_Dome762', 'Product_Code_Dome459', 'Product_Code_Natu723',
                              'Product_Code_Inte327', 'Product_Code_Dome615', 'Product_Code_Dome797',
                              'Product_Code_Dome527', 'Product_Code_Brit555', 'Product_Code_Natu461',
                              'Product_Code_Dome104',
                              'SIZE_85', 'SIZE_142', 'SIZE_190', 'SIZE_317', 'SIZE_283', 'SIZE_125', 'SIZE_114', 'SIZE_481']].copy()


# In[253]:


specified_columns3.head()


# In[254]:


# Se realiza sample de los datos
sampled_data = specified_columns3.sample(frac=0.03, random_state=42)


# In[255]:


sampled_data.head()


# In[256]:


X5 = sampled_data.drop(columns=['Total_OoS']) 
y5 = sampled_data['Total_OoS']

# Dividimos en train y set
X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, test_size=0.2, random_state=42)

X5.head()


# In[257]:


# Escalamos las columnas numéricas
numerical_columns = ['Total_Sales', 'Total_Delivered', 'Cost_price', 'Sell_price', 'Margin']
scaler = StandardScaler()
X_train5[numerical_columns] = scaler.fit_transform(X_train5[numerical_columns])
X_test5[numerical_columns] = scaler.transform(X_test5[numerical_columns])

# Hacemos también oversampling para valores que no son 0 y undersample de los 0
resampling_strategy = [('over', RandomOverSampler()), ('under', RandomUnderSampler())]

# Creamos pipeline
resampling_pipeline = Pipeline(steps=resampling_strategy)

# Ajustamos el resample
X_train_resampled5, y_train_resampled5 = resampling_pipeline.fit_resample(X_train5, y_train5)

# Iniciamos el SVM
svm_classifier = SVC(random_state=42)

# Ajustamos el modelo
svm_classifier.fit(X_train_resampled5, y_train_resampled5)

# Hacemos predicciones
y_pred5 = svm_classifier.predict(X_test5)

# Evaluamos el clasificador
print(classification_report(y_test5, y_pred5))

# Calculamos precisión
precision = precision_score(y_test5, y_pred5, average='weighted')
print(f"Precision: {precision}")


# In[258]:


#Revisamos con un plot:
plt.scatter(y_test5, y_pred5)
plt.title('Linear Regression: Predicted vs. Actual Total_OoS')
plt.xlabel('Actual Total_OoS')
plt.ylabel('Predicted Total_OoS')
plt.show()


# La precisión de SVM para F1 score como vemos es umás alta (0.40), pero requiere bajar mucho la muestra de datos (el límite estaría en frac= 0.03), lo que hace posible que aun haya menos valores Total_OoS diferentes a 0. De hecho no se han predicho valores por encima de 48.
# 
# Se realizan muchas menos predicciones aunque también hay menos erróneas.

# Por tanto tenemos la opción de usar un SVM con el uso de una pequeña muestra y más precisión o un Decision Tree con todos los datos pero poca precisión. Pero la conclusión general es que la cantidad de valores 0 en Total_OoS ha desequilibrado las predicciones de los modelos, incluso a pesar de utilizar resampling.

# ## PREDICCIONES

# Creamos un dataframe para usar en predicciones. Se crea una columna nueva Month_11 y se hace una regresión lineal para añadir datos a esa nueva columna. Lo hacemos a partir del modelo DecisionTreeClassifier con oversampling

# In[259]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Nuevo dataframe
X4 = df_product_cat[['Total_Sales', 'Total_Delivered', 'Cost_price', 'Sell_price', 'Margin',
                              'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
                              'Product_Code_Natu408', 'Product_Code_Dome206', 'Product_Code_Dome427',
                              'Product_Code_Natu079', 'Product_Code_Brit627', 'Product_Code_Natu969',
                              'Product_Code_Dome164', 'Product_Code_Natu122', 'Product_Code_Brit700',
                              'Product_Code_Dome213', 'Product_Code_Inte404', 'Product_Code_Dome363',
                              'Product_Code_Dome762', 'Product_Code_Dome459', 'Product_Code_Natu723',
                              'Product_Code_Inte327', 'Product_Code_Dome615', 'Product_Code_Dome797',
                              'Product_Code_Dome527', 'Product_Code_Brit555', 'Product_Code_Natu461',
                              'Product_Code_Dome104',
                              'SIZE_85', 'SIZE_142', 'SIZE_190', 'SIZE_317', 'SIZE_283', 'SIZE_125', 'SIZE_114', 'SIZE_481']].copy()

# Variable Objetivo
y4 = df_product_cat[['Total_OoS']].copy()


# In[260]:


from sklearn.linear_model import LogisticRegression

# Lista de columnas para prediccion (incluyendo Month_3 a Month_10)

df_product_OoS_selected = X4.copy()

columns_for_prediction = [col for col in df_product_OoS_selected.columns if col != 'Month_11']


# In[261]:


# Separamos features para training
X_train6 = df_product_OoS_selected[columns_for_prediction]

# Creamos dataframe vacío para la columna predicho 'Month_11'
df_with_predicted_month_11 = df_product_OoS_selected.copy()

# Creamos y ajustamos un modelo de regresión logística
model = LogisticRegression()
model.fit(X_train6, df_product_OoS_selected['Month_10'])

# Predecimos los valores de mes 'Month_11'
predicted_month_11 = model.predict(X_train6)

# Asignamos los modelos predichos a la columna 'Month_11'
df_with_predicted_month_11['Month_11'] = predicted_month_11

# Mostramos el df
print(df_with_predicted_month_11)


# In[262]:


df_with_predicted_month_11.columns


# In[263]:


#Filtramos los valores para Month_11

df_with_predicted_month_11 = df_with_predicted_month_11[df_with_predicted_month_11['Month_11'] == 1]


# In[264]:


df_with_predicted_month_11.head(20)


# In[265]:


#Quitamos columnas Month excepto Month_11

df_with_predicted_month_11.drop(['Month_3','Month_4','Month_5','Month_6','Month_7','Month_8','Month_9', 'Month_10'], axis=1, inplace=True)


# In[266]:


df_with_predicted_month_11.head()


# In[147]:


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
table_name = 'Prediccion_Rotura'

# Subimos el DataFrame a SQL Server
df_with_predicted_month_11.to_sql(table_name, engine, index=False, if_exists='replace')

print("DataFrame uploaded to SQL Server successfully!")


# In[ ]:




