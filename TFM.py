#!/usr/bin/env python
# coding: utf-8

# # TFM 1

# En esta primera parte realizaremos la importación desde SQL de los datos, realizaremos un análisis EDA, revisión y tratamiento de outliers entre otros.

# ### IMPORTACIÓN DE DATOS

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


#Si fuera necesario password:
#username = 'DESKTOP-VUJ9ETK\erocho'
#password = 'your_password'

#conn = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)


# ### df_all_tables

# df_all_tables incluye la información por meses de uds vendidas, enviadas, veces que se da rotura de stock por producto y tienda durante varios meses de 2015. Es una tabla con resultados por mes.

# ### df_Delivery_Route

# df_Delivery_Route permite comparar los mismos datos por día para las entregas realizadas con Delivery_Day y er si esas entregas se han realizado en la misma fecha que el día previsto de ruta, a que si se han realizado entregas fuera del día de ruta sabremos que se ha incurrido en costes de transporte extra.Relaciona cada línea también con el tipo de día (festivo o laborable).

# ### df_Sales_Rotura

# df_Sales_Rotura compara las ventas con los días de rotura y los relaciona también con tipo de día (festivo o laborable). Es un dataframe por día.

# ## Análisis EDA (Exploratory Data Analysis)

# ### Revisión de tipo de datos y nulls

# In[3]:


#Empezaremos revisando información de las tablas:
print("df_all_tables:")
print(df_all_tables.info())


# Aquí tenemos Nulls en poblacion y provincia, ya que hay codigos postales de Affiliated_Outlets que no encontraron correspondencia en la tabla de códigos postales.

# In[4]:


#Usaremos un fillna() en este caso con un valor por defecto:

df_all_tables['provincia'].fillna('Desconocido', inplace=True)
df_all_tables['poblacion'].fillna('Desconocido', inplace=True)

#Y revisaremos los Nulls de nuevo:

df_all_tables.isnull().sum()


# In[5]:


#Haremos lo mismo para df_Delivery_Route.

print("df_Delivery_Route:")
print(df_Delivery_Route.info())


# En este caso sí que hay Nulls en Route_Day, poblacion y provincia.
# 
# Además tenemos como problema el hecho de que Delivery_DAY y Route_DAY no aparecen con formato fecha:

# In[6]:


#Podemos ver mejor el detalle de los nulls con isnull()

df_Delivery_Route.isnull().sum()


# Los nulls de Route_DAY aparecen ya que hay días que no tenían prevista ruta pero sí se entregaron con DeliveryDay.
# 
# Hemos usado la columna Day_Mismatch para saber qué Route_DAYs son Nulls. Si aparece como "Match" significa que sí que hay un Route Day que corresponde a Delivery Day y si tenemos "Mismatch" significa que sí hay Delivery pero no Route Day.
# 
# En el caso de poblacion y provincia también tendremos el mismo problema de Nulls.

# No podemos asignar un valor a los Nulls de Route_DAY con fillna porque tendríamos 
# problemas si añadimos un valor que no tiene formato fecha.
# Tampoco podemos añadir una media o hacer drop de las filas, ya que contienen más información que nos podria ser útil.

# In[7]:


#En caso de que quisieramos eliminar Nulls de Route_DAY podríamos usar por ejemplo:
#df_Delivery_Route['Route_DAY'].fillna('Sin_ruta', inplace=True)


# In[8]:


#También resolvemos los Nulls para población y provincia.

df_Delivery_Route['provincia'].fillna('Desconocido', inplace=True)
df_Delivery_Route['poblacion'].fillna('Desconocido', inplace=True)


# In[9]:


#Ya no tendremos Nulls, excepto para Route_Day_
df_Delivery_Route.isnull().sum()


# In[10]:


#Ahora cambiamos las columnas de fechas que hemos visto como incorrectas:
df_Delivery_Route['Delivery_DAY'] = pd.to_datetime(df_Delivery_Route['Delivery_DAY'])
df_Delivery_Route['Route_DAY'] = pd.to_datetime(df_Delivery_Route['Route_DAY'])


# In[11]:


print(df_Delivery_Route.dtypes)


# In[12]:


#Revisamos ahora df_Sales_Rotura
print("df_Sales_Rotura:")
print(df_Sales_Rotura.info())


# In[13]:


#Tendremos que cambiar el data type de Sales_DAY y OoS_DAY
df_Sales_Rotura['Sales_DAY'] = pd.to_datetime(df_Sales_Rotura['Sales_DAY'])
df_Sales_Rotura['OoS_DAY'] = pd.to_datetime(df_Sales_Rotura['OoS_DAY'])


# In[14]:


print(df_Sales_Rotura.info())


# In[15]:


#También tenemos Null en población, provincia y OoS_DAY
df_Sales_Rotura.isnull().sum()


# In[16]:


df_Sales_Rotura['provincia'].fillna('Desconocido', inplace=True)
df_Sales_Rotura['poblacion'].fillna('Desconocido', inplace=True)


# In[17]:


#Dejaremos los Nulls de OoS_DAY ya que usaremos principalmente la columna "Rotura" para obtener la misma información
df_Sales_Rotura.isnull().sum()


# ### Valores únicos

# In[18]:


#Para el EDA también podemos ver cuantos valores únicos tenemos en las columnas categóricas:
for col in df_all_tables.select_dtypes(include='object').columns:
    print(col, df_all_tables[col].nunique())   


# En este caso,  Affiliated_Code y Affiliated_Name tienen muchos valores únicos ya que se trata de los diferentes establecimientos. Poblacion también tiene un valor alto.
# 
# Location y tam_m2 hacen referencia a características de los establecimientos, y solo tienen 8 valores únicos.
# 
# Product_Code y Format hacen referencia a características de los productos.

# In[19]:


#Revisamos también df_Delivery_Route:
for col in df_Delivery_Route.select_dtypes(include='object').columns:
    print(col, df_Delivery_Route[col].nunique())


# Tenemos muchos campos categóricos coincidentes con df_all_tables,aunque en este caso también se han añadido Delivery_out_of_Route que indica si las fechas de delivery_day y route_day son coincidentes o no y Festivo, que indica si el delivery se ha hecho en festivo/domingo o no.

# In[20]:


#Hacemos los mismo para df_Sales_Rotura:
for col in df_Sales_Rotura.select_dtypes(include='object').columns:
    print(col, df_Sales_Rotura[col].nunique())


# También campos coincidentes, pero en este caso tenemos "Rotura" que indica si se ha dado rotura de stock el mismo día que la venta o no.
# 
# Festivo en este caso indica si Sales_Day coincide con un festivo/domingo o no.

# ### SUMARIO ESTADÍSTICO

# In[21]:


#Otra parte del análisis EDA es realizar un sumario estadístico mediante describe()

# Sumario estadístico para las tablas.
print("Sumario estadístico de df_all_tables:")
print(df_all_tables[['Engage','Management_Cluster','Cost_price', 'Sell_price', 'Margin', 'Total_Delivered', 'Total_Sales','Total_OoS']].describe())


# Respecto al campo Engage los valores van de 1 a 3, con una media de 2.20 y una desviación standard de 0.51 Este valor es mejor cuanto mayor es.
# 
# Management_Cluster va de 1 a 4, con media de 2.02 y desviación standard de 1.68
# Este valor también es mejor cuanto mayor es.
# 
# Cost_Price abarca desde un valor mínimo de 5.9 a un valor máximo de 66.6, con media de 32.95 y desviación de 17.99
# 
# Sell_Price va de 7 a 80.30 con media de 39.89 y desviación de 20.63
# 
# Margin, diferencia entre Sell_Price y Cost_Price, tiene un mínimo de 1.10 y máximo de 17.22 con media de 6.67 y desviación de 3.33
# 
# Total_Delivered va de un mínimo de -195, ya que se reflejan las entregas erróneas, a un máximo de 1132 con media de 1132 y desviación de 20.75.
# 
# Total_Sales también refleja ventas negativas con un -8 de min y 1134 de máximo. La media es 14.82 y desviación 20.88.
# 
# Total_OoS muestra la cantidad de veces que un producto ha estado en una tienda en rotura de stock. 
# El mínimo es 0 y el máximo 68. La media es 0.75 y la desviación 3.22. Como vemos los valores medios se acercan mucho a 0 debido a la alta presencia de este valor con respecto a los demás.

# In[22]:


#En el caso de df_Delivery_Route solo podemos analizar dos campos.

print("Sumario estadístico de df_Delivery_Route:")
print(df_Delivery_Route[['Engage','Management_Cluster']].describe())


# Engage que va también de 1 a 3, pero la media y desviación son ligeramente distintas, 2.17 y 0.53 respectivamente.
# 
# Management Cluster también va de 1 a 4, su media es 1.85 y desviación standard 1.58
# 
# Recordemos que en el anterior df teníamos las líneas organizadas por mes, sin embargo en este df tenemos líneas por días.

# In[23]:


#Para df_Sales_Rotura solo vale la pena revisar la columna Sales_Uds.

pd.options.display.float_format = '{:.2f}'.format
print("Sumario estadístico de df_Sales_Rotura:")
print(df_Sales_Rotura['Sales_Uds'].describe())

#Hemos cambiado el display format para evitar valores confusos.


# El mínimo es -57 ya que se reflejan las ventas erróneas y el máximo 351
# Tenemos una media de 1.96 y desviación de 1.69

# ### VISUALIZACIONES

# #### df_all_tables

# In[24]:


#También para realizar el EDA haremos una visualización para algunas de las variables categóricas:

#Empezaremos con la tabla df_all_tables

#Hacemos visualización de Count por Month:
plt.figure(figsize=(10, 6))
sns.countplot(x='Month', data=df_all_tables, order=df_all_tables['Month'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Count of Month')
plt.show()


# Vemos que los registros son muy similares durante todos los meses excepto para Octubre. Es posible que se deba a que no hay registros para el mes completo. Recordemos que son registros de solo una parte de 2015. 

# In[25]:


#Seguimos con la tabla df_all_tables

#Hacemos visualización de Count por provincia:
plt.figure(figsize=(10, 6))
sns.countplot(x='provincia', data=df_all_tables, order=df_all_tables['provincia'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Count of Provincia')
plt.show()


# Los registros se concentran en Madrid, Barcelona y Valencia, provincias con más población.

# In[26]:


#También en la tabla df_all_tables tenemos Engage, que aunque es numérico se puede valorar como categórico
#, al tener solo 3 valores.

plt.figure(figsize=(10, 6))
sns.countplot(x='Engage', data=df_all_tables, order=df_all_tables['Engage'].value_counts().index)
plt.xticks(rotation=0)
plt.title('Count of Engage')
plt.show()


# Vemos una mayor concentración de resgistros en Engage 2, el resto están muy por debajo. 
# Recordemos que el mejor valor es 3, que sería el valor en el centro, pero a mucha distancia al 2.

# In[27]:


#Igualmente tenemos Management_Cluster, que aunque es numérico se puede valorar como categórico al tener solo 4 valores.

plt.figure(figsize=(10, 6))
sns.countplot(x='Management_Cluster', data=df_all_tables, order=df_all_tables['Management_Cluster'].value_counts().index)
plt.xticks(rotation=0)
plt.title('Count of Management_Cluster')
plt.show()


# También es mejor un mayor número de cluster. En este caso el cluster 4 es el más predominante aunque seguido también por el más bajo.

# In[28]:


#Para la columna Location tenemos:
plt.figure(figsize=(10, 6))
sns.countplot(x='Location', data=df_all_tables, order=df_all_tables['Location'].value_counts().index)
plt.xticks(rotation=0)
plt.title('Count of Location')
plt.show()


# Buena parte de registros están en ANY o Village. 
# Vemos que aunque Madrid, Barcelona y Valencia aparecían como provincias principales, aquí no tenemos City como categoría principal, por lo que seguramente los registros se den en ciudades más pequeñas dentro de esas provincias.

# In[29]:


#Realizamos la visualización para Tam_m2 de df_all_tables.

plt.figure(figsize=(10, 6))
sns.countplot(x='Tam_m2', data=df_all_tables, order=df_all_tables['Tam_m2'].value_counts().index)
plt.xticks(rotation=0)
plt.title('Count of Tam_m2')
plt.show()


# Vemos que la mayoría de valores están entre los 2-5m, 5-10 y 10-20.
# Esto nos indica que el tamaño de estos establecimientos suele ser medio.

# In[30]:


#Realizamos la visualización para los códigos de producto de df_all_tables.
plt.figure(figsize=(10, 6))
sns.countplot(x='Product_Code', data=df_all_tables, order=df_all_tables['Product_Code'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Count of Product_Code')
plt.show()


# Hay algo muy interesante y es que para la mitad de productos casi no hay registros, no hay ventas o deliveries por ejemplo.
# Esto hace plantear la necesidad de tener esos productos disponibles tanto como para analizar sus datos como para realmente tenerlos en venta.

# In[31]:


#Realizamos la visualización para SIZE de df_all_tables.

plt.figure(figsize=(10, 6))
sns.countplot(x='SIZE', data=df_all_tables, order=df_all_tables['SIZE'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Count of SIZE')
plt.show()


# Hay una Size (85) que acapara gran parte de los registros y que casi la mitad de Size no tienen registros, por lo que también seria posible dejar de trabajar con ellos.

# In[32]:


#Realizamos la visualización para Format de df_all_tables.

plt.figure(figsize=(10, 6))
sns.countplot(x='Format', data=df_all_tables, order=df_all_tables['Format'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Count of Format')
plt.show()


# El Format predominante es ASL, muy por debajo están ETO y ATA.

# In[33]:


#Además, aunque no se trata de un campo categórico, realizamos la revisión de los registros que aparecen con rotura de stock.

plt.figure(figsize=(10, 6))
sns.countplot(x='Total_OoS', data=df_all_tables, order=df_all_tables['Total_OoS'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Count of Total_OoS')
plt.show()

Vemos que hay muchísimos más registros sin rotura que con ella, por lo que no parece que la rotura de stock sea un gran problema a priori. Sin embargo es interesante estudiarla, ya que la rotura de stock supone menos ventas y por tanto menos ingresos.

El hecho de que la mayoría de valores sean 0 pueden también condicionar el entrenamiento de modelos de predicción, ya que el peso de ese valor será mayor.
# #### df_Delivery_Route

# In[34]:


#Realizamos un ejercicio similar para los campos categóricos de df_Delivery_Route

#Hacemos visualización de Count por provincia:
plt.figure(figsize=(10, 6))
sns.countplot(x='provincia', data=df_Delivery_Route, order=df_Delivery_Route['provincia'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Count Entregas por Provincia')
plt.show()


# Podemos apreciar que la mayoría del peso de Delivery por provincia se queda en Madrid, seguido por Barcelona y Valencia, que son ciudades más grandes. 
# Hay diferencia de los valores más grandes de esta tabla y la que indica solo meses (df_all_tables), entre Madrid y el resto de provincias.

# In[35]:


#Hacmeos el mismo ejercicio para "Location"
plt.figure(figsize=(10, 6))
sns.countplot(x='Location', data=df_Delivery_Route, order=df_Delivery_Route['Location'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Count of Location')
plt.show()


# En este caso gran parte se queda en "Any" y "Village". La distribución de registros es similar a los de df_all_tables.

# In[36]:


#Tam_m2 de df_Delivery_Route

plt.figure(figsize=(8, 6))
sns.countplot(x='Tam_m2', data=df_Delivery_Route, order=df_Delivery_Route['Tam_m2'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Count of Tam_m2')
plt.show()


# La distribución de valores es muy similar a la tabla df_all_tables

# In[37]:


#Revisamos Count Delivery_out_of_Route:

plt.figure(figsize=(8, 6))
sns.countplot(x='Delivery_out_of_Route', data=df_Delivery_Route, order=df_Delivery_Route['Delivery_out_of_Route'].value_counts().index)
plt.xticks(rotation=0)
plt.title('Count of Delivery_out_of_Route')
plt.show()


# Tenemos 70000 counts para No, que significa que 70000 entregas se han hecho en fecha de ruta. Sin embargo también tenemos casi 5000 counts para Yes, donde se han realizado entregas fuera de ruta y por tanto con coste extra.

# In[38]:


#Para df_Delivery_Route revisamos también el 
#campo Festivo que nos va a indicar si se han hecho entregas en días festivos/domingos o no.

plt.figure(figsize=(8, 6))
sns.countplot(x='Festivo', data=df_Delivery_Route, order=df_Delivery_Route['Festivo'].value_counts().index)
plt.xticks(rotation=0)
plt.title('Count Entrega en Festivo')
plt.show()


# Vemos que la gran mayoría de entregas han sido el días laborables.

# #### df_Sales_Rotura

# In[39]:


#Ahora revisaremos el dataframe df_Sales_Rotura. 

plt.figure(figsize=(8, 6))
sns.countplot(x='provincia', data=df_Sales_Rotura, order=df_Sales_Rotura['provincia'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Count ventas provincia')
plt.show()


# Vemos que la distribución de valores es similar a los otros dataframes

# In[40]:


#Revisaremos el count de Sales por Product_Code.
plt.figure(figsize=(8, 6))
sns.countplot(x='Product_Code', data=df_Sales_Rotura, order=df_Sales_Rotura['Product_Code'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Count Ventas Product_Code')
plt.show()


# Aquí tenemos valores por días.En este caso el orden de ventas es similar, pero la diferencia entre productos es más acentuado que en la tabla df_all_tables donde se estudiaban los valores por meses.

# In[41]:


#Revisaremos el campo Rotura que nos indica si se ha producido problemas de stock a la vez que ventas, 
#lo que puede indicar que se hayan limitado esas ventas.

plt.figure(figsize=(8, 6))
sns.countplot(x='Rotura', data=df_Sales_Rotura, order=df_Sales_Rotura['Rotura'].value_counts().index)
plt.xticks(rotation=0)
plt.title('Count Ventas con Rotura')
plt.show()


# Vemos que los problemas de rotura parecen ser bajos.

# In[42]:


#Revisamos el campo "Festivo" para comprobar si se han dado ventas en días festivos o domingos, 
#ya que estos establecimientos abren en teoría solo 6 días a la semana, 
#lo que supondría que al menos los domingos no debería haber ventas.

plt.figure(figsize=(8, 6))
sns.countplot(x='Festivo', data=df_Sales_Rotura, order=df_Sales_Rotura['Festivo'].value_counts().index)
plt.xticks(rotation=0)
plt.title('Count Ventas en Festivo')
plt.show()


# Aunque hay pocos valores, sí que se dan casos de ventas en festivos/domingos. Esto se puede deber por un lado a que los campos de festivos no sean correctos, pero al revisar con sql se han visto ventas en días que caían en domingo.
# 
# Por ejemplo para asegurarnos podemos buscar algun domingo o festivo y revisar si hay ventas.

# In[43]:


#Vemos que varios de los resultados que se obtienen son para el 2015-08-30, que fue domingo.

df_Sales_Rotura[(df_Sales_Rotura['Festivo'] == 'Domingo o Festivo') & (df_Sales_Rotura['Sales_DAY'] == '2015-08-30')]


# # ANÁLISIS UNIVARIANTE:

# In[44]:


#Análizamos los valores numéricos para df_all_tables.
df_all_tables[['Cost_price','Sell_price','Margin']].hist(figsize=(20, 20))


# In[45]:


#Hacemos visualización a parte de Total_Delivered y Total_Sales para ver mejor su detalle:

plt.figure(figsize=(5, 5))
plt.hist(df_all_tables['Total_Delivered'], bins=20)
plt.xlim(-200, 300)
plt.xlabel('Delivered')
plt.ylabel('Freq')
plt.title('Histogram Total_Delivered')
plt.show()

plt.figure(figsize=(5, 5))
plt.hist(df_all_tables['Total_Sales'], bins=20)
plt.xlim(-200, 300)
plt.xlabel('Sales')
plt.ylabel('Freq')
plt.title('Histogram Total_Sales')
plt.show()


# En Cost_Price tenemos valores que van desde 5 a menos de 70, con picos alrededor tanto de 15 como 45. Podría deberse por ejemplo a una agrupación de tipos de producto en torno a dos grupos de precio.
# 
# En Sell_Price son valores similares pero un poco más altos, hasta 80 con picos en 20 y 50 aprox. Pasa algo similar a Cost_Price, ya que los Sell_Price se han creado a partir de Cost_Price.
# 
# El márgen es más alto en 8 y en 13, y llega a un máximo de 17. No tenemos valores negativos, por lo que no se han vendido productos por un precio menor que su coste.
# 
# En cuanto a uds, tenemos para Total_Delivered valores entre -50 y casi 300 con una media que se situa entre 0 y 50. 
# Recordemos que los valores negativos se pueden dar debido a entregas erróneas, hurtos, deterioros, etc.
# El valor en -50 tiene una frecuencia bastante alta, lo que puede indicar que hay bastantes regularizaciones de entrega, por lo que es algo que se podría mejorar revisando si hay algún problema de entregas.
# 
# Para Total_Sales hay aun valores negativos aunque menos y el máximo supera 200.

# In[46]:


# df_Delivery_Route no tiene valores numéricos a valorar, son categóricos.

#Podemos revisar las uds diarias vendidas de la tabla df_Sales_Rotura:

plt.figure(figsize=(5, 5))
plt.hist(df_Sales_Rotura['Sales_Uds'], bins=20)
plt.xlim(-50, 50)
plt.xlabel('Sales')
plt.ylabel('Freq')
plt.title('Histogram Sales_Uds')
plt.show()


# En este df por día tenemos ventas diarias que van desde cerca de -20 a más de 20 uds

# # ANÁLISIS DE CORRELACIÓN:

# Este análisis es también parte de EDA y nos permite ver cómo se relacionan los diferentes campos entre ellos, cómo son de dependientes.
# Los valores van de -1 a 1. 1 indica una fuerte relación positiva entre variables,0 indica que no hay relación y -1 que hay relación negativa entre las dos variables

# In[47]:


#Para df_all_tables seleccionaremos las variables numéricas más importantes:

correlation_matrix_1 = df_all_tables[['Engage','Management_Cluster'
                                      ,'SIZE', 'Cost_price','Sell_price','Margin',
                                      'Total_Delivered','Total_Sales', 'Total_OoS' ]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_1, annot=True, cmap='coolwarm')
plt.show()


# En este caso lo más obvio son las relaciones entre Cost_price, Sell_price,Margin y entre Total_Delivered y Total_Sales.
# Esto implica una relación cercana a 1 entre estas variables. Esto nos indica que a un mayor Sell_price también se da un mayor Cost_price, a mayor Margin mayor Sell_Price, etc. Tal vez no sea necesario incluir en un mismo estudio por ejemplo Sell_Price y Cost_Price.
# 
# Además hay relaciones negativas entre Size y Cost_price, Sell_price,Margin,Total_Delivered y Total_Sales,lo que implica que esas variables bajan con el aumento de Size.
# 
# Total_OoS por otro lado no tiene una relación fuerte con ninguna otra variable.

# In[48]:


#En el caso de df_Delivery_Route solo hay 2 columnas que se pueden valorar, Engage y Management Cluster
correlation_matrix_2 = df_Delivery_Route[['Engage','Management_Cluster']].corr()
sns.heatmap(correlation_matrix_2, annot=True, cmap='coolwarm')
plt.show()


# Vemos que la relación es cercana a 0 lo que indica que a nivel de este df, que valora las entregas, no hay casi relación entre Engage y Management_Cluster

# Para df_Sales_Rotura solo hay 1 variable numérica (Sales_Uds), por lo que no vamos a poder hacer una correlation matrix.

# # DETECCIÓN DE OUTLIERS

# También como parte del EDA, revisaremos con un boxplot si tenemos outliers. Empezaremos con df_all_tables.

# #### Outliers df_all_tables/Cost_Price

# In[49]:


#Empezaremos con los valores por meses que nos da df_all_tables.

sns.boxplot(x=df_all_tables['Cost_price'])
plt.show()


# En este caso no tenemos Outliers, todos los valores están dentro del boxplot o sus "whiskers"

# #### Outliers df_all_tables/Sell_Price

# In[50]:


#Sell_Price
sns.boxplot(x=df_all_tables['Sell_price'])
plt.show()


# Tenemos la misma situación, no hay Outliers para Sell_Price.

# #### Outliers df_all_tables/Margin

# In[51]:


#Revisamos Margin: 
sns.boxplot(x=df_all_tables['Margin'])
plt.show()


# Sí que hay outliers con un margen de cerca de 18, cuando el máximo está en torno a 14.Esto indica que hay algún producto que tiene un precio de venta tal vez muy elevado con respecto al coste. Sin embargo no lo vamos a tratar como un error, lo vamos a dejar ya que es una variación del valor muy pequeña y la consideraremos como correcta.

# #### Outliers df_all_tables/Total_Delivered

# In[52]:


# Revisamos Total_Delivered:

sns.boxplot(x=df_all_tables['Total_Delivered'])
plt.show()


# En este caso tenemos muchísimos outliers, lo que dificulta incluso la visualización del boxpolot.

# In[54]:


#Vamos a usar el método IQR para revisar los outliers en detalle.
#Primero lo haremos para Total_Delivered

Q1 = df_all_tables['Total_Delivered'].quantile(0.25)http://localhost:8888/notebooks/TFM.ipynb#Outliers-Total_Delivered
Q3 = df_all_tables['Total_Delivered'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 2 * IQR
upper_bound = Q3 + 2 * IQR

outliers = df_all_tables[(df_all_tables['Total_Delivered'] < lower_bound) | (df_all_tables['Total_Delivered'] > upper_bound)]
print(outliers)


# Inicialmente con una sensibilidad del umbral de 1.5 para los límites superior e inferior inicialmente obteníamos 24939 outliers para Total_Delivered dado que tenemos una mediana de 13.63 y desviación standard 20.56, lo que supone una amplitud de datos grande, por lo que muchos puntos caían fuera del rango típico 1.5 definido en el metodo IQR.
# 
# Por eso, en este caso hemos decidido utilizar un rango de 2 para Total_Delivered, y hemos obtenido 17940  outliers, intentando equilibrar la cantidad de outliers que dejaremos fuera y la posible precisión en el entrenamiento de modelos.

# In[55]:


#Realizaremos el mismo ejercicio con el método Z-Score para Total_Delivered
from scipy import stats

threshold = 2
outliers2 = df_all_tables[np.abs(stats.zscore(df_all_tables['Total_Delivered'])) > threshold]

print(outliers2)


# En este caso obtenemos 16030  outliers on un umbral de 2. Nos quedaremos con este método para eliminar los outliers de Total_delivered al tener menos outliers con un umbral similar.

# #### Outliers df_all_tables/Total_Sales

# In[53]:


#Total_Sales:
sns.boxplot(x=df_all_tables['Total_Sales'])
plt.show()


# Tenemos el mismo problema, muchos outliers.

# In[56]:


#Usamos el método IQR para Total_Sales:

Q1 = df_all_tables['Total_Sales'].quantile(0.25)
Q3 = df_all_tables['Total_Sales'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 2 * IQR
upper_bound = Q3 + 2 * IQR

outliers3 = df_all_tables[(df_all_tables['Total_Sales'] < lower_bound) | (df_all_tables['Total_Sales'] > upper_bound)]
print(outliers3)


# También obtenemos muchos outliers para un umbral de 2: 20304

# In[57]:


#Realizaremos el mismo ejercicio con el método Z-Score para Total_Sales.

threshold = 2
outliers4 = df_all_tables[np.abs(stats.zscore(df_all_tables['Total_Sales'])) > threshold]

print(outliers4)


# En este caso los outliers se limitan a 16543 registros con un umbral de 2. Este será el método con el que eliminaremos outliers de Total_Sales.

# #### Outliers df_all_tables/Total_OoS

# In[58]:


#En el caso de Total_OoS tenemos:

sns.boxplot(x=df_all_tables['Total_OoS'])
plt.show()


# Como vemos la mayoría de registros está en 0, por lo que se considera que todo lo que está por encima de 0 es outlier.
# 
# En este caso no trataremos los valores por encima de 0 como outliers, ya que de lo contrario nos quedaremos sin datos para estudiar la rotura.

# #### Tratamiento de Outliers para df_all_tables

# En el caso de df_all_tables eliminaremos los outliers obtenidos con zscore para Total_Sales y Total_Delivered.
# 
# Se están perdiendo muchas filas pero al tratarse de un dataset muy grande no creemos que impacte tanto en los posteriores algoritmos que aplicaremos, que además como veremos en algunos casos no pueden gestionar correctamente todos los datos y necesitan realizar sampling para algunos algoritmos.

# In[59]:


#Como queremos tratar los outliers de 2 columnas de df_all_tables, realizaremos una unión de ellas y luego drop:

outliers_indices = outliers2.index.union(outliers4.index)

df_all_tables_no_outliers = df_all_tables.drop(outliers_indices)


# Podríamos haber usado una intersección para realizar el drop, pero en ese caso nos hubieramos quedado con muchos de los outliers sin drop.

# In[60]:


#Además haremos un drop de los valores negativos de estas 2 columnas, 
#ya que es posible que afecten a los resultados de análisis posteriores.

df_all_tables_no_outliers  = df_all_tables_no_outliers[(df_all_tables_no_outliers ['Total_Delivered'] >= 0)
                                                       & (df_all_tables_no_outliers ['Total_Sales'] >= 0)]


# El eliminar los valores negativos ha permitido también mejorar el desempeño de los algoritmos y algun feature que no funcionaba bien con ese tipo de valores.

# In[61]:


#Revisamos de nuevo los outliers
sns.boxplot(x=df_all_tables_no_outliers['Total_Delivered'])
plt.show()


# In[62]:


#Lo mismo para Total_Sales
sns.boxplot(x=df_all_tables_no_outliers['Total_Sales'])
plt.show()


# Después de comparar varios thresholds para delimiar outliers, hemos decidido usar un 2, ya que mayores thresholds hacían que apareciesen más outliers en la segunda revisión de los mismos al cambiar la distribución de los valores.

# Para df_Delivery_Route no tenemos columnas numéricas para estudiar outliers, ya que son campos categóricos. 

# #### Outliers df_Sales_Rotura/Sales_Uds

# In[64]:


#Revisamos el plot:

sns.boxplot(x=df_Sales_Rotura['Sales_Uds'])
plt.show()


# In[63]:


#Para df_Sales_Rotura podemos de nuevo estudiar Sales_Uds pero en este caso a nivel de día, 
#por lo que tendremos más filas y más outliers

Q1 = df_Sales_Rotura['Sales_Uds'].quantile(0.25)
Q3 = df_Sales_Rotura['Sales_Uds'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 5 * IQR
upper_bound = Q3 + 5 * IQR

outliers5 = df_Sales_Rotura[(df_Sales_Rotura['Sales_Uds'] < lower_bound) | (df_Sales_Rotura['Sales_Uds'] > upper_bound)]
print(outliers5)


# En este caso llegamos a las 40943 outliers con un umbral de 5.

# Vemos que tenemos muchos outliers también

# In[65]:


#Si usamos el método zscore sobre los mismos datos:

threshold = 3
outliers6 = df_Sales_Rotura[np.abs(stats.zscore(df_Sales_Rotura['Sales_Uds'])) > threshold]

outlier_indices6 = outliers6.index

print(outlier_indices6)


# Obtenemos para el mismo campo 40951 outliers pero con un umbral menor (de 3), por lo que usaremos este método para eliminar los outliers.

# In[66]:


#Hacemos drop para los outliers de df_Sales_Rotura para la columna Sales_Uds usando los resultados de Zscore

df_Sales_Rotura_no_outliers = df_Sales_Rotura.drop(outlier_indices6)


# In[67]:


#Ahora también eliminaremos los valores por debajo de 0

df_Sales_Rotura_no_outliers  = df_Sales_Rotura_no_outliers[(df_Sales_Rotura_no_outliers['Sales_Uds']>= 0)]


# In[68]:


sns.boxplot(x=df_Sales_Rotura_no_outliers['Sales_Uds'])
plt.show()


# Tendremos muchos menos outliers en este caso, habiendo eliminado también los valores negativos.

# In[ ]:




