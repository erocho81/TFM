--COMPLETNESS CHECK:revisi�n de completitud de las diferentes tablas

--Revisi�n de valores NULL. En general no hay valores NULL para ninguna tabla

--No hay Nulls para Affiliated_Outlets
SELECT *
FROM Affiliated_Outlets
WHERE Affiliated_Code IS NULL 
	OR Affiliated_NAME IS NULL
	OR POSTALCODE IS NULL 
	OR Engage IS NULL 
	OR Management_Cluster IS NULL
	OR Location IS NULL 
	OR Tam_m2 IS NULL;

-- No hay Nulls para DeliveryDay
SELECT *
FROM DeliveryDay
WHERE Delivery_DAY IS NULL 
	OR Affiliated_Code IS NULL
	OR Product_Code IS NULL
	OR Delivery_Uds IS NULL

-- No hay Nulls para OoSDay
SELECT *
FROM OoSDay
WHERE OoS_DAY IS NULL 
	OR Affiliated_Code IS NULL
	OR Product_Code IS NULL

-- No hay Nulls para Product
SELECT *
FROM Product
WHERE Product_Code IS NULL 
	OR SIZE IS NULL
	OR Format IS NULL

-- No hay Nulls para RouteDay

SELECT *
FROM RouteDay
WHERE Route_DAY IS NULL 
	OR Affiliated_Code IS NULL

-- No hay Nulls para SalesDay
SELECT *
FROM SalesDay
WHERE Sales_DAY IS NULL 
	OR Affiliated_Code IS NULL
	OR Product_Code IS NULL
	OR Sales_Uds IS NULL

-------------------------------------------

--Revisi�n de integridad referencial entre tablas:

-- Vamos a intentar identificar si hay claves for�neas que no tienen correspondecia con claves principales y que por lo tanto impiden por ejemplo hacer Joins de manera correcta

-- Revisamos si en DeliveryDay hay registros de Affiliated_Code que no est�n en la tabla Affiliated_Outlets.
-- Vemos que no hay valores ausentes para DeliveryDay con Affiliated_Code- Affiliated_Outlets
SELECT *
FROM DeliveryDay d

LEFT JOIN Affiliated_Outlets a ON d.Affiliated_Code = a.Affiliated_Code
WHERE a.Affiliated_Code IS NULL;

-- Tampoco hay valores ausentes para DeliveryDay con Product_Code de la tabla Product
SELECT *
FROM DeliveryDay d

LEFT JOIN Product a ON d.Product_Code = a.Product_Code
WHERE a.Product_Code IS NULL;

-- Tampoco hay valores ausentes para DeliveryDay con  Affiliated_Code  de la tabla SalesDay
SELECT *
FROM DeliveryDay d

LEFT JOIN SalesDay a ON d.Affiliated_Code = a.Affiliated_Code
WHERE a.Affiliated_Code IS NULL;


-- Revisamos registros en la tabla DeliveryDay que no se corresponda con registros en la tabla de producto para Product code y para los mismos Affiliated_Code tambi�n de Product

SELECT d.*
FROM DeliveryDay d

LEFT JOIN Product p ON d.Product_Code = p.Product_Code
WHERE p.Product_Code IS NULL 
	AND d.Affiliated_Code IN (
    SELECT Affiliated_Code
    FROM Product
	);

----

-- Hacemos lo mismo con OoSDay para revisar si hay Affiliated_Code que no est� en la tabla Affiliated_Outlets
-- No hay valores ausentes para OoSDay con Affiliated_Code- Affiliated_Outlets
SELECT *
FROM OoSDay o

LEFT JOIN Affiliated_Outlets a ON o.Affiliated_Code = a.Affiliated_Code
WHERE a.Affiliated_Code IS NULL;

-- Tampoco hay valores de OoSDay ausentes en Product_Code- Product
SELECT *
FROM OoSDay o

LEFT JOIN Product a ON o.Product_Code = a.Product_Code
WHERE a.Product_Code IS NULL;

----

-- Para RouteDay revisamos lo registros de Affiliated_Code en la tabla Affiliated_Outlets
-- No hay valores ausentes para RouteDay con Affiliated_Code- Affiliated_Outlets
SELECT *
FROM RouteDay r

LEFT JOIN Affiliated_Outlets a ON r.Affiliated_Code = a.Affiliated_Code
WHERE a.Affiliated_Code IS NULL;

----

-- Revisamos para la tabla SalesDay registros de Affiliated_Code que no est�n en Affiliated_Outlets
-- No hay valores ausentes para SalesDay con Affiliated_Code- Affiliated_Outlets
SELECT *
FROM SalesDay s

LEFT JOIN Affiliated_Outlets a ON s.Affiliated_Code = a.Affiliated_Code
WHERE a.Affiliated_Code IS NULL;

-- Tampoco hay valores ausentes para SalesDay con Product_Code- Product
SELECT *
FROM SalesDay s

LEFT JOIN Product a ON s.Product_Code = a.Product_Code
WHERE a.Product_Code IS NULL;

-------------------------------------------

--Revisi�n de consistencia de datos:
--Verificaremos si las reglas de consistencia de datos se est�n siguiendo.

--Por ejemplo verificamos su hay unidades negativas de ventas o entregas:

-- Para DeliveryDay-Delivery_Uds s� que hay unidades negativas debido a entregas err�neas que se compensan.
SELECT *
FROM DeliveryDay
WHERE Delivery_Uds < 0;

-- Para SalesDay-Sales_Uds tambi�n hay unidades negativas, debido a robos, deterioros, muestras, etc.
SELECT *
FROM SalesDay
WHERE Sales_Uds <0;

--Por el momento mantendremos estas uds negativas y revisaremos qu� hacer de cara al an�lisis en Python.

-------------------------------------------


-- VALORES DUPLICADOS

--Tabla Product:

SELECT  Product_Code,
        COUNT(*)
FROM dbo.Product
GROUP BY Product_Code     
HAVING COUNT(*) > 1

-- Vemos que el c�digo de referencia Natu122 est� duplicado, lo que no es correcto ya que deber�an ser valores �nicos.

-- Revisamos esa referencia para obtener m�s informaci�n
SELECT  Product_Code,
		Size,
		Format
FROM dbo.Product
WHERE Product_Code = 'Natu122'


--- Es probable que se trate de otro producto distinto, ya que el formato es distinto tambi�n. En este caso, si consideramos que es un producto distinto, 
--podriamos cambiar el Product_Code de uno de ellos. Aunque no podremos relacionarlo con otras tablas, ya que el c�digo no existir� hasta que realicemos el cambio, 
-- por lo que solo servir�a para futuras cargas de datos. 

-- Por ejemplo podr�amos cambiar el producto Natu122 con Format ASL al c�digo Natu123, que no existe hasta ahora en la tabla Product.

UPDATE Product
SET Product_Code = 'Natu123'
WHERE Product_Code = 'Natu122'
	 AND Format = 'ASL';

-- Ahora no tendremos duplicados y podremos establecer el Product_Code como primary key, lo antes no era posible debido a ese valor duplicado. Veremos que cada c�digo existe en la tabla:

SELECT  Product_Code,
		Size,
		Format
FROM dbo.Product
WHERE Product_Code = 'Natu122' OR Product_Code = 'Natu123'


--Tabla Affiliated Outlets:
--En este caso no tenemos duplicados, los c�digos de Affiliated_Code son �nicos, como deber�an ser para esta tabla.
SELECT  Affiliated_Code,
        COUNT(*)
FROM dbo.Affiliated_Outlets
GROUP BY Affiliated_Code     
HAVING COUNT(*) > 1


-------------------------------------------

--- PRECISI�N DE DATOS Y VALIDEZ.
---REVISI�N DE DOMINIONS: validaremos que los datos se encuentran entre los rangos o dominios establecidos. 
---Por ejemplo, ver si hay columnas num�ricas con grandes outliers o si las fechas est�n dentro de rangos razonables

-- Revisamos posbles outliers en la columna Tam_m2 de Affiliated_Outlets:
-- No hay ning�n datos que destaque demasiado a priori, pero s� bastantes ND.

SELECT
	Tam_m2,
	COUNT (*) AS total_count_m2
FROM dbo.Affiliated_Outlets
GROUP BY Tam_m2
ORDER BY total_count_m2 DESC

--Tampoco parece haber outliers en la cantidad de PostalCode.No hay valores muy altos para ning�n c�digo postal que destaqu demasiado.
SELECT
	POSTALCODE,
	COUNT (*) AS total_postal_code
FROM [dbo].[Affiliated_Outlets]
GROUP BY POSTALCODE
ORDER BY total_postal_code DESC

--Consistencia en fechas: para las fechas, revisamos si hay fechas anteriores a 2015 0 2023 en DeliveryDay. 
--En teor�a los datos est�n acotados entre Marzo y Diciembre de 2015 aproximadamente.

--Revisamos si hay fechas anteriores a 2015 o posteriores a 2023 en DeliveryDay.
-- No encontramos nada.
SELECT *
FROM DeliveryDay
WHERE Delivery_DAY < '2015-01-01' 
	OR Delivery_DAY > '2023-12-31' 

-- Hacemos lo mismo para OoSDay, RouteDay y SalesDay, y vemos que tampoco hay ninguna fecha fuera de rango.
SELECT *
FROM OoSDay
WHERE OoS_DAY < '2015-01-01' 
	OR OoS_DAY > '2023-12-31'

SELECT *
FROM RouteDay
WHERE Route_DAY < '2015-01-01' 
	OR Route_DAY > '2023-12-31'

SELECT *
FROM SalesDay
WHERE Sales_DAY < '2015-01-01' 
	OR Sales_DAY > '2023-12-31'

