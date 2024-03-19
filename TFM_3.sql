---REVISIÓN DE FECHAS Y CREACIÓN DE VISTAS

--Las fechas de las tablas RouteDay y DeliveryDay suponen conceptos diferentes. Mientras que Route_Day nos muestra fechas de entregas preprogramadas, 
--Delivery_Day muestra todas las entregas, tanto las realizadas en fechas programadas como fuera de ellas. Las entregas fuera de Route_Day suponen un coste extra. 

-- Por eso las entregas donde la fecha de RouteDay es igual a la de DeliveryDay son entregas sin coste.

--Por ejemplo podemos comprobar entregas para un afiliado concreto:

SELECT DeliveryDay.Affiliated_Code
	,Delivery_DAY
	,Route_DAY
 
FROM dbo.DeliveryDay

LEFT JOIN dbo.RouteDay
	ON DeliveryDay.Affiliated_Code=RouteDay.Affiliated_Code
	and  DeliveryDay.Delivery_DAY = RouteDay.Route_DAY

where DeliveryDay.Affiliated_Code = 'XCdSAAX'

group by Delivery_DAY,DeliveryDay.Affiliated_Code,Route_DAY


-- También vamos a revisar si las fechas de DeliveryDAY y SalesDay podrían relacionarse mediante el mes de esas fechas.
-- Si bien es cierto que se trata de conceptos distintos, estas fechas nos pueden permitir agrupar datos con cierto márgen de error.

-- Es decir, para DeliveryDAY tendríamos:

SELECT     
    MONTH(Delivery_DAY) AS Delivery_Month,
    Affiliated_Code,
    Product_Code,
    SUM(Delivery_Uds) AS Total_Delivered
FROM DeliveryDay

GROUP BY MONTH(Delivery_DAY),
    Affiliated_Code,
    Product_Code

ORDER BY MONTH(Delivery_DAY),
    Affiliated_Code,
    Product_Code

-- Para Sales_DAY sería:

SELECT
    MONTH(Sales_DAY) AS Sales_Month,
    Affiliated_Code,
    Product_Code,
    SUM(Sales_Uds) AS Total_Sales
FROM 
    SalesDay
	   
GROUP BY    
    MONTH(Sales_DAY),
    Affiliated_Code,
    Product_Code

--ORDER BY     
    --MONTH(Sales_DAY),
    --Affiliated_Code,
    --Product_Code


---Una vez tenemos estas dos queries, vamos a crear una Vista combinandolas junto a otra con datos de rotura.
---Llamaremos a esta vista fact_tables y nos dará información por mes de Affiliated_Code, Product_Code, Total_Delivered, Total_Sales y Total_OoS:

CREATE VIEW fact_tables AS

WITH DeliveredMonthly AS (
    SELECT 
        MONTH(DeliveryDay.Delivery_DAY) AS Delivery_Month,
        DeliveryDay.Affiliated_Code,
        DeliveryDay.Product_Code,
        SUM(DeliveryDay.Delivery_Uds) AS Total_Delivered
    FROM 
		dbo.DeliveryDay

    GROUP BY
        MONTH(DeliveryDay.Delivery_DAY),
        DeliveryDay.Affiliated_Code,
        DeliveryDay.Product_Code
	),

SoldMonthly AS (
    SELECT 
        MONTH(SalesDay.Sales_DAY) AS Sales_Month,
        SalesDay.Affiliated_Code,
        SalesDay.Product_Code,
        SUM(SalesDay.Sales_Uds) AS Total_Sales
    FROM 
        dbo.SalesDay

    GROUP BY
        MONTH(SalesDay.Sales_DAY),
        SalesDay.Affiliated_Code,
        SalesDay.Product_Code
	),

OoSMonthly AS (
	SELECT 
		MONTH (OoSDay.OoS_Day) AS Oos_Month,
		OoSDay.Affiliated_Code,
		OoSDay.Product_Code,
		COUNT (*) AS OoS_Times

	FROM 
		dbo.OoSDay

	GROUP BY
		Affiliated_Code,
		Product_Code,
		MONTH (OoS_Day) 
	)

SELECT 
    COALESCE(DeliveredMonthly.Delivery_Month, SoldMonthly.Sales_Month, OoSMonthly.Oos_Month) AS Month,
    COALESCE(DeliveredMonthly.Affiliated_Code, SoldMonthly.Affiliated_Code, OoSMonthly.Affiliated_Code) AS Affiliated_Code,
    COALESCE(DeliveredMonthly.Product_Code, SoldMonthly.Product_Code, OoSMonthly.Product_Code) AS Product_Code,
    COALESCE(Total_Delivered, 0) AS Total_Delivered,
    COALESCE(Total_Sales, 0) AS Total_Sales,
	COALESCE(OoS_Times,0) AS Total_OoS

FROM 
    DeliveredMonthly

FULL JOIN SoldMonthly 
    ON DeliveredMonthly.Delivery_Month = SoldMonthly.Sales_Month
    AND DeliveredMonthly.Affiliated_Code = SoldMonthly.Affiliated_Code
    AND DeliveredMonthly.Product_Code = SoldMonthly.Product_Code

FULL JOIN OoSMonthly
	ON DeliveredMonthly.Delivery_Month = OoSMonthly.Oos_Month
    AND DeliveredMonthly.Affiliated_Code = OoSMonthly.Affiliated_Code
    AND DeliveredMonthly.Product_Code = OoSMonthly.Product_Code

--ORDER BY 
    --Month, 
    --Affiliated_Code, 
    --Product_Code;


-- Creamos otra vista agrupada por mes llamada all_tables donde añadimos los datos más importantes de las tablas, incluídas poblacion y provincia de la nueva tabla Postal_Codes.
CREATE VIEW all_tables AS

SELECT
	Month,
	fact_tables.Affiliated_Code,
	Affiliated_NAME,
	Affiliated_Outlets.POSTALCODE,
	poblacion,
	provincia,
	Engage,
	Management_Cluster,
	Location,
	Tam_m2,
	fact_tables.Product_Code,
	SIZE,
	Format,
	Cost_price,
	Sell_price,
	Margin,
	Total_Delivered,
	Total_Sales,
	Total_OoS
	
FROM fact_tables

LEFT JOIN Affiliated_Outlets
	ON fact_tables.Affiliated_Code= Affiliated_Outlets.Affiliated_Code

LEFT JOIN Product
	ON fact_tables.Product_Code= Product.Product_Code

LEFT JOIN Postal_Codes
	ON Affiliated_Outlets.POSTALCODE = Postal_Codes.PostalCode

--order by Month

--- Crearemos otra vista llamada Delivery_Route donde relacionaremos datos de las tiendas, entregas por Route_DAY y Delivery_DAY 
-- e indicaremos si las entregas se han realizado dentro de la rura establecida y si los días de entrega se han realizado en día festivo o domingo
-- al cruzar Delivery_DAY con las fechas de festivos de la tabla Holidays. Esta tabla estará indicada por días.

CREATE VIEW Delivery_Route AS

SELECT 
    DeliveryDay.Affiliated_Code,
    Affiliated_Name,
    Affiliated_Outlets.POSTALCODE,
    poblacion,
    Postal_Codes.provincia,
    Engage,
    Management_Cluster,
    Location,
    Tam_m2,
    Delivery_DAY,
    Route_DAY,
    CASE 
        WHEN DeliveryDay.Delivery_DAY <> RouteDay.Route_DAY OR 
             (DeliveryDay.Delivery_DAY IS NULL AND RouteDay.Route_DAY IS NOT NULL) OR
             (DeliveryDay.Delivery_DAY IS NOT NULL AND RouteDay.Route_DAY IS NULL) 
        THEN 'Yes' 
        ELSE 'No' 
    END AS Delivery_out_of_Route,
	ISNULL(Festivo, 'Laborable') AS Festivo


FROM dbo.DeliveryDay

LEFT JOIN dbo.RouteDay 
	ON DeliveryDay.Affiliated_Code = RouteDay.Affiliated_Code 
    AND DeliveryDay.Delivery_DAY = RouteDay.Route_DAY

LEFT JOIN dbo.Affiliated_Outlets 
	ON DeliveryDay.Affiliated_Code = Affiliated_Outlets.Affiliated_Code

LEFT JOIN dbo.Postal_Codes 
	ON Affiliated_Outlets.POSTALCODE = Postal_Codes.PostalCode

LEFT JOIN dbo.Holidays 
	ON Postal_Codes.provincia = Holidays.Provincia
	AND DeliveryDay.Delivery_DAY = Holidays.Fecha

--GROUP BY Delivery_DAY, DeliveryDay.Affiliated_Code, Route_DAY, Affiliated_Name, Affiliated_Outlets.POSTALCODE, poblacion, Postal_Codes.provincia, Engage, Management_Cluster, Location, Tam_m2,Festivo;


--- Finalmente crearemos una tabla similar pero en este caso centrada en los días de venta, relacionando datos de Affiliated, Código de Producto, OoS_Day (día de rotura), Indicador de rotura y Festivo

CREATE VIEW Sales_Rotura AS

SELECT 
    SalesDay.Sales_DAY,
    SalesDay.Affiliated_Code,
    Affiliated_Outlets.POSTALCODE,
    Postal_Codes.poblacion,
    Postal_Codes.provincia,
    SalesDay.Product_Code,
    SalesDay.Sales_Uds,
	OoS_DAY,
    CASE 
        WHEN OoSDay.OoS_DAY IS NOT NULL THEN 'Yes'
        ELSE 'No'
    END AS Rotura,
    ISNULL(Festivo, 'Laborable') AS Festivo
FROM dbo.SalesDay

LEFT JOIN OoSDay ON SalesDay.Sales_DAY = OoSDay.OoS_DAY
    AND SalesDay.Affiliated_Code = OoSDay.Affiliated_Code
    AND SalesDay.Product_Code = OoSDay.Product_Code

LEFT JOIN dbo.Affiliated_Outlets ON SalesDay.Affiliated_Code = Affiliated_Outlets.Affiliated_Code

LEFT JOIN dbo.Postal_Codes ON Affiliated_Outlets.POSTALCODE = Postal_Codes.PostalCode

LEFT JOIN dbo.Holidays ON Postal_Codes.provincia = Holidays.Provincia
    AND SalesDay.Sales_DAY = Holidays.Fecha
