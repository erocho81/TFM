-- En este código crearemnos nuevas columnas y añadiremos valores a esas columnas.

-- ALTER TABLE Product añadiendo columnas Cost_Price y Sell_Price.

ALTER TABLE dbo.Product
ADD  Cost_Price DECIMAL(5,2),
 	Sell_Price DECIMAL(5,2);


--Añadimos valores inventados para las nuevas columnas:

UPDATE [dbo].[Product]
SET Cost_Price = 
    CASE 
        WHEN Product_Code = 'Brit090' THEN '8.57'
        WHEN Product_Code = 'Brit555' THEN '10.30'
		WHEN Product_Code = 'Brit627' THEN '12.25'
		WHEN Product_Code = 'Brit700' THEN '15.38'
		WHEN Product_Code = 'Dome004' THEN '8.30'
		WHEN Product_Code = 'Dome019' THEN '9.50'
		WHEN Product_Code = 'Dome104' THEN '7.15'
		WHEN Product_Code = 'Dome164' THEN '6.20'
		WHEN Product_Code = 'Dome200' THEN '16.45'
		WHEN Product_Code = 'Dome206' THEN '17.38'
		WHEN Product_Code = 'Dome213' THEN '19.78'
		WHEN Product_Code = 'Dome363' THEN '22.30'
		WHEN Product_Code = 'Dome415' THEN '21.50'
		WHEN Product_Code = 'Dome427' THEN '18.56'
		WHEN Product_Code = 'Dome459' THEN '27.40'
		WHEN Product_Code = 'Dome464' THEN '29.37'
		WHEN Product_Code = 'Dome514' THEN '33.20'
		WHEN Product_Code = 'Dome527' THEN '40.57'
		WHEN Product_Code = 'Dome564' THEN '35.48'
		WHEN Product_Code = 'Dome575' THEN '20.90'
		WHEN Product_Code = 'Dome586' THEN '18.56'
		WHEN Product_Code = 'Dome611' THEN '30.07'
		WHEN Product_Code = 'Dome615' THEN '45.90'
		WHEN Product_Code = 'Dome735' THEN '48.57'
		WHEN Product_Code = 'Dome762' THEN '52.03'
		WHEN Product_Code = 'Dome770' THEN '55.07'
		WHEN Product_Code = 'Dome797' THEN '58.57'
		WHEN Product_Code = 'Dome907' THEN '60.30'
		WHEN Product_Code = 'Don122' THEN '15.30'
		WHEN Product_Code = 'Don843' THEN '16.27'
		WHEN Product_Code = 'Fren043' THEN '7.57'
		WHEN Product_Code = 'Fren323' THEN '12.23'
		WHEN Product_Code = 'Fren628' THEN '18.81'
		WHEN Product_Code = 'Fren771' THEN '20.56'
		WHEN Product_Code = 'Fren831' THEN '19.91'
		WHEN Product_Code = 'Fren933' THEN '23.19'
		WHEN Product_Code = 'Inte190' THEN '28.63'
		WHEN Product_Code = 'Inte227' THEN '32.43'
		WHEN Product_Code = 'Inte314' THEN '36.62'
		WHEN Product_Code = 'Inte327' THEN '42.57'
		WHEN Product_Code = 'Inte404' THEN '35.41'
		WHEN Product_Code = 'Inte414' THEN '63.08'
		WHEN Product_Code = 'Inte615' THEN '20.57'
		WHEN Product_Code = 'Inte657' THEN '18.40'
		WHEN Product_Code = 'Inte755' THEN '16.05'
		WHEN Product_Code = 'Inte943' THEN '17.57'
		WHEN Product_Code = 'Inte947' THEN '9.25'
		WHEN Product_Code = 'Natu079' THEN '40.30'
		WHEN Product_Code = 'Natu122' THEN '42.06'
		WHEN Product_Code = 'Natu123' THEN '45.09'
		WHEN Product_Code = 'Natu408' THEN '47.58'
		WHEN Product_Code = 'Natu461' THEN '49.07'
		WHEN Product_Code = 'Natu508' THEN '53.57'
		WHEN Product_Code = 'Natu623' THEN '56.03'
		WHEN Product_Code = 'Natu723' THEN '60.20'
		WHEN Product_Code = 'Natu969' THEN '62.30'
		WHEN Product_Code = 'Trad110' THEN '64.40'
		WHEN Product_Code = 'Trad210' THEN '66.60'
		WHEN Product_Code = 'Trad310' THEN '8.37'
		WHEN Product_Code = 'Trad610' THEN '5.90'

		ELSE ''
    END,

Sell_Price = 
    CASE 
        WHEN Product_Code = 'Brit090' THEN '10.25'
        WHEN Product_Code = 'Brit555' THEN '12.50'
		WHEN Product_Code = 'Brit627' THEN '15.25'
		WHEN Product_Code = 'Brit700' THEN '18.45'
		WHEN Product_Code = 'Dome004' THEN '12.75'
		WHEN Product_Code = 'Dome019' THEN '13.50'
		WHEN Product_Code = 'Dome104' THEN '9'
		WHEN Product_Code = 'Dome164' THEN '8.20'
		WHEN Product_Code = 'Dome200' THEN '22.35'
		WHEN Product_Code = 'Dome206' THEN '25.40'
		WHEN Product_Code = 'Dome213' THEN '26'
		WHEN Product_Code = 'Dome363' THEN '27.30'
		WHEN Product_Code = 'Dome415' THEN '25.50'
		WHEN Product_Code = 'Dome427' THEN '23.75'
		WHEN Product_Code = 'Dome459' THEN '32'
		WHEN Product_Code = 'Dome464' THEN '34.30'
		WHEN Product_Code = 'Dome514' THEN '36.70'
		WHEN Product_Code = 'Dome527' THEN '46.90'
		WHEN Product_Code = 'Dome564' THEN '42.85'
		WHEN Product_Code = 'Dome575' THEN '25.99'
		WHEN Product_Code = 'Dome586' THEN '23.50'
		WHEN Product_Code = 'Dome611' THEN '36.30'
		WHEN Product_Code = 'Dome615' THEN '50.99'
		WHEN Product_Code = 'Dome735' THEN '54.20'
		WHEN Product_Code = 'Dome762' THEN '56'
		WHEN Product_Code = 'Dome770' THEN '58'
		WHEN Product_Code = 'Dome797' THEN '65.50'
		WHEN Product_Code = 'Dome907' THEN '70.30'
		WHEN Product_Code = 'Don122' THEN '18'
		WHEN Product_Code = 'Don843' THEN '23'
		WHEN Product_Code = 'Fren043' THEN '12.59'
		WHEN Product_Code = 'Fren323' THEN '15.65'
		WHEN Product_Code = 'Fren628' THEN '23.40'
		WHEN Product_Code = 'Fren771' THEN '27.35'
		WHEN Product_Code = 'Fren831' THEN '26.25'
		WHEN Product_Code = 'Fren933' THEN '28.10'
		WHEN Product_Code = 'Inte190' THEN '33.40'
		WHEN Product_Code = 'Inte227' THEN '36.60'
		WHEN Product_Code = 'Inte314' THEN '42.65'
		WHEN Product_Code = 'Inte327' THEN '48.79'
		WHEN Product_Code = 'Inte404' THEN '43.50'
		WHEN Product_Code = 'Inte414' THEN '80.30'
		WHEN Product_Code = 'Inte615' THEN '29.50'
		WHEN Product_Code = 'Inte657' THEN '25.40'
		WHEN Product_Code = 'Inte755' THEN '24.05'
		WHEN Product_Code = 'Inte943' THEN '22.27'
		WHEN Product_Code = 'Inte947' THEN '13'
		WHEN Product_Code = 'Natu079' THEN '48.70'
		WHEN Product_Code = 'Natu122' THEN '50.45'
		WHEN Product_Code = 'Natu123' THEN '54.20'
		WHEN Product_Code = 'Natu408' THEN '56.49'
		WHEN Product_Code = 'Natu461' THEN '60'
		WHEN Product_Code = 'Natu508' THEN '63'
		WHEN Product_Code = 'Natu623' THEN '65.25'
		WHEN Product_Code = 'Natu723' THEN '74.30'
		WHEN Product_Code = 'Natu969' THEN '75'
		WHEN Product_Code = 'Trad110' THEN '77.77'
		WHEN Product_Code = 'Trad210' THEN '78.87'
		WHEN Product_Code = 'Trad310' THEN '12'
		WHEN Product_Code = 'Trad610' THEN '7'
		ELSE ''
    END;

-- Creamos la columna calculada con márgen por unidad para la tabla Product.

ALTER TABLE [dbo].[Product]
ADD Margin DECIMAL(5, 2);

UPDATE [dbo].[Product]
SET Margin = Sell_Price - Cost_Price;

--Creación de columna calculada Delivery_MONTH para DeliveryDay para obtener los datos de mes de la columna de fecha existente Delivery_DAY
ALTER TABLE dbo.DeliveryDay
ADD Delivery_MONTH int;

UPDATE dbo.DeliveryDay
SET Delivery_MONTH =  MONTH([Delivery_DAY]);

--Creación de columna calculada OoS_MONTH para OoSDay
ALTER TABLE dbo.OoSDay
ADD OoS_MONTH int;

UPDATE dbo.OoSDay
SET OoS_MONTH =  MONTH([OoS_DAY]);

--Creación de la columna calculada Sales_MONTH para SalesDay
ALTER TABLE dbo.SalesDay
ADD Sales_MONTH int;

UPDATE dbo.SalesDay
SET Sales_MONTH =  MONTH([Sales_DAY]);

--Creación de la columna calculada Route_MONTH para RouteDay
ALTER TABLE dbo.RouteDay
ADD Route_MONTH int;

UPDATE dbo.RouteDay
SET Route_MONTH =  MONTH([Route_DAY]);

-- Además hemos añadido las siguientes tablas
--Postal_Codes que incluye datos de Código Postal, Población y Provincia.
--Holidays que incluye datos de Provincia, Fecha e indicativo de festivo según provincia.

-- La idea es poder reacionar la tabla Affiliated con códigos postales para obtener datos de población y provincia.
-- La tabla Holidays se puede relacionar con cualquier tabla que tenga datos de fecha por día.