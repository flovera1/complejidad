
Mineria de datos
Readme

El archivo pt.py aplica el POS-tagger de nltk a los textos
de la carpeta corp, generando los archivos contenidos en la
carpeta poscorp. En cada archivo de esta carpeta se enumeran
las palabras del texto y sus POS tags correspondientes.     

El archivo buildDataset.py genera el csv a partir de los datos
contenidos en la carpeta poscorp, generando el archivo dataset1.csv.
El archivo tiene 373 filas mas el header.
En este archivo cada fila corresponde a un texto.
Los atributos son los POS tags donde los valores son el
numero de incidencias de estos atributos en cada texto.	
La clase es el nivel de complejidad de los textos: {1,2,3}

El archivo tf.py prepara los archivos en corps para 
aplicarles el tf-idf, generando el archivo tfidf.csv. 
El archivo tiene 373 filas mas el header.
En este archivo cada fila corresponde a un texto. 
Los atributos son el id del documento y un string que contiene 
las palabras del texto sustituidas por su POS tag respectivo. 
La clase es el nivel de complejidad de los textos.

El archivo tagCodes.py genera tagCodes.csv. Para generar este archivo
se calculo la longitud promedio de los textos. Si un texto tiene una 
longitud mayor a la longitud promedio es dividido en varios textos.
Si un texto tiene una longitud menor a la longitud promedio se 
rellena el resto con ceros.
El archivo tiene 803 filas mas el header.
En este archivo cada fila corresponde a 33 palabras de un texto.
El numero de atributos corresponde a la longitud promedio de los 
textos (33), donde el valor de cada atributo es el codigo
correspondiente al tag que corresponde a esa palabra en el texto.
La clase es el nivel de complejidad de los textos: {1,2,3}

El archivo neural.py contiene la implementacion de la red neuronal.
A la red se le pueden suministrar los csv:
	tagCodes.csv
	dataset1.csv

Los resultados de la red fueron muy malos, lo cual se puede deber a 
un error en la implementacion o a falta de instancias para el
entrenamiento adecuado de la red.
