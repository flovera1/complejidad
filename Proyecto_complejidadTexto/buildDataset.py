
#Mineria de datos
#crea el dataset

#Importaciones:
import os

dataSetFile = open("dataset1.csv", "w")
tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
 "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS",
 "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP",
 "WP$", "WRB"]

# se escribe la primera linea del archivo con los nombres de los atributos
# dataSetFile.write("id;") 
for i in tags:
	dataSetFile.write(i)
	dataSetFile.write(";")
dataSetFile.write("Class\n")

# para cada texto en las carpetas de poscorp se cuentan las incidencias
# de cada postag y se escribe en el csv
dirname = "poscorp/"
for i in range (1,4):
	filePath = dirname+str(i) # -> poscorp/1 poscorp/2 ...
	files = os.listdir(filePath)
	for f in files:
		# dataSetFile.write(f+";") # -> el id pa diferencialos mientras
		for t in tags:
			wFile = open(filePath+"/"+f, "r")
			tIncidence = 0
			for line in wFile.readlines():
				tagcito = line.split(",")[2]
				tagcote = t+"\n"
				if tagcito == tagcote:
					tIncidence += 1
			dataSetFile.write(str(tIncidence))
			dataSetFile.write(";")		
		dataSetFile.write(str(i)+"\n")