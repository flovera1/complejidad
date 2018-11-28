
#Mineria de datos
#crea el dataset

#Importaciones:
import os

dataSetFile = open("tagCodes.csv", "w")
tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
 "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS",
 "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP",
 "WP$", "WRB"]

# para cada texto en las carpetas de poscorp se cuentan las incidencias
# de cada postag y se escribe en el csv
dirname = "poscorp/"
treshold = 33
for i in range (1,34):
	dataSetFile.write("tag"+str(i)+",")
dataSetFile.write("class\n")
for i in range (1,4):
	filePath = dirname+str(i) # -> poscorp/1 poscorp/2 ...
	files = os.listdir(filePath)
	for f in files:
		wFile = open(filePath+"/"+f, "r")
		acc = 0
		string = ""
		for line in wFile.readlines():
			tagcito = line.split(",")[2]
			tagcito = tagcito.split("\n")[0]
			if tagcito != ".":
				codeTag = 1
				for t in tags:
					if (tagcito == t):
						string += str(codeTag)+","
						acc += 1
					codeTag += 1
				if (acc > 32):
					acc = 0
					string += str(i)+"\n"
					dataSetFile.write(string)
					string = "" 
		if (acc > 25):
			for j in range(acc, 33):
				string += "0,"
			dataSetFile.write(string)	
			dataSetFile.write(str(i)+"\n")
		string = ""
		acc = 0