
#Mineria de datos
#POS TAGGER

#Importaciones:
import nltk
import os

#Funcion que genera el POS tag de una oracion
def tag(s):
	t = nltk.word_tokenize(s)
	return nltk.pos_tag(t)

#Funcion que abre un archivo y devuelve su contenido
def gettext(f):
	fl = open(f,'r')
	txt = ""
	for line in fl.readlines():
		txt = txt + line + "\n"
	return txt

#Funcion que escribe un archivo con el resultado del pos tag
def wpos(f,l):
	fl = open(f,'w')
	i = 0
	for tp in l:
		i = i + 1
		fl.write(str(i) + "," + tp[0] + "," + tp[1] + "\n")

#Genera los pos tags para los textos de la carpeta 1
dirname = "corp/1"
fls = os.listdir(dirname)
for f in fls:
	wpos('poscorp/1/' + f,tag(gettext("corp/1/" + f)))

#Genera los pos tags para los textos de la carpeta 2
dirname = "corp/2"
fls = os.listdir(dirname)
for f in fls:
	wpos('poscorp/2/' + f,tag(gettext("corp/2/" + f)))

#Genera los pos tags para los textos de la carpeta 3
dirname = "corp/3"
fls = os.listdir(dirname)
for f in fls:
	wpos('poscorp/3/' + f,tag(gettext("corp/3/" + f)))