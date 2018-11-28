
#Mineria de datos
#POS TAGGER - Formato TF-IDF

#Importaciones:
import nltk
# remove stop words (meaningless words for the language-meaning)
from nltk.corpus import stopwords
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pandas import DataFrame


#Funcion que genera el POS tag de una oracion
def tag(s):
	t = nltk.word_tokenize(s)
	return nltk.pos_tag(t)

#Funcion que abre un archivo y devuelve su contenido
def gettext(f):
	fl = open(f,'r')
	txt = ""
	for line in fl.readlines():
		# use regular expressions to replace email addresses, URLs, phone numbers, other numbers

		# email addresses with 'email'
		processed = line.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')
		# webadress
		processed = processed.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumbr')
		# phonenumbers
		processed = processed.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumbr')
		# numbers 
		processed = processed.replace(r'\d+(\.\d+)?', 'numbr')
		# Remove punctuation
		processed = processed.replace(r'[^\w\d\s]', ' ')
		# Replace whitespace between terms with a single space
		processed = processed.replace(r'\s+', ' ')
		# Remove leading and trailing whitespace
		processed = processed.replace(r'^\s+|\s+?$', '')
		# change words to lower case - Hello, HELLO, hello are all the same word
		processed = processed.lower()
		# remove stopwords
		stop_words = set(stopwords.words('english'))
		#processed  = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
		txt = txt + processed + "\n"

	return txt


#Funcion que escribe un archivo con el resultado del pos tag
def wpos(f,s):
	fl = open(f,'w')
	fl.write(s)



def getCode(thelist):
	#returnlist
	returnlist = []
	# first define a dictionary to get the code of every POS
	code_dict = {'CC':1., 'CD': 2., 'DT': 3., 'EX': 4., 'FW': 5.,
				 'IN': 6., 'JJ': 7., 'JJR': 8., 'JJS': 9., 'LS': 10.,
				 'MD': 11., 'NN': 12., 'NNS': 13., 'NNP': 14., 'NNPS': 15.,
				 'PDT': 16., 'POS': 17., 'PRP': 18., 'PRP$': 19.,
				 'RB': 20., 'RBR': 21., 'RBS': 22., 'RP': 23., 'SYM': 24.,
				 'TO': 25., 'UH': 26., 'VB': 27., 'VBD': 28., 'VBG': 29.,
				 'VBN': 30., 'VBP':31., 'VBZ': 32., 'WDT': 33., 'WP': 34., 
				 'WP$': 35., 'WRB': 36.}
	for pair in thelist:
		try:
			element = pair[1]
			returnlist.append(code_dict[element])		
		except:
			element = 0
		
	return returnlist

def padding(listoflist, list1, list2, list3):
	returnlist   = []
	maxlongitude = 0

	# this will determine the list's maximum size
	for i in list1:
		newlongitude = len(i)
		if newlongitude > maxlongitude:
			maxlongitude = newlongitude

	for i in list2:
		newlongitude = len(i)
		if newlongitude > maxlongitude:
			maxlongitude = newlongitude

	for i in list3:
		newlongitude = len(i)
		if newlongitude > maxlongitude:
			maxlongitude = newlongitude

	for j in listoflist:
		size = len(j)
		while(size < maxlongitude):
			j.append(0.0)
			size = size + 1
		returnlist.append(j)

	return returnlist

def addLabelAtTheEnd(listOfList, label):

	for sublist in listOfList:
		sublist.append(label)

	return listOfList


def transformToDataFrame(listOfList):

	df = DataFrame(listOfList)
	return df

def concatenateLists(list0, list1, list2):
	newlist = []
	for i in list0:
		newlist.append(i)
	for i in list1:
		newlist.append(i)
	for i in list2:
		newlist.append(i)

	return newlist


def normalize(listOfList):
	
	returnlist = []

	
	for sublist in listOfList:
		newlist = []
		for i in sublist:
			elem = i / 36.0
			newlist.append(elem)

		returnlist.append(newlist)

	return returnlist


def main():
	#script. Main?
	#Escribimos la primera linea del csv
	st = "Doc_ID,TAGS,CLASS\n"

	#Genera una lista con los archivos de la carpeta 1
	dirname = "corp/1"
	fls = os.listdir(dirname)
	i = 0
	listoflist  = []
	list0 = []
	for f in fls:
		#Guardamos el id del texto
		#i = i + 1
		#Obtenemos los tags para f
		tags = tag(gettext("corp/1/" + f))
		#Creamos un dicionario con los tags obtenidos
		d = dict(tags)
		#Creamos un string auxiliar con los tags que contenia el texto
		lists = list(gettext("corp/1/" + f).split(" "))
		listoflist.append(lists)
		# procesar los textos separados
	for li in listoflist:
		ll = []
		for j in li:
			t = nltk.word_tokenize(j)
			pair = nltk.pos_tag(t)
			ll.append(next(iter(pair), None))
	
		#print(ll)
		list0.append(ll)
	#corpus 2
	dirname = "corp/2"
	fls = os.listdir(dirname)
	i = 0
	listoflist  = []
	list1 = []
	for f in fls:
		#Guardamos el id del texto
		#i = i + 1
		#Obtenemos los tags para f
		tags = tag(gettext("corp/2/" + f))
		#Creamos un dicionario con los tags obtenidos
		d = dict(tags)
		#Creamos un string auxiliar con los tags que contenia el texto
		lists = list(gettext("corp/2/" + f).split(" "))
		listoflist.append(lists)
	# procesar los textos separados
	for li in listoflist:
		ll = []
		for j in li:
			t = nltk.word_tokenize(j)
			pair = nltk.pos_tag(t)
			ll.append(next(iter(pair), None))
	
		#print(ll)
		list1.append(ll)


	#corpus 2
	dirname = "corp/3"
	fls = os.listdir(dirname)
	i = 0
	listoflist  = []
	list2 = []
	for f in fls:
		#Guardamos el id del texto
		#i = i + 1
		#Obtenemos los tags para f
		tags = tag(gettext("corp/3/" + f))
		#Creamos un dicionario con los tags obtenidos
		d = dict(tags)
		#Creamos un string auxiliar con los tags que contenia el texto
		lists = list(gettext("corp/3/" + f).split(" "))
		listoflist.append(lists)
	# procesar los textos separados
	for li in listoflist:
		ll = []
		for j in li:
			t = nltk.word_tokenize(j)
			pair = nltk.pos_tag(t)
			ll.append(next(iter(pair), None))
	
		#print(ll)
		list2.append(ll)


	codifiedList0 = []
	codifiedList1 = []
	codifiedList2 = []
	for li in list0:
		newlist = getCode(li)
		codifiedList0.append(newlist)
	for li in list1:
		newlist = getCode(li)
		codifiedList1.append(newlist)
	for li in list2:
		newlist = getCode(li)
		codifiedList2.append(newlist)



	# padding
	codifiedList0 = padding(codifiedList0, codifiedList0, codifiedList1, codifiedList2)
	codifiedList1 = padding(codifiedList1, codifiedList0, codifiedList1, codifiedList2)
	codifiedList2 = padding(codifiedList2, codifiedList0, codifiedList1, codifiedList2)
	# normalize using minmax
	codifiedList0 = normalize(codifiedList0)
	codifiedList1 = normalize(codifiedList1)
	codifiedList2 = normalize(codifiedList2)


	# Add label at the end.
	codifiedList0 = addLabelAtTheEnd(codifiedList0, 1.0)#
	codifiedList1 = addLabelAtTheEnd(codifiedList1, 2.0)#
	codifiedList2 = addLabelAtTheEnd(codifiedList2, 3.0)#
	
	
	# Concatenate all the lists in one big list
	concatenated  = concatenateLists(codifiedList0, codifiedList1, codifiedList2)

	

	
	
	
	

	#transform to dataframes:
	df 			= transformToDataFrame(concatenated)

	df_shuffled = df.sample(frac = 1)

	df_shuffled.to_csv("outcome.csv") 


	print(df_shuffled)


	

if __name__ =="__main__":
	main()



