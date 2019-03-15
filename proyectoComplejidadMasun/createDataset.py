import nltk
# remove stop words (meaningless words for the language-meaning)
from nltk.corpus import stopwords
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pandas import DataFrame
import numpy as np

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
	returnlist    = []
	totalPOSvalue = 0.0
	# first define a dictionary to get the code of every POS
	code_dict = {'CC':1.0, 'CD': 2.0, 'DT': 3.0, 'EX': 4.0, 'FW': 5.0,
				 'IN': 6.0, 'JJ': 7.0, 'JJR': 8.0, 'JJS': 9.0, 'LS': 10.0,
				 'MD': 11.0, 'NN': 12.0, 'NNS': 13.0, 'NNP': 14.0, 'NNPS': 15.0,
				 'PDT': 16.0, 'POS': 17.0, 'PRP': 18.0, 'PRP$': 19.0,
				 'RB': 20.0, 'RBR': 21.0, 'RBS': 22.0, 'RP': 23.0, 'SYM': 24.0,
				 'TO': 25.0, 'UH': 26.0, 'VB': 27.0, 'VBD': 28.0, 'VBG': 29.0,
				 'VBN': 30.0, 'VBP':31.0, 'VBZ': 32.0, 'WDT': 33.0, 'WP': 34.0, 
				 'WP$': 35.0, 'WRB': 36.0}
	for pair in thelist:
		try:
			element 		= pair[1]
			totalPOSvalue   = totalPOSvalue + code_dict[element]
			returnlist.append(code_dict[element])		
		except:
			element 		= 0
			totalPOSvalue   = totalPOSvalue 

		
	return (returnlist, totalPOSvalue)

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

def addLabelAtTheEnd(listOfList, label, factor):

	liClass1     = [1.0, 0.0, 0.0]
	liClass2     = [0.0, 1.0, 0.0]
	liClass3     = [0.0, 0.0, 1.0]
	theListClass = []

	if(label == 1.0):
		#class1
		theListClass = liClass1
		theListClass = theListClass * factor
	elif(label == 2.0):
		theListClass = liClass2
		theListClass = theListClass * factor
	else:
		theListClass = liClass3
		theListClass = theListClass * factor

	for sublist in listOfList:
		sublist.append(np.array(theListClass))

	#print(listOfList)
	return listOfList
def addLabelAtTheEnd2(listOfList, label):

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

def computeTotalPOS(sublist):
	#sublist is list of pais, i.e (word, POS value)
	total = 0.0
	code_dict = {'CC':1., 'CD': 2.0, 'DT': 3.0, 'EX': 4.0, 'FW': 5.0,
				 'IN': 6.0, 'JJ': 7.0, 'JJR': 8.0, 'JJS': 9.0, 'LS': 10.0,
				 'MD': 11.0, 'NN': 12.0, 'NNS': 13.0, 'NNP': 14.0, 'NNPS': 15.0,
				 'PDT': 16.0, 'POS': 17.0, 'PRP': 18.0, 'PRP$': 19.0,
				 'RB': 20.0, 'RBR': 21.0, 'RBS': 22.0, 'RP': 23.0, 'SYM': 24.0,
				 'TO': 25.0, 'UH': 26.0, 'VB': 27.0, 'VBD': 28.0, 'VBG': 29.0,
				 'VBN': 30.0, 'VBP':31.0, 'VBZ': 32.0, 'WDT': 33.0, 'WP': 34.0, 
				 'WP$': 35.0, 'WRB': 36.0}

	for pair in sublist:
		try:
			element  = pair[1]	
			subtotal = code_dict[element]	
		except:
			subtotal = 0.0
		total = total + float(subtotal)

	return total

def calculate_len(codifiedList0):
	max = 0
	for i in codifiedList0:
		local_max = len(i)
		if local_max > max:
			max = local_max

	return max


def main():
	#script. Main?
	#Escribimos la primera linea del csv
	st = "Doc_ID,TAGS,CLASS\n"

	#Genera una lista con los archivos de la carpeta 1
	dirname     = "corp/1"
	fls         = os.listdir(dirname)
	i           = 0
	listoflist  = []
	list0       = []
	factor      = 50
	for f in fls:
		#Guardamos el id del texto
		#i = i + 1
		#Obtenemos los tags para f
		tags = tag(gettext("corp/1/" + f))
		#Creamos un dicionario con los tags obtenidos
		d1 = dict(tags)
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
		d2 = dict(tags)
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


	#corpus 3
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
		d3 = dict(tags)
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
	for li in list0: # lis is a list of pairs (letter, POS)
		(newlist, totalPOSsub) 	= getCode(li)
		normalizedlist0			= []
		for element in newlist:
			element = element / totalPOSsub
			roundelement  = round(element, 3)
			
			normalizedlist0.append(roundelement)
		codifiedList0.append(normalizedlist0)

	for li in list1:
		(newlist, totalPOSsub)  = getCode(li)
		normalizedlist1			= []
		for element in newlist:
			element = element / totalPOSsub
			roundelement  = round(element, 3)
			normalizedlist1.append(roundelement)
		codifiedList1.append(normalizedlist1)

	for li in list2:
		(newlist, totalPOSsub) = getCode(li)
		normalizedlist2			= []
		for element in newlist:
			element = element / totalPOSsub
			roundelemnt  = round(element, 3)
			normalizedlist2.append(roundelemnt)
		codifiedList2.append(normalizedlist2)



	# padding
	codifiedList0 = padding(codifiedList0, codifiedList0, codifiedList1, codifiedList2)
	codifiedList1 = padding(codifiedList1, codifiedList0, codifiedList1, codifiedList2)
	codifiedList2 = padding(codifiedList2, codifiedList0, codifiedList1, codifiedList2)
	# normalize using minmax
	#codifiedList0 = normalize(codifiedList0)
	#codifiedList1 = normalize(codifiedList1)
	#codifiedList2 = normalize(codifiedList2)


	# Add label at the end.
	# li means the maximum length of the input vector
	# factor is determined by the expiriments.
	# the minimum output vector's length would be 3.

	liLocal0 = calculate_len(codifiedList0)
	liLocal1 = calculate_len(codifiedList1)
	liLocal2 = calculate_len(codifiedList2)
	li 		 = max(liLocal0, liLocal1, liLocal2)
	#before we were adding a factor that said: (li/factor).
	#now we obly use 1.
	#addLabelAtTheEnd(codifiedList0, 1.0, 1)

	
	li = calculate_len(codifiedList1)
	#addLabelAtTheEnd(codifiedList1, 2.0, 1)


	
	li = calculate_len(codifiedList2)
	#addLabelAtTheEnd(codifiedList2, 3.0, 1)
	
	addLabelAtTheEnd2(codifiedList0, 1)
	addLabelAtTheEnd2(codifiedList1, 2)
	addLabelAtTheEnd2(codifiedList2, 3)









	#codifiedList0 = addLabelAtTheEnd(codifiedList0, 1.0, factor)#
	#codifiedList1 = addLabelAtTheEnd(codifiedList1, 2.0, factor)#
	#codifiedList2 = addLabelAtTheEnd(codifiedList2, 3.0, factor)#
	"""
	codifiedList0 = addLabelAtTheEnd2(codifiedList0, 1.0)#
	codifiedList1 = addLabelAtTheEnd2(codifiedList1, 2.0)#
	codifiedList2 = addLabelAtTheEnd2(codifiedList2, 3.0)#
	"""

	# Concatenate all the lists in one big list
	concatenated  = concatenateLists(codifiedList0, codifiedList1, codifiedList2)


		
		
	
	

	#transform to dataframes:
	df 			= transformToDataFrame(concatenated)

	df_shuffled = df.sample(frac = 1)

	print(df_shuffled)
	df_shuffled.to_csv("outcome.csv", index=False) 


	print(df)


if __name__ =="__main__":
	main()



