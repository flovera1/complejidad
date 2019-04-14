import glob
import errno


path  		= '/home/fernsndo/Documents/proyecto_complejidad/proyectoComplejidadMasun/corp/2/*.txt'
files 		= glob.glob(path)
bigCorpus	= []

	
for name in files:
    try:
        with open(name) as f:
        	for line in f:
        		currentPlace = line[:-1]
        		bigCorpus.append(currentPlace)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise


with open('WHOLECORPUS.txt', 'w') as f:
    for item in bigCorpus:
        f.write("%s\n" % item)



print(bigCorpus)