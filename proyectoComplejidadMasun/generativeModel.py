import glob
import os


directory = '/home/fernsndo/Documents/proyecto_complejidad/proyectoComplejidadMasun/corp/1/'
file_list = glob.glob(os.path.join(directory, "1", "*.txt"))


corpus = []

for file in directory:
	with open(file) as f_input:
		corpus.append(f_input.read())

print(corpus)
