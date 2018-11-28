# LSTM for sequence classification in the IMDB dataset
import numpy
from pandas import read_csv
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from keras import optimizers


def create_dataset(dataset):
	dataX, dataY = [], []
	for i in range(len(dataset)):
		x = dataset[i][:len(dataset[0])-1]
		y = dataset[i][-1]
		dataX.append(x)
		dataY.append(y)
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 1000
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

# Se carga la data
dataframe = read_csv('outcome.csv', engine='python')
# Se desordenan las instancias
dataframe = dataframe.sample(frac=1)
dataset = dataframe.values
print("DATASET")
print(dataset)

# se dividen la dataset en 80% para entrenamiento
# y 20% para prueba
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)
print("Numero de instancias de dataset de entrenamiento: " + str(len(trainX)))
print("Numero de instancias de dataset de prueba: " + str(len(testX)))

categoricalTrainY = to_categorical(trainY, num_classes=None)
categoricalTestY = to_categorical(testY, num_classes=None)

# normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# trainX = scaler.fit_transform(trainX)
# testX = scaler.fit_transform(testX)

embedding_vector_length = 179
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=179))
model.add(LSTM(4))
model.add(Dense(4, activation='sigmoid'))


sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# do not use logcosh as loss value
# better use binary_crossentropy
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
model.fit(trainX, categoricalTrainY, nb_epoch=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(testX, categoricalTestY, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(scores
#)print model.predict(testX)
