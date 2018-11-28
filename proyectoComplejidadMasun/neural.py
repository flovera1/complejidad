# LSTM for sequence classification in the IMDB dataset
import numpy as np
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
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import BatchNormalization
from keras.layers.normalization import BatchNormalization
import keras.backend as K

import matplotlib.pyplot as plt

from _perceptron import Perceptron

from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

#trying a multilayer perceptron to fit the data
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt



def create_dataset(dataset):
	dataX, dataY = [], []
	for i in range(len(dataset)):
		x = dataset[i][:len(dataset[0])-1]
		y = dataset[i][-1]
		dataX.append(x)
		dataY.append(y)
	return np.array(dataX), np.array(dataY)


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5500
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

# Se carga la data
dataframe = read_csv('outcome.csv', engine='python')
# Se desordenan las instancias
dataframe = dataframe.sample(frac=1)
#print(dataframe)
#dataset   = dataframe.values
#print("DATASET")
#print(dataset)

# se dividen la dataset en 80% para entrenamiento
# y 20% para prueba

X 				 				 = np.array(dataframe.drop(['179'], axis = 1))
y 				 				 = np.array(dataframe['179'])
labelencoder_y_1 				 = LabelEncoder()
y 				 				 = labelencoder_y_1.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

categoricalTrainY 				 = to_categorical(y_train, num_classes=None)
categoricalTestY  				 = to_categorical(y_test, num_classes=None)




#y_train = np.array(y_train, dtype = float)
#y_train = np.reshape(-1, 11)

#y_test = np.array(y_test, dtype = float)
#y_test = np.reshape(-1, 11)

"""

train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)


print("Numero de instancias de dataset de entrenamiento: " + str(len(trainX)))
print("Numero de instancias de dataset de prueba: " + str(len(testX)))

#categoricalTrainY = to_categorical(trainY, num_classes=None)
#categoricalTestY = to_categorical(testY, num_classes=None)

# normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# trainX = scaler.fit_transform(trainX)
# testX = scaler.fit_transform(testX)
"""

embedding_vector_length = 179
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=179))

model.add(LSTM(4,stateful=False,
                      kernel_initializer='he_normal',
                      activation='tanh',
                      dropout = 0.1 , 
                      recurrent_dropout = 0.1 ))
"""
model.add(BatchNormalization(momentum=0.99, 
							epsilon=0.001, 
							center=True, 
                            scale=True, 
                            beta_initializer='zeros', 
                            gamma_initializer='ones', 
                            moving_mean_initializer='zeros', 
                            moving_variance_initializer='ones'))
model.add(LSTM(4, return_sequences = True,
                     stateful=False,
                     kernel_initializer='he_normal',
                      activation='tanh',
                      dropout = 0.1 , 
                      recurrent_dropout = 0.1))
model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, 
                                   scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))


model.add(LSTM(4,
                     stateful=False,
                     kernel_initializer='he_normal',
                      activation='tanh',
                      dropout = 0.1 , 
                      recurrent_dropout = 0.1))
model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, 
                                   scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
"""
model.add(Dense(3, activation='softmax', kernel_initializer = 'he_normal'))
print (model.summary())

opt  = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
opt2 = optimizers.Adam(lr=0.00001)
opt3 = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
opt4 = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
sgd  = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# do not use logcosh as loss value
# better use binary_crossentropy
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'], weighted_metrics=['accuracy'])
print(model.summary())

early_stopping_monitor = EarlyStopping(patience=3)

model.fit(X_train, categoricalTrainY, validation_split=0.10, epochs=30, batch_size=64, callbacks=[early_stopping_monitor])
# Final evaluation of the model
scores = model.evaluate(X_test, categoricalTestY, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(scores)
# print model.predict(X_test)

"""
clf             = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='adam', verbose=0, tol=1e-8, random_state=1,
                    learning_rate_init=.01)


clf.fit(X_train, y_train)

N_TRAIN_SAMPLES = X_train.shape[0]
N_EPOCHS        = 25
N_BATCH         = 128
N_CLASSES       = np.unique(y_train)

scores_train    = []
scores_test     = []
epoch           = 0

while(epoch < N_EPOCHS):
  print('epoch: ', epoch)
  # SHUFFLING
  random_perm      = np.random.permutation(X_train.shape[0])
  mini_batch_index = 0
  while(True):
    # MINI-BATCH
    indices           = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
    clf.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
    mini_batch_index += N_BATCH
    if mini_batch_index >= N_TRAIN_SAMPLES:
      break
    # SCORE TRAIN
    scores_train.append(clf.score(X_train, y_train))
    # SCORE TEST
    scores_test.append(clf.score(X_test, y_test))

    epoch += 1




fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].plot(scores_train)
ax[0].set_title('Train')
ax[1].plot(scores_test)
ax[1].set_title('Test')
fig.suptitle("Accuracy over epochs", fontsize=14)
plt.show()



plt.plot(scores_train, color='green', alpha=0.8, label='Train')
plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
plt.title("Accuracy over epochs", fontsize=14)
plt.xlabel('Epochs')
plt.legend(loc='upper left')
plt.show()

"""