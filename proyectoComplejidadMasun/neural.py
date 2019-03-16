# LSTM for sequence classification in the IMDB dataset
import numpy as np
from pandas import read_csv
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
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

def main():
  # fix random seed for reproducibility
  np.random.seed(7)
  # load the dataset but only keep the top n words, zero the rest
  top_words = 50
  #(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words = top_words)

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

  X 				 				                = np.array(dataframe.drop(['179'], axis = 1))
  y 				 				                = np.array(dataframe['179'])
  #print(y)
  #labelencoder_y_1 	                = LabelEncoder()
  #y 				 				                = labelencoder_y_1.fit_transform(y)



  X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.20, random_state = 42)

  categoricalTrainY 				        = to_categorical(y_train, num_classes=None)
  categoricalTestY  				        = to_categorical(y_test, num_classes=None)

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

  model.add(LSTM(4, return_sequences = True,
                  stateful=False,
                        kernel_initializer='he_normal',
                        activation='relu',
                        dropout = 0.20, 
                        recurrent_dropout = 0.20 ))

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
                        activation='relu',
                        dropout = 0.20, 
                        recurrent_dropout = 0.20))
  '''
  model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, 
                                     scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
  '''

  model.add(LSTM(4,
                       stateful=False,
                       kernel_initializer='he_normal',
                        activation='softmax',
                        dropout = 0.20 , 
                        recurrent_dropout = 0.20))
  '''model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, 
                                     scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))

  '''
  model.add(Dropout(0.2))
  model.add(Dense(4, activation='softmax', kernel_initializer = 'he_normal'))
  print (model.summary())

  opt  = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
  opt2 = optimizers.Adam(lr=0.00001)
  opt3 = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
  opt4 = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
  sgd  = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

  # do not use logcosh as loss value
  # better use binary_crossentropy with adam
  # we used: mean_squared_error with sgd
  # mean_absolute_error

  model.compile(loss = 'binary_crossentropy', optimizer = opt2, metrics = ['accuracy'], weighted_metrics = ['accuracy'])
  #print(model.summary())

  early_stopping_monitor = EarlyStopping(patience = 3)

  model.fit(X_train, categoricalTrainY, validation_split = 0.10, epochs = 50, batch_size = 100, callbacks = [early_stopping_monitor])
  # Final evaluation of the model
  scores = model.evaluate(X_test, categoricalTestY)
  print("Accuracy: %.2f%%" % (scores[1]*100))
  print(scores)
  print(scores[1]*100)


if __name__ =="__main__":
  main()
