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

#different machine learning algorithms
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


# weight regularization
from keras.regularizers import L1L2





def main():

  n_epochs                          = 500
  batch_size                        = 500
  # fix random seed for reproducibility
  np.random.seed(7)
  # load the dataset but only keep the top n words, zero the rest
  top_words                         = 50
  # Se carga la data
  dataframe                         = read_csv('outcome.csv', engine='python')
  # Se desordenan las instancias
  dataframe                         = dataframe.sample(frac=1)
  X 				 				                = np.array(dataframe.drop(['179'], axis = 1))
  y 				 				                = np.array(dataframe['179'])

  
  X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.20, random_state = 42)
  categoricalTrainY 				        = to_categorical(y_train, num_classes=None)
  categoricalTestY  				        = to_categorical(y_test, num_classes=None)
  embedding_vector_length           = 179
  model                             = Sequential()
  model.add(Embedding(top_words, embedding_vector_length, input_length=179))
  #LSTM 1
  model.add(LSTM(4, return_sequences = True, stateful=False, kernel_initializer='random_uniform', activation='relu', dropout = 0.20, recurrent_dropout = 0.20, inner_activation='hard_sigmoid', bias_regularizer=L1L2(l1=0.01, l2=0.01)))
  # Batch normalization
  model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
  #LSTM 2
  model.add(LSTM(4, return_sequences = True, stateful=False, kernel_initializer='random_uniform', activation='relu', dropout = 0.20, recurrent_dropout = 0.20, inner_activation='hard_sigmoid', bias_regularizer=L1L2(l1=0.01, l2=0.01)))
  # Batch normalization
  model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
  #LSTM 3
  model.add(LSTM(4, stateful=False, kernel_initializer='random_uniform', activation='softmax', dropout = 0.20 , recurrent_dropout = 0.20, inner_activation='hard_sigmoid', bias_regularizer=L1L2(l1=0.01, l2=0.01)))
  # Batch normalizaion
  model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))  
  #model.add(Dropout(0.2))
  model.add(Dense(4, activation='softmax', kernel_initializer = 'random_uniform'))
  print (model.summary())

  opt   = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
  #opt2  = optimizers.Adam(lr=0.00001)
  opt3  = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
  opt4  = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
  sgd   = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

  adam  = optimizers.Adam(lr=0.01, beta_1=0.91, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
  # do not use logcosh as loss value
  # better use binary_crossentropy with adam
  # we used: mean_squared_error with sgd
  # mean_absolute_error

  model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'], weighted_metrics = ['accuracy'])
  #print(model.summary())

  earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80,  verbose=1, mode='min')
  callbacks_list = [earlystop]

  model.fit(X_train, categoricalTrainY, validation_split = 0.5, epochs = n_epochs, batch_size = batch_size, callbacks = callbacks_list)
  # Final evaluation of the model
  scores = model.evaluate(X_test, categoricalTestY)
  print("Accuracy: %.2f%%" % (scores[1]*100))
  print(scores)
  print(scores[1]*100)


'''
  seed = 7
  models = []
  models.append(('LR', LogisticRegression()))
  models.append(('DT', DecisionTreeClassifier()))
  models.append(('GNB', GaussianNB()))
  models.append(('RF', RandomForestClassifier()))
  models.append(('SVM', svm.SVC()))
  models.append(('Clustering', KMeans(n_clusters=10, random_state=0)))

  # evaluate each model in turn
  results = []
  names = []
  scoring = 'accuracy'
  for name, model in models:
      kfold = model_selection.KFold(n_splits=10, random_state = seed)
      cv_results = model_selection.cross_val_score(model, X, y, cv = kfold, scoring = scoring)
      results.append(cv_results)
      names.append(name)
      msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
      print(msg)
'''



if __name__ =="__main__":
  main()

