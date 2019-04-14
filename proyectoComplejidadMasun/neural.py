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

#Bidirectional training
from keras.layers import Bidirectional 
# Preprocessing for the input data.
from sklearn import preprocessing

# for conv1d
from keras import layers

# for the f1_score
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import metrics


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
  #
  #
  # In here you need to normalize the input vector. For
  #
  #


  X                                 = preprocessing.normalize(X, norm='l2')
  y 				 				                = np.array(dataframe['179'])

  
  X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.05, random_state = 42)
  categoricalTrainY 				        = to_categorical(y_train, num_classes=None)
  categoricalTestY  				        = to_categorical(y_test, num_classes=None)
  embedding_vector_length           = 179
  model                             = Sequential()
  model.add(Embedding(top_words, embedding_vector_length, input_length=179))

  # define LSTM with Bidirectional layer at the beginning.


  #LSTM 1
  # adding the Conv1 decrease the accuracy.
  model.add(layers.Conv1D(22, 5, activation='relu'))
  model.add(Bidirectional(LSTM(20, return_sequences = True, stateful=False, kernel_initializer='random_uniform', activation='softmax', dropout = 0.20, recurrent_dropout = 0.20, inner_activation='hard_sigmoid', bias_regularizer=L1L2(l1=0.01, l2=0.01))))
  # Batch normalization
  model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
  #LSTM 2
  model.add(Bidirectional(LSTM(20, return_sequences = True, stateful=False, kernel_initializer='random_uniform', activation='softmax', dropout = 0.20, recurrent_dropout = 0.20, inner_activation='hard_sigmoid', bias_regularizer=L1L2(l1=0.01, l2=0.01))))
  # Batch normalization
  model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
  #LSTM 3
  model.add(Bidirectional(LSTM(20, stateful=False, kernel_initializer='random_uniform', activation='softmax', dropout = 0.20 , recurrent_dropout = 0.20, inner_activation='hard_sigmoid', bias_regularizer=L1L2(l1=0.01, l2=0.01))))
  # Batch normalizaion
  model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))  
  #model.add(Dropout(0.2))
  model.add(Dense(4, activation='softmax', kernel_initializer = 'random_uniform'))

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

  model.compile(loss = 'binary_crossentropy', optimizer = opt4, metrics = ['accuracy'], weighted_metrics = ['accuracy'])
  #print(model.summary())

  earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=80,  verbose=1, mode='min')
  callbacks_list = [earlystop]

  model.fit(X_train, categoricalTrainY, validation_split = 0.5, epochs = n_epochs, batch_size = batch_size, callbacks = callbacks_list)
  # Final evaluation of the model
  scores = model.evaluate(X_test, categoricalTestY)
  print("Accuracy: %.2f%%" % (scores[1]*100))
  print(scores)
  print(scores[1]*100)

'''
  print("**************** More scores **************")
  categoricalTestY  = np.array(categoricalTestY)
  y_pred_class      = np.array(model.predict(X_test))
  # calculate accuracy
  print("(0) Recomputation of test accuracy")
  print(metrics.accuracy_score(categoricalTestY, y_pred_class))
  
  #Null accuracy: accuracy that could be achieved by always predicting the most frequent class
  
  # Examine the class distribution of the testing set (using a Pandas Series method)
  print("(1) Distribution testing set: ")
  print(categoricalTestY.value_counts().head(1) / len(categoricalTestY))
  # Comparing the true and predicted response values
  print("(2) True and predicted response values")
  print('True:', categoricalTestY.value_countss[0:25])
  print('False:', y_pred_class[0:25])
'''


  #categ1oricalTestY = list(categoricalTestY)
  #y_pred           = list(y_pred)
  #print("the lists!")
  #print(categoricalTestY)
  #print(y_pred)
  #print(f1_score(categoricalTestY, y_pred, average = "macro"))
  #print(precision_score(categoricalTestY, y_pred, average = "macro"))
  #print(recall_score(categoricalTestY, y_pred, average = "macro"))    
  #print(f1_score(categoricalTestY, y_pred, average = 'macro') )


if __name__ =="__main__":
  main()

