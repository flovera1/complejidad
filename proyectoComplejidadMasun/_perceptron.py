import numpy as np
class Perceptron(object):
	#The constructor of our class.
	def __init__(self, learningRate=0.01, n_iter=50, random_state=1):
		self.learningRate = learningRate
		self.n_iter = n_iter
		self.random_state = random_state
		self.errors_ = []

	def z(self, X):
		#np.dot(X, self.w_[1:]) + self.w_[0]
		z = np.dot(X, self.weights[1:]) + self.weights[0] 
		return z
        
	def predict(self, X):
		 #Heaviside function. Returns 1 or 0 
		return np.where(self.z(X) >= 0.0, 1, 0)

	def fitin(self, X, y):
		#for reproducing the same results
		random_generator  = np.random.RandomState(self.random_state)
		#Step 0 = Get the shape of the input vector X
		#We are adding 1 to the columns for the Bias Term
		x_rows, x_columns = X.shape
		x_columns		  = x_columns+1
		#Step 1 - Initialize all weights to 0 or a small random number  
		#weight[0] = the weight of the Bias Term
		self.weights 	   = random_generator.normal(loc=0.0, scale=0.001, size=x_columns) 
		#for how many number of training iterrations where defined
		for _ in range(self.n_iter):
			errors = 0
			for xi, y_actual in zip(X, y):
				#create a prediction for the given sample xi
				y_predicted = self.predict(xi)
				#calculte the delta
				delta = self.learningRate*(y_actual - y_predicted)
				#update all the weights but the bias
				self.weights[1:] += delta * xi
				#for the bias delta*1 = delta
				self.weights[0] += delta
				#if there is an error, increase the error count for the batch
				errors += int(delta != 0.0)
			self.errors_.append(errors) 

