#encoding=utf-8

import tensorflow as tf
import numpy as np



class MLP(object):
	"""
	Example usage might look something like this.

	data = {
		X_train:... ,
		y_train:... ,
		X_val:... ,
		y_val:... 
	}
	hidden_dim = [100,100]
	model = MLP(data,input_dim, hidden_dim, output_dim,
				learning_rate:1e-3,
				dropout_prob: 1.0,
				l2_strength: 0.0,
				num_epochs: 10,
				batch_size: 200,
				print_every: 100,
				verbose:False)
	model.train()
	model.predict(X_test)	
	"""

	def __init__(self,data,input_dim, hidden_dim, output_dim, **kwargs):
		self.data = data
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.learning_rate = kwargs.pop('learning_rate', 1e-3)
		self.weight_scale = kwargs.pop('weight_scale', 1.0)
		self.dropout_prob = kwargs.pop('dropout_prob', 0.0)
		self.l2_strength = kwargs.pop('l2_strength', 0.0)
		self.num_epochs = kwargs.pop('num_epochs', 50)
		self.batch_size = kwargs.pop('batch_size', 200)
		self.print_every = kwargs.pop('print_every', 100)
		self.verbose = kwargs.pop('verbose', False)
		if len(kwargs) > 0:
			extra = ','.join('%s ' % (k for k in kwargs.keys()))
			raise ValueError('Unrecognition arguments %s' % extra)

		self._reset()
		self._initGraph()

	def _reset(self):
		"""
		Init variables used inside. Do not call it manualy!
		"""
		self.epoch = 0
		self.loss_history = []
		self.train_acc_history = []
		self.val_acc_history = []
		self.test_acc_history = []
		self.params = {}

	def _initGraph(self):
		"""
		Init tensorflow graph. Do not call this manualy!
		"""

		# Set inputX and inputY in placeholder.
		Xs = tf.placeholder(tf.float32, (None,self.input_dim))
		ys = tf.placeholder(tf.float32, (None,self.output_dim))

		# Add hidden layers
		layer_dim = np.hstack([self.input_dim, self.hidden_dim])
		output = Xs
		l2_loss = 0.0
		for idx in xrange(len(layer_dim) - 1):
			W = tf.Variable(tf.random_normal([layer_dim[idx], layer_dim[idx+1]]) * self.weight_scale, dtype=tf.float32)
			b = tf.Variable(tf.zeros(layer_dim[idx+1]), dtype=tf.float32)
			output = tf.add(tf.matmul(output, W), b)
			output = tf.nn.relu(output)
			l2_loss += tf.nn.l2_loss(W)
			if self.dropout_prob != 0:
				output = tf.nn.dropout(output, self.dropout_prob)
				print ('dropout')

		# Add output layers
		W = tf.Variable(tf.random_normal([layer_dim[-1], self.output_dim]) * self.weight_scale, dtype=tf.float32)
		b = tf.Variable(tf.zeros(self.output_dim), dtype=tf.float32)
		logits = tf.add(tf.matmul(output, W), b)
		l2_loss += tf.nn.l2_loss(W)

		# Loss
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ys))
		loss += self.l2_strength * l2_loss

		# Train
		train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

		# Accuracy
		prediction = tf.nn.softmax(logits)
		correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.session = tf.Session()
		self.params = {
			'loss': loss,
			'train_step': train_step,
			'prediction': prediction,
			'accuracy': accuracy,
			'Xs': Xs,
			'ys': ys
		}

	def train(self):
		sess = self.session

		# 1. Import step
		sess.run(tf.global_variables_initializer())

		# 2. Get iteration nums
		num_trains = self.data['X_train'].shape[0]
		iterations_per_epoch = max(num_trains/self.batch_size, 1)
		num_iterations = self.num_epochs * iterations_per_epoch

		# 3. Train loop
		for t in xrange(num_iterations):

			# TODO: I know this is  ugly, but I don't know any other way!
			loss, train_step, prediction, accuracy, Xs, ys = self.params['loss'],  self.params['train_step'],self.params['prediction'],  self.params['accuracy'],self.params['Xs'],  self.params['ys']

			# Minibatch 
			# Method 1
			batch_mask = np.random.choice(num_trains, self.batch_size)
			X_batch = self.data['X_train'][batch_mask]
			y_batch = self.data['y_train'][batch_mask]

			# offset = (t * self.batch_size) % (num_trains - self.batch_size)
			# X_batch = self.data['X_train'][offset:(offset + self.batch_size), :]
			# y_batch = self.data['y_train'][offset:(offset + self.batch_size), :]

			# Train
			_,l = sess.run([train_step, loss], feed_dict={ Xs:X_batch, ys: y_batch })

			# Record
			self.loss_history.append(l)
			# TODO: 优化这部分代码，还没想好怎么打印 epoch还没用
			if self.verbose and (t % self.print_every == 0):
				train_acc = sess.run(accuracy, feed_dict={Xs:X_batch, ys:y_batch })
				val_acc = sess.run(accuracy, feed_dict={Xs: self.data['X_val'], ys: self.data['y_val']})
				self.train_acc_history.append(train_acc)
				self.val_acc_history.append(val_acc)
				if 'X_test' in self.data.keys():
					test_acc = sess.run(accuracy, feed_dict={Xs: self.data['X_test'], ys:self.data['y_test']})
					self.test_acc_history.append(test_acc)
					print('(Iteration %d / %d) train acc: %.2f%%; val_acc: %.2f%%; test_acc: %.2f%%' % (t, num_iterations, train_acc*100, val_acc*100, test_acc*100))
				else:
					print('(Iteration %d / %d) train acc: %.2f%%; val_acc: %.2f%%' % (t, num_iterations, train_acc*100, val_acc*100))


	def predict(self, X, y=None):
		"""
		If y is None, return score predictions, else return accuracy.
		"""
		accuracy,prediction, Xs, ys = self.params['accuracy'], self.params['prediction'], self.params['Xs'], self.params['ys']
		if y == None:
			return self.session.run(prediction, feed_dict={Xs:X, ys:np.ones((X.shape[0], self.output_dim))})
		else:
			return self.session.run(accuracy, feed_dict={Xs:X, ys:y})

