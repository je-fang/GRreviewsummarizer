from __future__ import print_function
import numpy as np

class RBM:
  
  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = False

    # initializing weight matrix
    np_rng = np.random.RandomState(1234)

    self.weights = np.asarray(np_rng.uniform(
			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
      high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
      size=(num_visible, num_hidden)))


    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 1000, learning_rate = 0.1):

    num_examples = data.shape[0]

    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):      
      # positive CD phase
      pos_hidden_activations = np.dot(data, self.weights)      
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      pos_associations = np.dot(data.T, pos_hidden_probs)

      # negative CD phase
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # update weights
      self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      if self.debug_print:
        print("Epoch %s: error is %s" % (epoch, error))

  #get hidden given visible
  def run_visible(self, data):
    
    num_examples = data.shape[0]
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
    # bias units of 1 inserted into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    hidden_activations = np.dot(data, self.weights)
    hidden_probs = self._logistic(hidden_activations)
    hidden_states[:,:] = hidden_probs
  
    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states
    

  def run_hidden(self, data):

    num_examples = data.shape[0]
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # bias units of 1 inserted into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    visible_activations = np.dot(data, self.weights.T)
    visible_probs = self._logistic(visible_activations)
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states
    
        
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))
