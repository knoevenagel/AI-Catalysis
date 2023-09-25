"""Instance-wise Variable Selection (INVASE) module - with baseline

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
           "IINVASE: Instance-wise Variable Selection using Neural Networks," 
           International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@gmail.com
"""

# Necessary packages
from csv import writer
from keras.layers import Input, Dense, Multiply
from keras.layers import BatchNormalization, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K

import math
import tensorflow as tf
import numpy as np
import pandas as pd
tf.compat.v1.disable_eager_execution()

from utils import bernoulli_sampling

#smooth L1 loss
import tensorflow as tf
import numpy as np


class invase():
  """INVASE class.
  
  Attributes:
    - x_train: training features
    - y_train: training labels
    - model_type: invase or invase_minus
    - model_parameters:
      - actor_h_dim: hidden state dimensions for actor
      - critic_h_dim: hidden state dimensions for critic
      - n_layer: the number of layers
      - batch_size: the number of samples in mini batch
      - iteration: the number of iterations
      - activation: activation function of models
      - learning_rate: learning rate of model training
      - lamda: hyper-parameter of INVASE
  """
    
  def __init__(self, x_train, y_train, model_type, model_parameters):
    self.penality = model_parameters['penality']
    self.lamda = model_parameters['lamda']
    self.actor_h_dim = model_parameters['actor_h_dim']
    self.critic_h_dim = model_parameters['critic_h_dim']
    self.n_layer = model_parameters['n_layer']
    self.batch_size = model_parameters['batch_size']
    self.iteration = model_parameters['iteration']
    self.activation = model_parameters['activation']
    self.learning_rate = model_parameters['learning_rate']
    self.lr2=model_parameters['lr2']
    self.lamda2 = model_parameters['lamda2']
    self.dim = x_train.shape[1] 
    self.label_dim = y_train.shape[1]
    
    self.model_type = model_type

    optimizer = Adam(self.learning_rate)
    optimizer2=Adam(self.lr2)
    # Build and compile critic
    self.critic = self.build_critic()
    self.critic.compile(loss='mae', 
                        optimizer=optimizer, metrics=['mae', 'mae'])

   
    # Build and compile the actor
    self.actor = self.build_actor()
    self.actor.compile(loss=self.actor_loss, optimizer=optimizer2)

    if self.model_type == 'invase':
      # Build and compile the baseline
      self.baseline = self.build_baseline()
      self.baseline.compile(loss='mae', 
                            optimizer=optimizer, metrics=['mae', 'mae'])
       

  def actor_loss(self, y_true, y_pred):
    """Custom loss for the actor.
    
    Args:
      - y_true:
        - actor_out: actor output after sampling
        - critic_out: critic output 
        - baseline_out: baseline output (only for invase)
      - y_pred: output of the actor network
        
    Returns:
      - loss: actor loss
    """           

    # Actor output
    actor_out = y_true[:, :self.dim]
    # Critic output
    critic_out = y_true[:, self.dim:(self.dim+self.label_dim)]
    
    if self.model_type == 'invase':
      # Baseline output
      baseline_out = \
      y_true[:, (self.dim+self.label_dim):(self.dim+2*self.label_dim)]
      # Ground truth label
      y_out = y_true[:, (self.dim+2*self.label_dim):]
        
#     elif self.model_type == 'invase_minus':
#       # Ground truth label
#       y_out = y_true[:, (self.dim+self.label_dim):]         
        
    # Critic loss
    #critic：select and output
    
    #actor:select prob
    #用mse代替
    
    critic_loss = -tf.reduce_sum((y_out - critic_out)**2, axis = 1)  

    if self.model_type == 'invase':        
      # Baseline loss
      baseline_loss = -tf.reduce_sum((y_out -baseline_out)**2,axis = 1)  
      # Reward
      Reward = -(critic_loss - baseline_loss)
        #baseline-critic
#     elif self.model_type == 'invase_minus':
#       Reward = -critic_loss
    
    
    # Policy gradient loss computation.
    KL=tf.reduce_sum(actor_out * K.log(y_pred + 1e-8) + (1-actor_out) * K.log(1-y_pred + 1e-8), axis = 1)
    custom_actor_loss = -Reward*KL+ self.lamda2*tf.reduce_sum(y_pred*(1-y_pred), axis = 1)+abs(Reward)*self.lamda* tf.reduce_mean(y_pred, axis = 1)
#     custom_actor_loss = -Reward*tf.reduce_sum(y_pred*actor_out, axis = 1)+ self.lamda2*tf.reduce_sum(y_pred*(1-y_pred), axis = 1)+self.lamda* tf.reduce_mean(y_pred, axis = 1   
    
#     custom_actor_loss = tf.reduce_sum((critic_out -baseline_out)**2,axis = 1) - self.lamda2 * tf.reduce_mean(y_pred*(1-y_pred), axis = 1)
    
    #self.lamda * tf.reduce_mean(1-K.exp(-self.lamda2*y_pred), axis = 1)
    
    # self.lamda * tf.reduce_mean(y_pred, axis = 1) + self.lamda2 * tf.reduce_mean(y_pred*K.log(y_pred+1e-8), axis = 1)

    
    # custom actor loss
    custom_actor_loss = tf.reduce_mean(custom_actor_loss)#*Reward
    #reward
    return custom_actor_loss


  def build_actor(self):
    """Build actor.
    
    Use feature as the input and output selection probability
    """
    actor_model = Sequential()    
    actor_model.add(Dense(self.actor_h_dim, activation=self.activation, 
                          kernel_regularizer=regularizers.l2(self.penality), 
                          input_dim = self.dim))
    for _ in range(self.n_layer - 2):
      actor_model.add(Dense(self.actor_h_dim, activation=self.activation, 
                            kernel_regularizer=regularizers.l2(self.penality)))
      
    actor_model.add(Dense(self.dim, activation = 'sigmoid', 
                          kernel_regularizer=regularizers.l2(self.penality)))

    feature = Input(shape=(self.dim,), dtype='float32')

    selection_probability = actor_model(feature)
    
    actor_model.summary()

    return Model(feature, selection_probability)


  def build_critic (self):
    """Build critic.
        
    Use selected feature as the input and predict labels
    """
    critic_model = Sequential()
                
    critic_model.add(Dense(self.critic_h_dim, activation=self.activation, 
                           kernel_regularizer=regularizers.l2(self.penality), 
                           input_dim = self.dim)) 
#     critic_model.add(Dropout(0.4))     
    critic_model.add(BatchNormalization())
    for _ in range(self.n_layer - 2):
      critic_model.add(Dense(self.critic_h_dim, activation=self.activation, 
                             kernel_regularizer=regularizers.l2(self.penality)))
#       critic_model.add(Dropout(0.4))
      critic_model.add(BatchNormalization())

    critic_model.add(Dense(self.label_dim, kernel_regularizer=regularizers.l2(self.penality)))
        
    ## Inputs
    # Features
    feature = Input(shape=(self.dim,), dtype='float32')
    # Binary selection
    selection = Input(shape=(self.dim,), dtype='float32')         
        
    # Element-wise multiplication
    critic_model_input = Multiply()([feature, selection])
    y_hat = critic_model(critic_model_input)

    return Model([feature, selection], y_hat)
        

  def build_baseline (self):
    """Build baseline.
    
    Use the feature as the input and predict labels
    """
    baseline_model = Sequential()
                
    baseline_model.add(Dense(self.critic_h_dim, activation=self.activation, 
                           kernel_regularizer=regularizers.l2(self.penality), 
                           input_dim = self.dim)) 
#     baseline_model.add(Dropout(0.5))                       
    baseline_model.add(BatchNormalization())
    for _ in range(self.n_layer - 2):
      baseline_model.add(Dense(self.critic_h_dim, activation=self.activation, 
                               kernel_regularizer=regularizers.l2(self.penality)))
#       baseline_model.add(Dropout(0.5))
      baseline_model.add(BatchNormalization())

    baseline_model.add(Dense(self.label_dim,  
                             kernel_regularizer=regularizers.l2(self.penality)))
            
    # Input
    feature = Input(shape=(self.dim,), dtype='float32')       
    # Output        
    y_hat = baseline_model(feature)

    return Model(feature, y_hat)


  def train(self, x_train, y_train):
    """Train INVASE.
    
    Args:
      - x_train: training features
      - y_train: training labels
    """

    for iter_idx in range(self.iteration):

      ## Train critic
      # Select a random batch of samples
      idx = np.random.randint(0, x_train.shape[0], self.batch_size)
      x_batch = x_train[idx,:]
      y_batch = y_train[idx,:]

      # Generate a batch of selection probability
      selection_probability = self.actor.predict(x_batch) 
                   
      # Sampling the features based on the selection_probability


      selection = bernoulli_sampling(selection_probability)  

      # Critic loss
      critic_loss = self.critic.train_on_batch([x_batch, selection], y_batch)                        
      # Critic output
      critic_out = self.critic.predict([x_batch, selection])
         
      # Baseline output
      if self.model_type == 'invase':   
        # Baseline loss
        baseline_loss = self.baseline.train_on_batch(x_batch, y_batch)                        
        # Baseline output
        baseline_out = self.baseline.predict(x_batch)
            
      ## Train actor
      # Use multiple things as the y_true: 
      # - selection, critic_out, baseline_out, and ground truth (y_batch)
      if self.model_type == 'invase':
        y_batch_final = np.concatenate((selection, 
                                        np.asarray(critic_out), 
                                        np.asarray(baseline_out), 
                                        y_batch), axis = 1)
      elif self.model_type == 'invase_minus':
        y_batch_final = np.concatenate((selection, 
                                        np.asarray(critic_out), 
                                        y_batch), axis = 1)
        
      # Train the actor
      actor_loss = self.actor.train_on_batch(x_batch, y_batch_final)

      if self.model_type == 'invase':
        # Print the progress
        dialog = 'Iterations: ' + str(iter_idx) + \
                 ', critic mae loss: ' + str(np.round(critic_loss[1],3)) + \
                 ', baseline mae loss: ' + str(np.round(baseline_loss[1],3)) + \
                 ', actor loss: ' + str(np.round(actor_loss,3))
      elif self.model_type == 'invase_minus':
        # Print the progress
        dialog = 'Iterations: ' + str(iter_idx) + \
                 ', critic loss: ' + str(critic_loss[1]) + \
                 ', actor loss(mse+sparse penality): ' + str(np.round(actor_loss,4))

      if iter_idx % 100 == 0:
        pd.DataFrame(np.asarray(selection_probability)).to_csv('.\prob\g'+str(iter_idx/100)+'.csv')
        print(dialog)
    
      
  def importance_score(self, x):
    """Return featuer importance score.
    
    Args:
      - x: feature
      
    Returns:
      - feature_importance: instance-wise feature importance for x
    """        
    feature_importance = self.actor.predict(x)        
    return np.asarray(feature_importance)
     

  def predict(self, x):
    """Predict outcomes.

    Args:
      - x: feature
      
    Returns:
      - y_hat: predictions    
    """        
    # Generate a batch of selection probability
    selection_probability = self.actor.predict(x)            
    # Sampling the features based on the selection_probability
    selection = bernoulli_sampling(selection_probability)   
    print('selectionshape:',selection.shape)
    # Prediction 
    y_hat = self.critic.predict([x, selection])
     
    return np.asarray(y_hat)

