# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split

import argparse
import numpy as np
import pandas as pd

from data_generation import generate_dataset
from invase import invase
from utils import feature_performance_metric, prediction_performance_metric   


def main (args,df):
  """Main function for INVASE.
  
  Args:
    - df: data, df['y'] is our target. 
    - data_type: synthetic data type (syn1 to syn6)
    - train_no: the number of samples for training set
    - train_no: the number of samples for testing set
    - dim: the number of features
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
    
  Returns:
    - performance:
      - mean_tpr: mean value of true positive rate
      - std_tpr: standard deviation of true positive rate
      - mean_fdr: mean value of false discovery rate
      - std_fdr: standard deviation of false discovery rate
      - auc: area under roc curve
      - apr: average precision score
      - acc: accuracy
  """
  
  # Generate dataset
#   x_train, y_train, g_train = generate_dataset (n = args.train_no, 
#                                                 dim = args.dim, 
#                                                 data_type = args.data_type, 
#                                                 seed = 0)
  
#   x_test, y_test, g_test = generate_dataset (n = args.test_no,
#                                              dim = args.dim, 
#                                              data_type = args.data_type, 
#                                              seed = 0)

  
  data = df
  train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
  y_train_tmp = np.array(train_set['y'])
  x_train = np.array(train_set.drop(['y'],axis=1))
  y_train = np.zeros([y_train_tmp.shape[0],2])
  y_train[:,0] = y_train_tmp
  
  y_train[:,1] = 1-y_train[:,0]
  y_test_tmp = np.array(test_set['y'])
  x_test = np.array(test_set.drop(['y'],axis=1))
  y_test = np.zeros([y_test_tmp.shape[0],2])
  y_test[:,0] = y_test_tmp
  y_test[:,1] = 1-y_test[:,0]
  data = np.array(data.drop(['y'],axis=1))
  print(x_train.shape)
  print(x_test.shape)


  model_parameters = {'lamda': args.lamda,
                      'lamda2': args.lamda2,
                      'actor_h_dim': args.actor_h_dim, 
                      'critic_h_dim': args.critic_h_dim,
                      'n_layer': args.n_layer,
                      'batch_size': args.batch_size,
                      'iteration': args.iteration, 
                      'activation': args.activation, 
                      'learning_rate': args.learning_rate,
                      'penality' :args.penality,
                      'lr2':args.lr2}
  
  # Train the model
  model = invase(x_train, y_train, args.model_type, model_parameters)
 
  model.train(x_train, y_train)    
  
  ## Evaluation
  # Compute importance score
  gg = model.importance_score(data)
  pd.DataFrame(np.asarray(gg)).to_csv('101_importance.csv')

  g_hat = model.importance_score(x_test)
  
  importance_score = 1.*(g_hat > 0.5)

  y_hat = model.predict(x_test)
  
  # Evaluate the performance of feature importance
#   auc, apr, acc = prediction_performance_metric(y_test, y_hat)
    
  from sklearn.metrics import mean_absolute_error

  mae=mean_absolute_error(y_test,y_hat)
   
  # Print the performance of feature importance    
  print('MAE: ' + str(np.round(mae, 3)))
#   model.save('less_model.h5')
  return mae
  
      

def mae(dim,lamda1,actor_h_dim,critic_h_dim,n_layer,learning_rate,lamda2,lr2,iteration,penality,df):
    dim=int(dim)
    actor_h_dim=int(actor_h_dim)
    critic_h_dim=int(critic_h_dim)
    n_layer=int(n_layer)
    if __name__ == '__main__':

      # Inputs for the main function
      parser = argparse.ArgumentParser()
      parser.add_argument(
          '--data_type',
          choices=['syn1','syn2','syn3','syn4','syn5','syn6'],
          default='syn4',
          type=str)
      parser.add_argument(
          '--train_no',
          help='the number of training data',
          default=3983,
          type=int)
      parser.add_argument(
          '--test_no',
          help='the number of testing data',
          default=1708,
          type=int)
      parser.add_argument(
          '--dim',
          help='the number of features',
          choices=[11,1000],
          default=140,
          type=int)
      parser.add_argument(#0.1
          '--lamda',
          help='inavse hyper-parameter lambda1',
          default=lamda1,
          type=float)
      parser.add_argument(#100
          '--actor_h_dim',
          help='hidden state dimensions for actor',
          default=actor_h_dim,
          type=int)
      parser.add_argument(#400
          '--critic_h_dim',
          help='hidden state dimensions for critic',
          default=critic_h_dim,
          type=int)
      parser.add_argument(#3
          '--n_layer',
          help='the number of layers',
          default=n_layer,
          type=int)
      parser.add_argument(
          '--batch_size',
          help='the number of samples in mini batch',
          default=300,
          type=int)
      parser.add_argument(
          '--iteration',
          help='the number of iteration',
          default=int(iteration),
          type=int)
      parser.add_argument(
          '--penality',
          help='penality',
          default=penality,
          type=int)
      parser.add_argument(
          '--activation',
          help='activation function of the networks',
          choices=['selu','relu'],
          default='selu',
          type=str)
      parser.add_argument(#0.0001
          '--learning_rate',
          help='learning rate of model training',
          default=learning_rate,
          type=float)
      parser.add_argument(
          '--model_type',
          help='inavse or invase- (without baseline)',
          choices=['invase','invase_minus'],
          default='invase',
          type=str)
      parser.add_argument(
          '--lamda2',
          help='inavse hyper-parameter lamda2',
          default=lamda2,
          type=float)
      parser.add_argument(
          '--lr2',
          help='learning rate for actor',
          default=lr2,
          type=float)

      args_in = parser.parse_args(args=[]) 

      # Call main function  
      performance = main(args_in,df=df)
      return performance