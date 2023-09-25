#README

##How to use Modifed INVASE algorithm

The modified INVASE algorithm instance-wisely distinguish important and irrelevent features for chemical problems. The detail of the modified INVASE can be found in the Method section and Supplementary Note 3&4. 
The algorithm is based on Yoon, J., Jordan, J., Schaar, M.: Invase: Instance-wise variable selection using neural networks. ICLR 2019 (2019).

You can use this algorithm by using the mae function:
mae(actor_h_dim,critic_h_dim,dim,iteration,lamda1,lamda2,learning_rate,lr2,n_layer,penality,batch_size,df)

in which:
actor_h_dim is the hidden state dimensions for actor network;

critic_h_dim is the hidden state dimensions for critic network;

iteration is the number of iteration;

dim is the number of features;

lambda1, lambda2 are hyperparameters of modified invase (see Method);

learning_rate is the learning rate of critic and baseline network;

lr2 is the learning rate of actor network;

batch_size is the number of samples in each mini batch;

penality is the normalization coefficient of each layer in the neural network; 

df is the dataset, please rename column of the training target as "y". 


