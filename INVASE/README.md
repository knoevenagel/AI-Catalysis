# README

## 如何使用sparse-invase 算法

sparse-invase算法旨在对于数据集进行instance层面上的变量选择。

以target为MAE为例，INVASE算法可以通过main_invase.py中的mae函数进行引用:mae(actor_h_dim,critic_h_dim,dim,iteration,lamda1,lamda2,learning_rate,lr2,n_layer,penality,batch_size,df)

actor_h_dim为hidden state dimensions for actor.

critic_h_dim为'hidden state dimensions for critic'

iteration为the number of iteration.

dim为'the number of features'

lambda1、lambda2分别为我们sparse-invase中的超参数。

learning_rate为critic和baseline的学习率。

lr2为actor的学习率。

batch_size为'the number of samples in mini batch'.

penality为我们对于神经网络中每层的正则化系数。

df 是我们的数据集，需要注意的是我们的target的columns名需设定为'y'.



