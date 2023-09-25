# README

## How to use the greedy deletion function

The flow chart of greedy deletion function can be found in Figure S2b. It sequentially removes least impactful feature from the feature set to generate a lean feature set. Autogluon package is needed for this function.

The function has following form:
greedy_delete(data,feature_list,label,train_model,train_test_rate,mae_path = 'mae.csv',feature_path = 'feature.csv',train_data_index = None,test_data_index = None,model_path = None)

in which:
data is the baseline feature set for deletion (F_base in Figure S2b);

feature_list is the deletion sequence (optional);

label is the range of deletion (F_i in Figure S2b);

train_model is the type of ML algorithm，which is subjected to Autogluon package. For Autogluon 0.8.2, it can be ['XGB','GBM','CAT','RF','XT','LR','KNN','NN_MXNET','NN_TORCH','FASTAI'];

train_test_rate is the ratio of training set, Autogluon automatically set a portion of the training set as validation set, the ratio of validation set can be set in Autogluon; 

mae_path/feature_path are the stroage path of the mae and feature list returned by the function, results are stored in the same path of the .py file if not specified;

model_path is the storage path of autogluon model, models are stored at the same path of the .py file if not specified;

train_data_index are test_data_index can be used to fix the training/testing dataset (instead of letting Autogluon randomly select from data)

## How to use the random deletion function

The flow chart of random deletion function can be found in Figure S2a. It generates a random deletion sequence, than deletion each feature in the sequence. Autogluon package is needed for this function.

The function has following form:
random_choose_deletion(data,feature_list,label,train_model,repeat_times,train_test_rate ,each_point_times = 5,mae_path = 'mae.csv',var_path = 'var.csv',feature_path = 'feature.csv',train_data_index = None,test_data_index = None,model_path = None)

in which:
data is the baseline feature set for deletion (F_base in Figure S2a);

feature_list is the deletion sequence (optional);

label is the range of deletion (F_i in Figure S2a);

train_model is the type of ML algorithm，which is subjected to Autogluon package. For Autogluon 0.8.2, it can be ['XGB','GBM','CAT','RF','XT','LR','KNN','NN_MXNET','NN_TORCH','FASTAI'];

repeat_times is the number of experiments (i.e. the number of random sequence used for deletion);

train_test_rate is the ratio of training set, Autogluon automatically set a portion of the training set as validation set, the ratio of validation set can be set in Autogluon; 

each_point_times is the number of models trained for each feature set;

mae_path/feature_path are the stroage path of the mae and feature list returned by the function, results are stored in the same path of the .py file if not specified;

model_path is the storage path of autogluon model, models are stored at the same path of the .py file if not specified;

train_data_index are test_data_index can be used to fix the training/testing dataset (instead of letting Autogluon randomly select from data).
