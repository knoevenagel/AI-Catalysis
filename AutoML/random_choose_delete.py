import pandas as pd
import numpy as np
import random

random.seed(1)
np.random.seed(1)

from autogluon.tabular import  TabularPredictor

def random_choose_deletion(data,feature_list,label,train_model,repeat_times,train_test_rate ,each_point_times = 5,mae_path = 'mae.csv',var_path = 'var.csv',feature_path = 'feature.csv',train_data_index = None,test_data_index = None,model_path = None):
    """ 
    feature_list is the list of featrues to drop
    label is the name of target column
    train_model is the model you want to train ,it includes ['XGB','GBM','CAT','RF','XT','LR','KNN','NN_MXNET','NN_TORCH','FASTAI']
    repeat_times is the times you want to repeat the experiment
    train_test_rate is the rate of #train_set/#test_set
    mae_path,feature_path are the path to the recorded mae and featrue drop list
    train_data_index ,test_data_index are index of train_data and test_data ,if one of them is None ,then each iteration the function will randomly choose train_data and test_data
    model_path is the path to the models 
    """
    
    features_record = []
    
    record_mae = []
    mae_even = []
    var_all = []
    excluded_model_types = ['XGB','GBM','CAT','RF','XT','LR','KNN','NN_MXNET','NN_TORCH','FASTAI']
    excluded_model_types.remove(train_model)
    data_full = data
    if train_data_index == None or test_data_index == None:
                        train_data = data_full.sample(n = int(train_test_rate*len(data_full)))
                        test_data = data_full.drop(train_data.index)
    else:
        train_data = data.loc[train_data_index]
        test_data = data.loc[test_data_index]
    predictor = TabularPredictor(label=label,eval_metric='mean_absolute_error',path = model_path).fit(train_data,excluded_model_types = excluded_model_types)  # Fit models for 120s
    leaderboard = predictor.leaderboard(test_data,extra_metrics = ['mae'],silent=True)
    mae_ori = leaderboard['mean_absolute_error'][0]
    
    for k in range(repeat_times):
        #mae contains average information
        #mae_times contains all information
        mae = [mae_ori]
        var_each = []
        for i in range(len(feature_list)):
            mae_times = []
            for j in range(each_point_times):
                data_full = data.drop(feature_list[:i],axis = 1)
                if train_data_index == None or test_data_index == None:
                    train_data = data_full.sample(n = int(train_test_rate*len(data_full)))
                    test_data = data_full.drop(train_data.index)
                else:
                    train_data = data.loc[train_data_index]
                    test_data = data.loc[test_data_index]
                predictor = TabularPredictor(label=label,eval_metric='mean_absolute_error').fit(train_data,excluded_model_types = excluded_model_types)  # Fit models for 120s
                leaderboard = predictor.leaderboard(test_data,extra_metrics = ['mae'],silent=True)
                mae_times = mae_times + [leaderboard['mean_absolute_error'][0]]
            mae = mae + [np.mean(mae_times)] 
            var = np.std(mae_times)
            var_each = var_each + [var]

        mae_even = mae_even + [mae]
        var_all = var_all + [var_each]
        features_record = features_record + [feature_list.copy()]
        random.shuffle(feature_list)
        record_mae = record_mae + [mae_even]
        
        #save datas
        #turn to DataFrame type
        mae_even_save = pd.DataFrame(mae_even)
        var_all_save = pd.DataFrame(var_all)
        features_record_save = pd.DataFrame(features_record)
        #save to csv
        mae_even_save.to_csv(mae_path)
        var_all_save.to_csv(var_path)
        features_record_save.to_csv(feature_path) 