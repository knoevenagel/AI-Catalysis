import pandas as pd
import numpy as np
import random
random.seed(1)
np.random.seed(1)

from autogluon.tabular import  TabularPredictor

def find_next_feature_list(data,feature_list,label,train_model,train_test_rate, train_data_index = None,test_data_index = None,model_path = None):
    

    excluded_model_types = ['XGB','GBM','CAT','RF','XT','LR','KNN','NN_MXNET','NN_TORCH','FASTAI']
    excluded_model_types.remove(train_model)
    #['root_mean_squared_error', 'mean_squared_error','mean_absolute_error', 'median_absolute_error', 'r2']
    features = feature_list
    record_mae = []
    for feature in features:
        if train_data_index == None or train_data_index == None:
            train_data = data.sample(n = int(train_test_rate*len(data)))
            test_data = data.drop(train_data.index)
            train_data_index = list(train_data.index)
            test_data_index = list(test_data.index)
        data_full = data.drop(feature,axis = 1)
        train_data = data_full.loc[train_data_index]
        test_data = data_full.loc[test_data_index]
        predictor = TabularPredictor(label=label,eval_metric='mean_absolute_error',path = model_path)\
            .fit(train_data,excluded_model_types = excluded_model_types)  # Fit models for 120s
        leaderboard = predictor.leaderboard(test_data,extra_metrics = ['mean_absolute_error'],silent=True)
        record_mae = record_mae + [leaderboard['mean_absolute_error'][0]]
    record_mae = [-x for x in record_mae]

    min_mae = np.min(record_mae)
    feature_to_drop = features[np.argmin(record_mae)]
    feature_list.remove(feature_to_drop)
    return (min_mae,feature_to_drop)

def greedy_delete(data,feature_list,label,train_model,train_test_rate,mae_path = 'mae.csv',feature_path = 'feature.csv',train_data_index = None,test_data_index = None,model_path = None):
    """ feature_list is the list of featrues to drop
        label is the name of target column
        train_model is the model you want to train ,it includes ['XGB','GBM','CAT','RF','XT','LR','KNN','NN_MXNET','NN_TORCH','FASTAI']
        train_test_rate is the rate of #train_set/#test_set
        mae_path,feature_path are the path to the recorded mae and featrue drop list
        train_data_index ,test_data_index are index of train_data and test_data ,if one of them is None ,then each iteration the function will randomly choose train_data and test_data
        model_path is the path to the models 
        """
    
    
    record_mae = []
    record_feature = []
    data_iter = data
    #caculation the full data training performance
    excluded_model_types = ['XGB','GBM','CAT','RF','XT','LR','KNN','NN_MXNET','NN_TORCH','FASTAI']
    excluded_model_types.remove(train_model)
    if test_data_index == None or test_data_index == None:
        train_data = data.sample(n = int(train_test_rate*len(data)))
        test_data = data.drop(train_data.index)
        train_data_index = list(train_data.index)
        test_data_index = list(test_data.index)
    train_data = data_iter.loc[train_data_index]
    test_data = data_iter.loc[test_data_index]
    predictor = TabularPredictor(label=label,eval_metric='mean_absolute_error',path = model_path)\
            .fit(train_data,excluded_model_types = excluded_model_types)  # Fit models for 120s
    leaderboard = predictor.leaderboard(test_data,extra_metrics = ['mean_absolute_error'],silent=True)
    record_mae = record_mae + [-leaderboard['mean_absolute_error'][0]]


    for i in range(len(feature_list)):
        (min_mae,feature_drop) = find_next_feature_list(data_iter,feature_list,label,train_model,train_test_rate,train_data_index,test_data_index)
        record_mae = record_mae + [min_mae]
        record_feature = record_feature + [feature_drop]
        data_iter = data_iter.drop(feature_drop,axis=1)
        record_mae_pd = pd.DataFrame(record_mae)
        record_feature_pd = pd.DataFrame(record_feature)
        record_mae_pd.to_csv(mae_path)
        record_feature_pd.to_csv(feature_path) 


    
