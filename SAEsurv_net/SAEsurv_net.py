import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score,KFold,StratifiedKFold,GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel,SelectKBest,VarianceThreshold
from sklearn import metrics
import time
import math
import lifelines
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import regularizers,initializers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam
import matplotlib.pyplot as plt
import json

#########################################################general used############################################################

def read_csv(file_name):
    '''
    Parameters
        ----------
        file_name: str
    Returns
        ----------
        data: dataframe,which doesn't include ['Unnamed: 0']
        '''
    data = pd.read_csv(file_name)
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'], axis = 1, inplace = True)
        
    return data

def get_optimizer(params):

    optimizer_dict = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamax': Adamax,
        'nadam': Nadam
    }
    if not optimizer_dict.get(params['optimizer_name']):
        LOG.exception('No optimizer!')
        exit(1)

    optimizer_name = optimizer_dict[params['optimizer_name']]

    return optimizer_name

def plot_loss(model):
    '''
    Parameters
        ----------
        model: model name which has been fit
    Returns
        ----------
        plot
        '''
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title(str(model),' loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    
######################################################only used in preprocess#######################################################

def zscore(train, test, col):
    '''
    Parameters
        ----------
        train,test:dataframe
        col:str
    Returns
        ----------
        none
        '''
    test[col] = (test[col] - train[col].mean())/train[col].std()
    train[col] = StandardScaler().fit_transform(np.array(train[col]).reshape(-1, 1))
    
def minmax(train, test, col):
    '''
    Parameters
        ----------
        train,test:dataframe
        col:str
    Returns
        ----------
        none
        '''
    test[col] = (test[col] - train[col].min())/(train[col].max()-train[col].min())
    train[col] = MinMaxScaler().fit_transform(np.array(train[col]).reshape(-1, 1))
    
def test_remove(train, test, threshold, data_type):#log rank test
    '''
    Parameters
        ----------
        train,test:dataframe
        threshold:float
        data_type:str
    Returns
        ----------
        train,test:dataframe
        '''
    input_col = list(filter(lambda x: x.split('_')[0]==data_type, train.columns))
    n_before = len(input_col)
    data = train[input_col]
    stay_name = list(filter(lambda x: multivariate_logrank_test(train['OS.time'], train[x], train['OS'],weightings=None).p_value<threshold,input_col))

    for col in stay_name:
        input_col.remove(col)
    train.drop(input_col, axis=1, inplace = True)
    test.drop(input_col, axis=1, inplace = True)
    print('original',data_type,'features：',n_before,
          'delected by log rank test：',len(input_col),
          'left：',len(stay_name))
    return train,test

def var_remove(train, test, data_type):
    '''
    Parameters
        ----------
        train,test:dataframe
        data_type:str
    Returns
        ----------
        train,test:dataframe
        '''
    input_col = list(filter(lambda x: x.split('_')[0]==data_type,train.columns))
    n_before = len(input_col)
    selector_var = VarianceThreshold().fit(pd.DataFrame(train[input_col]))
    stay_name = selector_var.get_feature_names_out()
    for col in list(stay_name):
        input_col.remove(col)
    train.drop(input_col, axis=1, inplace = True)
    test.drop(input_col, axis=1, inplace = True)
    print('original',data_type,'features：',n_before,
          'delected by 0 variance：',len(input_col),
          'left：',len(stay_name))
    return train,test

def dupli_remove(train, test, data_type):
    '''
    Parameters
        ----------
        train,test:dataframe
        input_col:list
        data_type:str
    Returns
        ----------
        train,test:dataframe
        '''
    input_col = list(filter(lambda x: x.split('_')[0]==data_type, train.columns))
    n_before = len(input_col)
    train_removed = pd.DataFrame(train[input_col]).T.drop_duplicates(keep='first', inplace=False)
    train_removed = pd.DataFrame(train_removed).T
    for col in train_removed.columns:
        input_col.remove(col)

    train.drop(input_col, axis=1, inplace = True)
    test.drop(input_col, axis=1, inplace = True)
    print('original',data_type,'features：',n_before,
          'delected by duplicate：',len(input_col),
          'left：',len(train_removed.columns))
    
    return train,test

######################################################only used in autoencoder#######################################################

def load_data_ae1(x,info_col,data_type):
    '''
    Parameters
        ----------
        x:str,file name of data
        info:list of str #['submitter_id', 'OS', 'OS.time']
        data_type:str #'gene','cnv'
    Returns
        -------
        X:array,features fed to ae1
        info:dataframe
        '''    
    data = read_csv(file_name = x)
    use_col = list(filter(lambda x: x.split('_')[0]==data_type, data.columns))
    X = data[use_col]
    
    X = X.values.astype(float)
    info = pd.DataFrame(data[info_col])
    
    return X,info

def load_data_ae2(x,info_col):
    '''
    Parameters
        ----------
        x:str,file name of data
        info_col:list of str #['submitter_id', 'OS', 'OS.time']
    Returns
        -------
        X:array,features going to be transformed
        info:dataframe
        '''  
    data = read_csv(x)
    X = data.drop(info_col, axis = 1, inplace = False).values
    X.astype(float)
    info = pd.DataFrame(data[info_col])
    
    return X,info

def encoder(params, X_train, encoder_name, hidden_layer):
    '''
    Parameters
        ----------
        params:dict
        X_train:array
        encoder_name:str
        hidden_layer:int #1,3
    Returns
        -------
        encoder:model
        '''  
    if hidden_layer == 1:
        encoder = keras.models.Sequential([
            keras.layers.Dropout(
                params.get('dropout_rate'), noise_shape = None, seed = 123,
                name = 'dropout_'+encoder_name),
            keras.layers.Dense(
                params.get('bottleneck_num'),
                activation = params.get('activation_name'),
                kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
                kernel_initializer=initializers.glorot_normal(seed=123),#xavier
                bias_initializer='zeros',
                name = 'dense_'+encoder_name)
        ])
    
    if hidden_layer == 3:
        encoder = keras.models.Sequential([
            keras.layers.Dropout(
                params.get('dropout_rate1'), noise_shape = None, seed = 123,
                name = 'dropout1_'+encoder_name),
            keras.layers.Dense(
                params.get('hidden_num'),
                activation = params.get('activation_name'),
                kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
                kernel_initializer=initializers.glorot_normal(seed=123),#xavier
                bias_initializer='zeros',
                name = 'dense1_'+encoder_name),
            keras.layers.Dropout(
                params.get('dropout_rate2'), noise_shape = None, seed = 123,
                name = 'dropout2_'+encoder_name),
            keras.layers.Dense(
                params.get('bottleneck_num'),
                activation = params.get('activation_name'),
                kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
                kernel_initializer=initializers.glorot_normal(seed=123),#xavier
                bias_initializer='zeros',
                name = 'dense2_'+encoder_name)
        ])
        
    return encoder

def decoder(params, X_train, hidden_layer):
    '''
    Parameters
        ----------
        params:dict
        X_train:array
        hidden_layer:int #1,3
    Returns
        -------
        encoder:model
        '''  
    if hidden_layer == 1:
        decoder = keras.models.Sequential([
            keras.layers.Dropout(params.get('dropout_rate'), noise_shape = None, seed = 123),
            keras.layers.Dense(
                X_train.shape[1], activation = params.get('activation_name'),
                kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
                kernel_initializer=initializers.glorot_normal(seed=123),#xavier
                bias_initializer='zeros')
        ])
        
    if hidden_layer == 3:
        decoder = keras.models.Sequential([
            keras.layers.Dropout(params.get('dropout_rate2'), noise_shape = None, seed = 123),
            keras.layers.Dense(
                params.get('hidden_num'), 
                activation = params.get('activation_name'),
                kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
                kernel_initializer=initializers.glorot_normal(seed=123),#xavier
                bias_initializer='zeros'),
            keras.layers.Dropout(params.get('dropout_rate1'), noise_shape = None, seed = 123),
            keras.layers.Dense(
                X_train.shape[1],
                activation = params.get('activation_name'),
                kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
                kernel_initializer=initializers.glorot_normal(seed=123),#xavier
                bias_initializer='zeros')
      ])    
        
    return decoder

######################################################only used in risknet#######################################################
def load_data_risknet(sae_train,merge_train):
    '''
    Parameters
        ----------
        sae_train: str,file name of data
        merge_train: str,file name of data
    Returns
        -------
        data_train_x: dataframe,includes gene+cnv features transformed by sae and clinical features
        data_train_e, data_train_t: dataframe
        '''
    sae_train = read_csv(sae_train)
    merge_train = read_csv(merge_train)
    
    cli_col = list(filter(lambda x: x.split('_')[0]=='cli', merge_train.columns))
    cli_col.append('submitter_id')
    data_train_x_cli = merge_train[cli_col]
     
    data_train = pd.merge(data_train_x_cli,sae_train,how="inner",on='submitter_id')
    data_train.sort_values(by='OS.time',inplace=True,ascending=True)
    
    data_train_x = data_train.drop(columns = {'submitter_id','OS.time','OS'}, axis=1, inplace = False)
    data_train_t = data_train['OS.time']
    data_train_e = data_train['OS'].astype(float)

    return data_train_x, data_train_e, data_train_t

def neg_log_partial_likelihood(e_true, pred):
    pred, e_true= tf.squeeze(pred), tf.squeeze(e_true)
    e_true = tf.cast(e_true,tf.float32)
    n_obs = tf.reduce_sum(e_true)
    
    hazard = tf.math.exp(pred)
    cost = tf.math.negative(tf.divide(tf.reduce_sum(tf.math.multiply(
            e_true,tf.subtract(pred,tf.math.log(tf.cumsum(hazard,reverse = True)))), axis=-1),n_obs))
    
    return cost

def model_risknet(data_train_x,hidden_layer,params):
    '''
    Parameters
        ----------
        data_train_x: str,file name of data
        hidden_layer: int
        params: dict
    Returns
        -------
        data_train_x: dataframe,includes gene+cnv features transformed by sae and clinical features
        data_train_e, data_train_t: dataframe
        '''
    inputs = keras.Input(shape=(data_train_x.shape[1],),name = 'inputs_concatcli')
    
    if hidden_layer == 0:#cox ph model(linear combination)
        risknet = Dense(
            units = 1,
            activation = None,
            kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
            kernel_initializer=initializers.glorot_normal(seed=123),#xavier
            bias_initializer='zeros')(inputs)
        
    if hidden_layer == 1:
        risknet = Dropout(params.get('dropout_rate'), noise_shape = None, seed = 123, name = 'dropout_ds')(inputs)
        risknet = Dense(
            params.get('hiddennueron_num'),
            activation = params.get('activation_name'),
            kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
            kernel_initializer=initializers.glorot_normal(seed=123),#xavier
            bias_initializer='zeros',
            name = 'dense1_ds'
        )(risknet)
        risknet = Dense(units = 1, name = 'dense2_ds')(risknet)

    if hidden_layer == 2:
        risknet = Dropout(params.get('dropout_rate1'), noise_shape = None, seed = 123, name = 'dropout1_ds')(inputs)
        risknet = Dense(
            units = params.get('hiddennueron_num1'),
            activation = params.get('activation_name'),
            kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
            kernel_initializer=initializers.glorot_normal(seed=123),#xavier
            bias_initializer='zeros',
            name = 'dense1_ds'
        )(risknet)
        risknet = Dropout(params.get('dropout_rate2'), noise_shape = None, seed = 123, name = 'dropout2_ds')(risknet)
        risknet = Dense(units = params.get('hiddennueron_num2'),
                         activation = params.get('activation_name'),
                         kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
                         kernel_initializer=initializers.glorot_normal(seed=123),#xavier
                         bias_initializer='zeros',
                         name = 'dense2_ds')(risknet)
        risknet = Dense(units = 1, name = 'dense3_ds')(risknet)
    
    risknet = keras.Model(
        inputs=inputs,
        outputs=risknet
    )
    
    risknet.compile(optimizer = get_optimizer(params)(lr = params.get('learning_rate')),
                     loss = neg_log_partial_likelihood)
    
    return risknet