import pandas as pd
import numpy as np
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler
import nni
import logging
import math
import lifelines
from lifelines.utils import concordance_index
from tensorflow import keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import regularizers,initializers
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import json
from SAEsurv_net import read_csv,load_data_risknet,get_optimizer,neg_log_partial_likelihood

LOG = logging.getLogger('gbmds_AutoML')

def get_params(hidden_layer):
    
    '''
    Parameters
        ----------
        hidden_layer: int,1 or 2
    Returns
        -------
        params: dict,default parameters
    '''
    
    if hidden_layer == 0:
        params = {
            'optimizer_name': 'nadam',
            'regularizer_l1': 0.00001,
            'regularizer_l2': 0,
            'learning_rate': 0.0001,
            'epoch_num': 10,
            'batch_num': 50
        }
        
    if hidden_layer == 1:
        params = {
            'dropout_rate': 0.3307,
            'hiddennueron_num': 60,
            'optimizer_name': 'nadam',
            'activation_name': 'sigmoid',
            'regularizer_l1': 0.00001,
            'regularizer_l2': 0,
            'learning_rate': 0.0001,
            'epoch_num': 10,
            'batch_num': 50
        }
        
    if hidden_layer == 2:
        params = {
            'dropout_rate1': 0.3307,
            'dropout_rate2': 0.3307,
            'hiddennueron_num1': 60,
            'hiddennueron_num2': 60,
            'optimizer_name': 'nadam',
            'activation_name': 'sigmoid',
            'regularizer_l1': 0.00001,
            'regularizer_l2': 0,
            'learning_rate': 0.001,
            'epoch_num': 10,
            'batch_num': 50
        }
    
    return params

class SendMetrics(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []
        
    def on_epoch_end(self, epoch, logs={}):
        LOG.debug(logs) 
        self.losses.append(logs['val_loss'])
        nni.report_intermediate_result(logs['val_loss']) 
    
def model(data_train_x,hidden_layer,params):
    
    inputs = keras.Input(shape=(data_train_x.shape[1],))
    
    if hidden_layer == 0:
        risknet = Dense(
            units = 1,
            activation = None,
            kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
            kernel_initializer=initializers.glorot_normal(seed=123),#xavier
            bias_initializer='zeros')(inputs)
    
    if hidden_layer == 1:
        risknet = Dropout(params.get('dropout_rate'), noise_shape = None, seed = 123)(inputs)
        risknet = Dense(
            params.get('hiddennueron_num'),
            activation = params.get('activation_name'),
            kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
            kernel_initializer=initializers.glorot_normal(seed=123),#xavier
            bias_initializer='zeros'
        )(risknet)
        risknet = Dense(units = 1)(risknet)

    if hidden_layer == 2:
        risknet = Dropout(params.get('dropout_rate1'), noise_shape = None, seed = 123)(inputs)
        risknet = Dense(
            units = params.get('hiddennueron_num1'),
            activation = params.get('activation_name'),
            kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
            kernel_initializer=initializers.glorot_normal(seed=123),#xavier
            bias_initializer='zeros'
        )(risknet)
        risknet = Dropout(params.get('dropout_rate2'), noise_shape = None, seed = 123)(risknet)
        risknet = Dense(units = params.get('hiddennueron_num2'),
                         activation = params.get('activation_name'),
                         kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')),
                         kernel_initializer=initializers.glorot_normal(seed=123),#xavier
                         bias_initializer='zeros'
                        )(risknet)
        risknet = Dense(units = 1)(risknet)
    
    risknet = keras.Model(
        inputs = inputs,
        output s= risknet
    )
    
    risknet.compile(optimizer = get_optimizer(params)(lr = params.get('learning_rate')),
                     loss = neg_log_partial_likelihood)
    
    return risknet

def run(data_train_x,data_train_e, data_train_t, 
        model, params):
    
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=5,restore_best_weights=True)
    
    score_val = []
    for train_index, test_index in KFold(n_splits=5,shuffle=True).split(data_train_x):
        train_index.sort()
        test_index.sort()
        print('train index',train_index)
        print('test index',test_index)
        history = model.fit(
            x = data_train_x.iloc[train_index].values,
            y = data_train_e.iloc[train_index],
            epochs = params.get('epoch_num'),
            batch_size = params.get('batch_num'),
            callbacks = [SendMetrics()]+[earlystop_callback],#+[keras.callbacks.TerminateOnNaN()],
            verbose = 1,
            validation_data = (data_train_x.iloc[test_index].values,
                               data_train_e.iloc[test_index])
        )
        model.summary()
        predict = model.predict(data_train_x.iloc[test_index].values)
    
        print(predict)
        score = concordance_index(event_times = data_train_t.iloc[test_index],
                              predicted_scores = -np.exp(np.squeeze(predict)),#输入是hazard时要加负号，输入是时间时不加负号
                              event_observed = data_train_e.iloc[test_index])
        score_val.append(score)
    score_mean = np.mean(score_val)
    LOG.debug('ci score: %s', score_mean)
    nni.report_final_result(score_mean) 
    
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('deepsurv loss')
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper right')
    #plt.show()
    
if __name__ == '__main__':

    model_type = 'cnv'#'gene','cnv','multi'
    
    if model_type == 'multi':
        data_train_x, data_train_e, data_train_t = load_data_ds('GBM_merge_after_sae_train.csv','GBM_merge_selected_train.csv')
    if model_type == 'gene':
        data_train_x, data_train_e, data_train_t = load_data_ds('GBM_hidden1_gene_train.csv','GBM_merge_selected_train.csv')
    if model_type == 'cnv':
        data_train_x, data_train_e, data_train_t = load_data_ds('GBM_hidden1_cnv_train.csv','GBM_merge_selected_train.csv')
        
    params = json.load(open('params.json'))
    hidden_layer = params.get('hidden_layer_ds')#0,1,2. 0 means linear combination(coxph)
    
    try:
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_params(hidden_layer)
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        deepsurv = model(data_train_x,params=PARAMS,hidden_layer=hidden_layer)
        run(data_train_x,
            data_train_e, data_train_t, 
            deepsurv, PARAMS)
    except Exception as exception:
        LOG.exception(exception)
        raise
 #nnictl create --config config_rn.yml