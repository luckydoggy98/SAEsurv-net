import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import regularizers,initializers
from tensorflow.keras.optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam
import nni
import logging
import argparse
import tempfile
import os 
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from SAEsurv_net import read_csv,get_optimizer,load_data_ae1,load_data_ae2

LOG = logging.getLogger('gbmae_AutoML')

class SendMetrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        
    def on_epoch_end(self, epoch, logs={}):
        LOG.debug(logs)
        nni.report_intermediate_result(logs['val_loss'])  
        self.losses.append(logs['val_loss'])
        
    def on_train_end(self, logs={}):
        last_loss = self.losses[-1]
        nni.report_final_result(last_loss)
        
def model(params, X_train, hidden_layer):
    '''
    Parameters
        ----------
        params:dict
        X_train:array
        hidden_layer:int
    Returns
        -------
        AutoEncoder
    '''
    inputs = keras.Input(shape=(X_train.shape[1],))
    if hidden_layer == 1:
        dropout1 = Dropout(params.get('dropout_rate'), noise_shape = None, seed = 123)(inputs)
        encode = Dense(
            params.get('bottleneck_num'),
            activation = params.get('activation_name'),
            kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')))(dropout1)
        
        dropout2 = Dropout(params.get('dropout_rate'), noise_shape = None, seed = 123)(encode)
        decode = Dense(
            X_train.shape[1],
            activation = params.get('activation_name'),
            kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')))(dropout2)
    
    if hidden_layer == 3:
        dropout1 = Dropout(params.get('dropout_rate1'), noise_shape = None, seed = 123)(inputs)
        hidden1 = Dense(
            params.get('hidden_num'),
            activation = params.get('activation_name'),
            kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')))(dropout1)
        dropout2 = Dropout(params.get('dropout_rate2'), noise_shape = None, seed = 123)(hidden1)
        encode = Dense(
            params.get('bottleneck_num'),
            activation = params.get('activation_name'),
            kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')))(dropout2)
        
        dropout3 = Dropout(params.get('dropout_rate2'), noise_shape = None, seed = 123)(encode)
        hidden2 = Dense(
            params.get('hidden_num'),
            activation = params.get('activation_name'),
            kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')))(dropout3)
        dropout4 = Dropout(params.get('dropout_rate1'), noise_shape = None, seed = 123)(hidden2)
        decode = Dense(
            X_train.shape[1],
            activation = params.get('activation_name'),
            kernel_regularizer=regularizers.l1_l2(l1=params.get('regularizer_l1'), l2=params.get('regularizer_l2')))(dropout4)

    AutoEncoder = keras.Model(
        inputs=inputs,
        outputs=decode
    )
    AutoEncoder.compile(optimizer = get_optimizer(params)(lr = params.get('learning_rate')), loss = 'mean_squared_error')

    return AutoEncoder

def run(X_train, model, params):
    '''
    Parameters
        ----------
        X_train:array
        model:a compiled nn model
        params:dict
    Returns
        -------
        AutoEncoder
    '''
    model.fit(X_train, X_train, epochs = params.get('epoch_num'), 
              batch_size = params.get('batch_num'),
              validation_split = 0.2,
              callbacks = [SendMetrics()],
              verbose = 0)

def get_params(hidden_layer):
    '''
    Parameters
        ----------
        hidden_layer:int
    Returns
        -------
        params:dict
    '''
    if hidden_layer == 1:
        params = {
            'dropout_rate': 0.1,
            'bottleneck_num': 200,
            'optimizer_name': 'adamax',
            'activation_name': 'relu',
            'regularizer_l1': 0.01,
            'regularizer_l2': 0.01,
            'learning_rate': 0.0001,
            'epoch_num': 20,
            'batch_num': 30
            }
        
    if hidden_layer == 3:
        params = {
            'dropout_rate1': 0.1,
            'dropout_rate2': 0.1,
            'hidden_num':250,
            'bottleneck_num': 200,
            'optimizer_name': 'adamax',
            'activation_name': 'relu',
            'regularizer_l1': 0.01,
            'regularizer_l2': 0.01,
            'learning_rate': 0.0001,
            'epoch_num': 20,
            'batch_num': 30
            }    
    return params

if __name__ == '__main__':
    
    ae_type = 'multi' #'gene', 'cnv', 'multi'
    
    X_train_gene,info_train = load_data_ae1(x = 'GBM_merge_selected_train.csv',
                                            info_col = ['submitter_id', 'OS', 'OS.time'],
                                            data_type = 'gene')
    X_train_cnv,info_train = load_data_ae1(x = 'GBM_merge_selected_train.csv',
                                           info_col = ['submitter_id', 'OS', 'OS.time'],
                                           data_type = 'cnv')
    X_train_multi,info_train = load_data_ae2(x = 'GBM_hidden1_train.csv',
                                             info_col = ['submitter_id', 'OS', 'OS.time'])
    params = json.load(open('params.json'))
    
    if ae_type == 'gene':
        X_train = X_train_gene
        hidden_layer = params.get('hidden_layer1')
    if ae_type == 'cnv':
        X_train = X_train_cnv
        hidden_layer = params.get('hidden_layer1')
    if ae_type == 'multi':
        X_train = X_train_multi
        hidden_layer = params.get('hidden_layer2')
    
    try:
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_params(hidden_layer)
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = model(params = PARAMS, X_train = X_train, hidden_layer = hidden_layer)
        run(X_train, model, PARAMS)
        print(LOG)
        
    except Exception as exception:
        LOG.exception(exception)
        raise
#nnictl create --config config_ae.yml