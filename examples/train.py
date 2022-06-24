import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score,KFold,StratifiedKFold,GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel,mutual_info_regression,SelectKBest,VarianceThreshold
from sklearn import metrics,preprocessing
import time
import math
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import regularizers,initializers
from tensorflow.keras.optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam
from tensorflow import keras
import nni
import argparse
import tempfile
import os 
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#if ModuleNotFoundError: No module named 'SAEsurv_net', then:
import sys
sys.path.append('D:\study\paper\code\SAEsurv-net')
from SAEsurv_net.SAEsurv_net import *
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
 
logger.addHandler(handler)
logger.info("Start print log")


##get hyper-parameters
params = json.load(open('params.json'))
params_gene = params.get('params_gene')
params_cnv = params.get('params_cnv')
params_multi = params.get('params_multi')
params_ds = params.get('params_ds')

hidden_layer1 = params.get('hidden_layer1')
hidden_layer2 = params.get('hidden_layer2')
hidden_layer_ds = params.get('hidden_layer_ds')

##read data including gene,cnv,cli without train/test-split nor preprocess
GBM_merge = read_csv('GBM_merge_crude.csv')

final_result=[]

##start to train and test
for iteration in np.arange(1,30,1):
    '''
    train/test split
    preprocess:imputation,normalization,feature selection
    '''
    time_start = time.time()

    logger.info('start experiment:'+str(iteration))

    x_train, x_test, t_train, t_test = train_test_split(GBM_merge.drop(columns={'OS.time'},inplace=False),
                                                        GBM_merge['OS.time'],
                                                        test_size = 0.2,
                                                        stratify = GBM_merge['OS'],
                                                        random_state = iteration)
    GBM_merge_train = pd.concat([x_train, t_train], axis=1)
    GBM_merge_test = pd.concat([x_test, t_test], axis=1)
    
    ##imputation
    GBM_merge_train.loc[GBM_merge_train.loc[GBM_merge_train['cli_radiation.therapy'].isnull()==True].index.tolist(), 
                        'cli_radiation.therapy'] = GBM_merge_train['cli_radiation.therapy'].mode()[0]
    GBM_merge_test.loc[GBM_merge_test.loc[GBM_merge_test['cli_radiation.therapy'].isnull()==True].index.tolist(), 
                       'cli_radiation.therapy'] = GBM_merge_train['cli_radiation.therapy'].mode()[0]
    
    ##clinical:
    ##normalization(onehot,z-score)
    ##feature selection：log rank test
    cli_col = []
    for col in GBM_merge_train.columns:
        if col.split('_')[0]=='cli':
            cli_col.append(col)
            if GBM_merge_train[col].dtype == 'float':
                zscore(train = GBM_merge_train, test = GBM_merge_test, col = col)
            if GBM_merge_train[col].dtype == 'object':
                GBM_merge_train[col] = pd.get_dummies(GBM_merge_train[col], prefix = [col], drop_first = True)
                GBM_merge_test[col] = pd.get_dummies(GBM_merge_test[col], prefix = [col], drop_first = True)
    GBM_merge_train, GBM_merge_test = test_remove(train = GBM_merge_train,
                                                  test = GBM_merge_test, 
                                                  data_type = 'cli',
                                                  threshold = 0.05)
    ##cnv:
    ##feature selection：var,dupli,log rank test
    GBM_merge_train, GBM_merge_test = var_remove(train = GBM_merge_train,
                                                 test = GBM_merge_test,
                                                 data_type = 'cnv')
    GBM_merge_train, GBM_merge_test = dupli_remove(train = GBM_merge_train,
                                                   test = GBM_merge_test,
                                                   data_type = 'cnv')
    GBM_merge_train, GBM_merge_test = test_remove(train = GBM_merge_train,
                                                  test = GBM_merge_test, 
                                                  data_type = 'cnv',
                                                  threshold = 0.05)
    ##gene:
    ##normalization(z-score,sparse and re-transform)
    ##feature selection：var,dupli,log rank test
    gene_col = list(filter(lambda x: x.split('_')[0]=='gene', GBM_merge_train.columns))
    for col in gene_col:
        zscore(train = GBM_merge_train, test = GBM_merge_test, col = col)

    GBM_merge_train, GBM_merge_test = var_remove(train = GBM_merge_train,
                                                 test = GBM_merge_test,
                                                 data_type = 'gene')
    GBM_merge_train, GBM_merge_test = dupli_remove(train = GBM_merge_train,
                                                   test = GBM_merge_test,
                                                   data_type = 'gene')
    ##sparse+log rank test
    gene_col = list(filter(lambda x: x.split('_')[0] == 'gene', GBM_merge_train.columns))
    gene_train, gene_test = np.array(GBM_merge_train[gene_col]), np.array(GBM_merge_test[gene_col])
    gene_train[gene_train <= -1],gene_test[gene_test <= -1] = -1,-1
    gene_train[abs(gene_train) <1],gene_test[gene_test <1] = 0,0
    gene_train[gene_train >= 1],gene_test[gene_test >= 1] = 1,1
    raw_train, raw_test = GBM_merge_train[gene_col], GBM_merge_test[gene_col]
    GBM_merge_train[gene_col], GBM_merge_test[gene_col] = gene_train, gene_test
     
    GBM_merge_train, GBM_merge_test = test_remove(train = GBM_merge_train,
                                                  test = GBM_merge_test,
                                                  data_type = 'gene',
                                                  threshold = 0.05)
    ##re-transform
    gene_col = list(filter(lambda x: x.split('_')[0]=='gene', GBM_merge_train.columns))
    for col in gene_col:
        GBM_merge_train[col], GBM_merge_test[col] = raw_train[col],raw_test[col]
    ##save data
    GBM_merge_train.to_csv(str(iteration)+'GBM_merge_selected_train.csv')
    GBM_merge_test.to_csv(str(iteration)+'GBM_merge_selected_test.csv')
    
    time_end = time.time()
    time_sum = time_end - time_start
    print('time of preprocess：',time_sum)
    logger.info('feature num after preprocess:'+str(len(GBM_merge_train.columns)-3))
    
    '''
    autoencoder: gene and cnv respectively,multi
    '''
    time_start = time.time()
    
    ##load data
    x_gene_train,info_train = load_data_ae1(x = str(iteration)+'GBM_merge_selected_train.csv',
                                            info_col = ['submitter_id', 'OS', 'OS.time'],
                                            data_type = 'gene')
    x_cnv_train,info_train = load_data_ae1(x = str(iteration)+'GBM_merge_selected_train.csv',
                                           info_col = ['submitter_id', 'OS', 'OS.time'],
                                           data_type = 'cnv')
    x_gene_test,info_test = load_data_ae1(x = str(iteration)+'GBM_merge_selected_test.csv',
                                            info_col = ['submitter_id', 'OS', 'OS.time'],
                                            data_type = 'gene')
    x_cnv_test,info_test = load_data_ae1(x = str(iteration)+'GBM_merge_selected_test.csv',
                                           info_col = ['submitter_id', 'OS', 'OS.time'],
                                           data_type = 'cnv')
    
    ##build encoder,decoder for gene,cnv respectively
    encoder_gene = encoder(params = params_gene, X_train = x_gene_train, hidden_layer = hidden_layer1, encoder_name = 'gene')
    decoder_gene = decoder(params = params_gene, X_train = x_gene_train, hidden_layer = hidden_layer1)
    encoder_cnv = encoder(params = params_cnv, X_train = x_cnv_train, hidden_layer = hidden_layer1, encoder_name = 'cnv')
    decoder_cnv = decoder(params = params_cnv, X_train = x_cnv_train, hidden_layer = hidden_layer1)
    
    AutoEncoder_gene = keras.models.Sequential([
        encoder_gene,
        decoder_gene
    ])
    AutoEncoder_cnv = keras.models.Sequential([
        encoder_cnv,
        decoder_cnv
    ])

    AutoEncoder_gene.compile(optimizer = get_optimizer(params_gene)(lr = params_gene.get('learning_rate')),
                             loss = 'mean_squared_error')
    AutoEncoder_cnv.compile(optimizer = get_optimizer(params_cnv)(lr = params_cnv.get('learning_rate')),
                             loss = 'mean_squared_error')
    
    ae_gene = AutoEncoder_gene.fit(x_gene_train, x_gene_train,
                                   epochs = params_gene.get('epoch_num'),
                                   batch_size = params_gene.get('batch_num'),
                                   validation_split = 0.2,
                                   verbose = 0)
    ae_cnv = AutoEncoder_cnv.fit(x_cnv_train, x_cnv_train,
                                 epochs = params_cnv.get('epoch_num'),
                                 batch_size = params_cnv.get('batch_num'),
                                 validation_split = 0.2,
                                 verbose = 0)
    
    ##save network weights,transformed data of encoders:
    ##2 .h5 contained weights of gene_encoder,cnv_encoder respectively,
    ##2 .csv contained gene+cnv,
    ##4 .csv contained gene,cnv respectively
    encoder_gene.save_weights(str(iteration)+'ae_hidden1_gene.h5')
    encoder_cnv.save_weights(str(iteration)+'ae_hidden1_cnv.h5')
    x_ae_train = pd.concat([pd.DataFrame(encoder_gene.predict(x_gene_train)),pd.DataFrame(encoder_cnv.predict(x_cnv_train))],axis = 1)
    x_ae_test = pd.concat([pd.DataFrame(encoder_gene.predict(x_gene_test)),pd.DataFrame(encoder_cnv.predict(x_cnv_test))],axis = 1)
    pd.concat([info_train, pd.DataFrame(x_ae_train)],axis = 1).to_csv(str(iteration)+'GBM_hidden1_train.csv')
    pd.concat([info_test, pd.DataFrame(x_ae_test)],axis = 1).to_csv(str(iteration)+'GBM_hidden1_test.csv')
    pd.concat([info_train, pd.DataFrame(encoder_gene.predict(x_gene_train))],axis = 1).to_csv(str(iteration)+'GBM_hidden1_gene_train.csv')
    pd.concat([info_test, pd.DataFrame(encoder_gene.predict(x_gene_test))],axis = 1).to_csv(str(iteration)+'GBM_hidden1_gene_test.csv')
    pd.concat([info_train, pd.DataFrame(encoder_cnv.predict(x_cnv_train))],axis = 1).to_csv(str(iteration)+'GBM_hidden1_cnv_train.csv')
    pd.concat([info_test, pd.DataFrame(encoder_cnv.predict(x_cnv_test))],axis = 1).to_csv(str(iteration)+'GBM_hidden1_cnv_test.csv')
    
    ##load data
    x_multi_train,info_train = load_data_ae2(x = str(iteration)+'GBM_hidden1_train.csv',
                                             info_col = ['submitter_id', 'OS', 'OS.time'])
    x_multi_test,info_test = load_data_ae2(x = str(iteration)+'GBM_hidden1_test.csv',
                                           info_col = ['submitter_id', 'OS', 'OS.time'])
    
    ##build multi-encoder,decoder
    encoder_multi = encoder(params = params_multi, X_train = x_multi_train, hidden_layer = hidden_layer2, encoder_name = 'multi')
    decoder_multi = decoder(params = params_multi, X_train = x_multi_train, hidden_layer = hidden_layer2)

    AutoEncoder_multi = keras.models.Sequential([
        encoder_multi,
        decoder_multi
    ])

    AutoEncoder_multi.compile(optimizer = get_optimizer(params_multi)(lr = params_multi.get('learning_rate')),
                              loss = 'mean_squared_error')

    ae_multi = AutoEncoder_multi.fit(x_multi_train, x_multi_train,
                                     epochs = params_multi.get('epoch_num'),
                                     batch_size = params_multi.get('batch_num'),
                                     validation_split = 0.2,
                                     verbose = 0)
    
    ##save network weights,transformed data of encoders:
    ##1 .h5 contained weights of multi_encoder,
    ##2 .csv contained gene+cnv
    encoder_multi.save_weights(str(iteration)+'ae_hidden2.h5')
    pd.concat([info_train, pd.DataFrame(encoder_multi.predict(x_multi_train))],axis = 1).to_csv(str(iteration)+'GBM_merge_after_sae_train.csv')
    pd.concat([info_test, pd.DataFrame(encoder_multi.predict(x_multi_test))],axis = 1).to_csv(str(iteration)+'GBM_merge_after_sae_test.csv')

    time_end = time.time()
    time_sum = time_end - time_start
    print('time of ae：',time_sum)
    
    '''
    train and test risknet
    '''
    time_start = time.time()
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=5,restore_best_weights=True)
    
    data_train_x, data_train_e, data_train_t = load_data_risknet(str(iteration)+'GBM_merge_after_sae_train.csv',
                                                            str(iteration)+'GBM_merge_selected_train.csv')
    data_test_x, data_test_e, data_test_t = load_data_risknet(str(iteration)+'GBM_merge_after_sae_test.csv',
                                                         str(iteration)+'GBM_merge_selected_test.csv')    
    risknet = model_risknet(data_train_x,params=params_ds,hidden_layer=hidden_layer_ds)

    train_index, test_index=train_test_split(range(0,data_train_x.shape[0],1),
                                             test_size=0.2, 
                                             random_state=None, 
                                             shuffle=True, 
                                             stratify=data_train_e)
    train_index.sort()
    test_index.sort()
    K.set_learning_phase(1)
    history = risknet.fit(
        x = {'inputs_concatcli':data_train_x.iloc[train_index]},
        y = data_train_e.iloc[train_index],
        epochs = params_ds.get('epoch_num'),
        batch_size = params_ds.get('batch_num'),
        callbacks = [earlystop_callback],#+[keras.callbacks.TerminateOnNaN()],
        verbose = 0,
        validation_data = (data_train_x.iloc[test_index].values,
                           data_train_e.iloc[test_index])
        )

    K.set_learning_phase(0)
    predict = risknet.predict(data_train_x.iloc[test_index].values)
    #print(predict)
    score = concordance_index(event_times = data_train_t.iloc[test_index],
                              predicted_scores = -np.exp(np.squeeze(predict)),#输入是hazard时要加负号，输入是时间时不加负号
                              event_observed = data_train_e.iloc[test_index])
    print('risknet has val cindex',score)                                         
    ##save weights
    risknet.save_weights(str(iteration)+'ds.h5')
    ##test cindex
    K.set_learning_phase(0)#预测模式
    predict = risknet.predict(data_test_x)
    ds_score = concordance_index(event_times = data_test_t,
                                 predicted_scores = -np.exp(np.squeeze(predict)),
                                 event_observed = data_test_e)
    #print('ds has test cindex',ds_score)
    logger.info('risknet has val cindex:'+str(np.mean(risknet_score_val))+'test cindex'+str(risknet_score))
    
    time_end = time.time()
    time_sum = time_end - time_start
    print('time of risknet：',time_sum)