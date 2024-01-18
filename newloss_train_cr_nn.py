# Author: Jaime Caballero
# Date: 13-11-2023

# LOAD LIBRARIES

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
#import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import tensorflow as tf
#from sklearn.model_selection import RepeatedKFold
from numpy import mean
from numpy import std
from numpy import asarray
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score as r2_score_metric
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
import datetime
#import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras import backend as K
#import scipy.io as sio
import time
import wandb
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
import math
import pickle
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# HELPER MODULES 

import models_fancy_loss as models
import data_loading_preprocessing as data

# plots 
import matplotlib.pyplot as plt

# CONFIGURATION PARAMETERS

config_params = {
        'batch_size': {'values': [10000]}, #SWEEPED

        'epochs': {'values': [10]},  #SWEEPED

        # dataset_name = 'raw_datasets/dataset_rec3_Irate_cap_1e19'
        'dataset' : {'value': 'raw_datasets/dataset_exp_master_'},

        'inputs_labels' : { 'value' : ['emi_3-2', 'emi_5-2', '728/706', '728/668']}, #Da,Dg  // Da, Db, Dg

        'inputs_prepro' : { 'value' : [3, 3, 3, 3]},

        'inputs_scaler' : { 'value' : 'StandardScaler'},

        'outputs_labels' : { 'value' : ['Te', 'ne', 'no', 'Irate', 'Rrate', 'Brec3/B3']},

        'outputs_prepro' : { 'value' : [3, 3, 3, 3, 3, 1]},

        'outputs_scaler' : { 'value' : 'MinMaxScaler'},

        'min_max_scale' : { 'value' : (0 , 10)},

        'model' : { 'values' : ['MLP_BNN_model_Denseflipout_multi_output']}, # 'MLP_NN_MCdropout' MLP_NN_simple

        'last_bayes_layer' : {'values': [100]},

        'l2_regularizer' : {'value' : 1e-8},

        # one for every input 
        'gau_noise': {'value': [0.05, 0.05, 0.05, 0.05]},

        'num_nodes' : { 'values' : [100]},

        'num_layers' : { 'values' : [5]},

        'optimizer' : { 'value' : 'adam'},

        'activation_fn' : { 'value' : 'relu'},

        'loss_fn' : { 'value' : 'mse'},

        'earlystopping_patience' : { 'value' : 4},

        'rec3_class' : { 'value' : 0.95}

     }


# EDIT CONFIG TO THE CORRECT FORMAT 
# make config_params keys get only the value
# if doing single run then convert config_params to a single value
# with this

for key in config_params.keys():
    if 'value' in config_params[key]:
        config_params[key] = config_params[key]['value']
    else:
        config_params[key] = config_params[key]['values'][0]
        print('Took first value of ', key, ' : ', config_params[key])


# getting parameters from config_params dictionary
# to make extra keys for preprocessing

inputs_labels = config_params['inputs_labels']
inputs_prepro = config_params['inputs_prepro']
inputs_scaler = StandardScaler() if config_params['inputs_scaler'] == 'StandardScaler' else MinMaxScaler()

outputs_labels = config_params['outputs_labels']
outputs_prepro = config_params['outputs_prepro']
outputs_scaler = MinMaxScaler(feature_range=config_params['min_max_scale']) \
                                    if config_params['outputs_scaler'] == 'MinMaxScaler' else StandardScaler()

# introduce the rec3_class in code 

rec3_class = config_params['rec3_class']

# ADD PREPROCESSING AND POSTPROCESSING KEYS 
# TO CONFIG_PARAMS

config_params['inputs_mean_params'],\
config_params['inputs_scale_params'],\
config_params['outputs_min_params'],\
config_params['outputs_scale_params'], _ = \
data.get_scalers_crazy_nn(inputs_labels, inputs_prepro, inputs_scaler, \
                outputs_labels, outputs_prepro, outputs_scaler, \
                data_folder=config_params['dataset'], \
                rec3_class=rec3_class)


# START A WnB RUN 

# model project or "last name"
project_name = 'experiments_11'

run = wandb.init(project=project_name, config=config_params)

# get model name 
folder_name_saved_model = run.name

# model local location 
model_location = 'models_v3/' + project_name + '/' + folder_name_saved_model 

# model_location = 'models_v2/experiment_crazy/model-18'
# [optional] use wandb.config as your config
# I prefer to use the local 
config = wandb.config

# print config 
print('###### CONFIG ######')
print(config_params)
print('####################')

# save config to model_location, create folder if it doesn't exist
# with open(model_location + '/config_params.pkl', 'wb') as f:
#     pickle.dump(config_params, f)


# LOAD DATA 

print("###### LOADING DATASET ######")
print("###### no sweep on data preprocessing ######")

# HERE SOME CRAZY THINGS WILL HAPPEN 
# I WILL INCLUDE IN THE OUTPUTS B3REC AND MODIFY IT 
# TO BE THE BINARY CLASSIFICATION 

X_train_real_unique, y_train_real, \
X_train, y_train, \
X_val, y_val, \
X_test, y_test, \
transform_input, inverse_input, \
transform_output, inverse_output = data.grab_data_preprocessing_crazy_nn(inputs_labels, inputs_prepro, inputs_scaler, \
                                                                outputs_labels, outputs_prepro, outputs_scaler, \
                                                                data_folder=config_params['dataset'], \
                                                                rec3_class=rec3_class)


# DATA SET-UP

# REGRESSION DATA FOR VALIDATION 
y_val_regression = y_val[:,:-1].copy()

# CLASSIFICATION DATA FOR VALIDATION
y_val_classification = y_val[:, -1].copy()

# in here the maximum value of y_val_classification is extracted
# and that represents passing the ratios from [0- 1] to [0 - 10] on MinMaxScaler
# this is to deduce the threshold for classification regarless of the scaler used 
# or just grabbed it from the config_params

rec3_class_max = config_params['min_max_scale'][1]

# make values 0 or 1 if they are above 9.5 or below 9.5
y_val_classification[y_val_classification < rec3_class*rec3_class_max] = 1
y_val_classification[y_val_classification >= rec3_class*rec3_class_max] = 0

# change array to integers type 
y_val_classification = y_val_classification.astype(int)

# count the number of 1s and 0s
print('number of 1s and 0s')
print(np.count_nonzero(y_val_classification == 1))
print(np.count_nonzero(y_val_classification == 0))

print("###### DATASET LOADED ######")

# DEFINE MODEL 

n_inputs, n_outputs = X_train.shape[1], y_val_regression.shape[1]

num_layers = config_params['num_layers']

num_nodes = config_params['num_nodes']

last_bayes_layer = config_params['last_bayes_layer']

####### model creation #######

from CRM_ADAS import Deuterium

def CRM_forward_model(y_pred):

    deu_crm = Deuterium()

    # Te_arr = y_pred[:, 0]
    # ne_arr = y_pred[:, 1]
    # no_arr = y_pred[:, 2]

    out_emis = np.zeros((y_pred.shape[0], 4))

    Te_ne_no_raw_arr = y_pred[:, :3]

    Te_ne_no_arr = np.power(Te_ne_no_raw_arr/np.array(config_params['outputs_scale_params'][:3], dtype = np.float32) \
                            + np.array(config_params['outputs_min_params'][:3], dtype = np.float32), \
                                np.array(outputs_prepro[:3], dtype = np.float32))

    # computing full data from CRM_ADAS module
    for idx, Te_ne_no in enumerate(Te_ne_no_arr):

        if Te_ne_no[0] >= 0.2001 and Te_ne_no[0] < 80 and Te_ne_no[1] >= 1e18 and Te_ne_no[1] < 1e20 and Te_ne_no[2] >= 1e15 and Te_ne_no[2] < 1e19:
            out_emis[idx, :2] = deu_crm.compute_emissivites(Te_ne_no, balmer_lines = [3,5])
        else:
            out_emis[idx, :2] = np.array([0.0, 0.0])
    
    return out_emis[:, 0]
    

def tf_precise_calculations(y_pred):
    return tf.py_function(CRM_forward_model, [y_pred], Tout=tf.float32)

# CUSTOM LOSS FUNCTION dummy
# this custom_loss function is used only to set it in an already train network
# for exporting, the actual loss function is the one inside the model definition

from keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer

def MLP_BNN_model_book_regression_classification(n_inputs, n_outputs, num_layers, num_nodes, \
    last_bayes_layer, act_fn, l2_reg, kernel_div, n_class, custom_loss_for_inference = True):

    # input layer 
    input_single = Input(shape=(n_inputs,))
    x = Dense(num_nodes, activation=act_fn, kernel_initializer='he_uniform', \
        kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(input_single)
    x = BatchNormalization()(x)
    for i in range(num_layers - 2):
        x = Dense(num_nodes, activation=act_fn, kernel_initializer='he_uniform', \
            kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        # x = layers.Dropout(0.1)(x)
    def relu_advanced(x):
        return K.relu(x, max_value=5.0)
    # last bayesian layer with just kernel as random variable
    # x = tfp.layers.DenseFlipout(last_bayes_layer, activation=relu_advanced, kernel_divergence_fn=kernel_div)(x)
    # last bayesian layer with kernel and bias as random variables
    x = tfp.layers.DenseFlipout(last_bayes_layer, activation='relu', kernel_divergence_fn=kernel_div, bias_prior_fn = tfp.layers.default_multivariate_normal_fn, \
     bias_divergence_fn = kernel_div)(x)

    # standard ouput layer:
    # output_regression = Dense(n_outputs, activation='linear')(x)
    # dense_flipout output layer:
    output_regression = tfp.layers.DenseFlipout(n_outputs, kernel_divergence_fn=kernel_div, activation='linear', name = 'out_regression')(x) #,\
        #bias_prior_fn = tfp.layers.default_multivariate_normal_fn, bias_divergence_fn = kernel_div)(x)
    output_classification = Dense(n_class, activation='softmax', name = 'out_classification')(x)

    model_output = concatenate([output_regression, output_classification])

    def custom_loss(y_true, y_pred):

        
        regression_loss_Te = K.abs(y_pred[:, 0] - y_true[:, 0])*(2.0*K.pow(K.ones_like(y_true[:, 0])*0.5, y_true[:, 0]) + 1) #*y_true[:, 5] + \
            #K.abs(y_pred[:, 0] - y_true[:, 0])*(1 - y_true[:, 5]) * K.random_uniform(shape=(1,), minval=0.0, maxval=5.0)

        regression_loss_ne = K.abs(y_pred[:, 1] - y_true[:, 1])*(2.0*K.pow(K.ones_like(y_true[:, 1])*0.5, y_true[:, 1]) + 1)

        # this output is ignore if it is in the non-ionized regime
        regression_loss_no = K.abs(y_pred[:, 2] - y_true[:, 2])*y_true[:, 5]*(2.0*K.pow(K.ones_like(y_true[:, 2])*0.5, y_true[:, 2]) + 1)

        # this output is ignore if it is in the non-ionized regime (ignores low values)
        regression_loss_Irate = K.abs(y_pred[:, 3] - y_true[:, 3])*y_true[:, 5]*(4.0*K.pow(K.ones_like(y_true[:, 3])*0.5, y_true[:, 3]) + 1)
        
        # this output is ignore if it is in the non-ionized regime (ignores high values)
        regression_loss_Rrate = K.abs(y_pred[:, 4] - y_true[:, 4])*y_true[:, 5]*(5.0*K.pow(K.ones_like(y_true[:, 4])*0.5, y_true[:, 4]) + 1)

        # doing MAE of losses 
        regression_loss = K.mean(regression_loss_Te +  regression_loss_ne + regression_loss_no + regression_loss_Irate + regression_loss_Rrate, axis = -1)

        # classification loss
        class_loss = K.sparse_categorical_crossentropy(y_true[:, 5], y_pred[:, 5:]) 

        # pressure loss
        physics_loss = K.mean(K.abs(2*y_pred[:, 0]*y_pred[:, 1] - 2*y_true[:, 0]*y_true[:, 1]))

        loss_make_up = K.mean(tf_precise_calculations(y_pred))

        # sum of all losses 
        return regression_loss + class_loss  + physics_loss + loss_make_up

    model = Model(inputs = input_single, outputs = model_output)

    if custom_loss_for_inference:
        model.compile(loss=custom_loss, optimizer='adam')
    else:
        model.compile(loss='mae', optimizer='adam')

    return model


if config_params['model'] == 'MLP_BNN_model_Denseflipout_multi_output':

    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) /  (X_train.shape[0] * 1.0)

    model = MLP_BNN_model_book_regression_classification(n_inputs, n_outputs, \
                                        num_layers, num_nodes, last_bayes_layer, \
                                        act_fn = config_params['activation_fn'], \
                                        l2_reg = config_params['l2_regularizer'], \
                                        kernel_div = kernel_divergence_fn,\
                                        n_class = 2)


# visualize model
print(model.summary())


##### TRANSFER LEARNING #####
# load model with "better" output 
model_path = 'models_v2/test_1_best/fancy-2-wind-12'
model_checkpoint, _ = models.load_model(model_path)

# set the weigths of the model to be trained 
# to the weights of the previously trained "better" model    
model.set_weights(model_checkpoint.get_weights())

# set learning rate to a small value
model.optimizer.lr = 1e-7

# freeze all layers except the last ones
for layer in model.layers[:-4]:
    layer.trainable = False


# early stopping callback
earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=config_params['earlystopping_patience'], verbose=0, mode='auto',
    restore_best_weights=True
    )

# custom callback, in here I am loading the r2 score on the validation set
# in real space, later something else can be added here
class CustomCallback(keras.callbacks.Callback):

    # in here it might be possible to enter test on real data

    def on_epoch_end(self, epoch, logs=None):

        #y_val = y_val_regression.copy()

        y_test_truth_realspace = data.data_decoder(y_val, outputs_prepro, inverse_output)[0]
        y_test_truth_realspace = y_test_truth_realspace[:,:-1].copy()

        iterations = 6

        # PREDICTIONS REGRESSION
        predictions = np.zeros((X_val.shape[0], y_val_regression.shape[1], iterations))
        # PREDICTIONS CLASSIFICATION
        predictions_class = np.zeros((X_val.shape[0], 1, iterations))

        # dummy array 
        current_prediction = np.zeros((X_val.shape[0], y_val_regression.shape[1] + 1))

        for i in range(iterations):

            holder = self.model.predict(X_val, batch_size = 50000, verbose = 0)
            # print(holder)
            # current_prediction[:, :-1], current_pred_class = self.model.predict(X_val, batch_size = 50000, verbose = 0)
            current_prediction[:, :-1] = holder[:, :5]
            current_pred_class = holder[:, 5:]
            
            predictions[:, :, i] = data.data_decoder(current_prediction, outputs_prepro, inverse_output)[0][:, :-1]
            # print(current_pred_class)
            now_current_pred_class = np.zeros_like(y_val_classification)
            now_current_pred_class[current_pred_class[:,-1] >= 0.5] = 1
            predictions_class[:, :, i] = now_current_pred_class.reshape(-1,1)
            # predictions_class[:, :, i] = current_pred_class.reshape(-1,1)

        y_predict_realspace = np.mean(predictions.copy(), axis = 2)
        y_predict_std_realspace = (100*np.std(predictions.copy(), axis = 2))/y_predict_realspace

        prediction_class = np.mean(predictions_class.copy(), axis = 2)
        final_prediction_class = np.zeros_like(prediction_class)
        final_prediction_class[prediction_class > 0.5] = 1

        # print(final_prediction_class.shape)

        # making nan values of no and Irate when they not have to be predicted

        y_predict_realspace[:,2][final_prediction_class[:,0] == 0] = np.nan

        y_predict_realspace[:,3][final_prediction_class[:,0] == 0] = np.nan

        y_predict_realspace[:,4][final_prediction_class[:,0] == 0] = np.nan


        print('\n---------------------------------------')

        print(' R2 scores - val set:')

        #################################

        # setting plots for visualization

        #################################

        # set figure size

        if y_train.shape[1] == 1:

            single_plot = True
            fig, ax = plt.subplots(1, y_train.shape[1], figsize = (4*y_train.shape[1], 5.20), dpi = 200)
            fig.tight_layout()
        else:
            single_plot = False
            fig, ax = plt.subplots(1, y_train.shape[1], figsize = (4*y_train.shape[1], 5.20), dpi = 200) 
                #gridspec_kw={'height_ratios': [1, 0.20]})
            # make ax[1,:] one thing 
            fig.tight_layout()

            
        
        save_plot_thing = False

        for i in range(y_val_regression.shape[1]):

            print("  " + outputs_labels[i], end= ": ")

            #if i == 0 or i ==  1 or i == 4:
            if i == 0 or i ==  1:

                r2_score = r2_score_metric(y_test_truth_realspace[:,i], y_predict_realspace[:,i])

            else:
                
                y_true_for_r2 = y_test_truth_realspace[:,i].copy()
                y_pred_for_r2 = y_predict_realspace[:,i].copy()

                non_nan_indices = ~np.isnan(y_pred_for_r2)

                y_true_no_nan = y_true_for_r2[non_nan_indices]
                y_pred_no_nan = y_pred_for_r2[non_nan_indices]

                r2_score = r2_score_metric(y_true_no_nan, y_pred_no_nan)
            
            
            print('{:.6f}'.format(r2_score), end = ' // ')

            # log r2 score to weight and biases
            wandb.log({outputs_labels[i] : r2_score})

            if epoch > 2 or second_round:

                save_plot_thing = True

                # do plot here

                error_line_arr = np.linspace(np.min(y_test_truth_realspace[:,i]), np.max(y_test_truth_realspace[:,i]), 100)
                
                if single_plot:

                    ax.plot(error_line_arr, error_line_arr, 'k-')
                    ax.plot(error_line_arr, error_line_arr*1.2, 'r--')
                    ax.plot(error_line_arr, error_line_arr*0.8, 'r--')
                    # figure out error bars
                    # ax.errorbar(y_test, mean, yerr=np.std(predictions, axis=1), fmt='o', alpha = 0.1)
                    #ax.plot(y_test_truth_realspace[:,i], y_predict_realspace[:,i], 'o', alpha = 0.05, markersize=2)
                    pcm = ax.scatter(y_test_truth_realspace[:,i], y_predict_realspace[:,i], \
                        c = y_predict_std_realspace[:,i], cmap = 'viridis', alpha = 0.5, s = 3,\
                            vmin = 0.0, vmax = 50.0)

                    ax.set_title('{}, r2_score = {:.4f}'.format(outputs_labels[i], r2_score))
                    ax.set_xlabel('True')
                    ax.set_ylabel('Predicted')
                    ax.set_xscale('log')
                    ax.set_yscale('log')

                    cbar = fig.colorbar(pcm, ax = ax, orientation = 'vertical', location = 'bottom')
                    cbar.set_label('std of predictions [%]')

                else:
                
                    ax[i].plot(error_line_arr, error_line_arr, 'k-')
                    ax[i].plot(error_line_arr, error_line_arr*1.2, 'r--')
                    ax[i].plot(error_line_arr, error_line_arr*0.8, 'r--')
                    # figure out error bars
                    # ax.errorbar(y_test, mean, yerr=np.std(predictions, axis=1), fmt='o', alpha = 0.1)

                    #ax[i].plot(y_test_truth_realspace[:,i], y_predict_realspace[:,i], 'o', alpha = 0.05, markersize=2)
                    pcm = ax[i].scatter(y_test_truth_realspace[:,i], y_predict_realspace[:,i], \
                        c = y_predict_std_realspace[:,i], cmap = 'jet', alpha = 0.5, s = 3, \
                            vmin = 0.0, vmax = 40.0)
                    

                    ax[i].set_title('{}, r2_score = {:.4f}'.format(outputs_labels[i], r2_score))
                    ax[i].set_xlabel('True')
                    if i == 0:
                        ax[i].set_ylabel('Predicted')
                    ax[i].set_xscale('log')
                    ax[i].set_yscale('log')

                    if i == 3:

                        ax[i].set_xlim(1e19, 3e25)
                        ax[i].set_ylim(1e19, 3e25)

                    if i == 4:

                        ax[i].set_xlim(1e16, 3e22)
                        ax[i].set_ylim(1e16, 3e22)

                    if i + 1 == y_train.shape[1] - 1:
                        # add color bar
                        cbar = fig.colorbar(pcm, ax = ax, orientation = 'horizontal', location = 'bottom')
                        cbar.set_label('std of predictions [%]')
        
        cm = confusion_matrix(y_val_classification, final_prediction_class)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['rec. +', 'rec. -'])
        disp.plot(ax = ax[5], cmap = 'Blues', values_format = 'd')
                

        if save_plot_thing and second_round:
            plt.savefig(model_location + '/r2_score_plot_epoch_{}_second_round.png'.format(epoch), dpi = 200, bbox_inches='tight')
        elif save_plot_thing:
            plt.savefig(model_location + '/r2_score_plot_epoch_{}.png'.format(epoch+1), dpi = 200, bbox_inches='tight')

        plt.close()

        print('\n---------------------------------------')

        if epoch == 2:

            with open(model_location + '/config_params.pkl', 'wb') as f:
        
                pickle.dump(config_params, f)



# WnB callback 
wandb_callback = WandbCallback(monitor="val_loss", save_model = True) #save_model = True)

# code out learning rate scheduler 

# def scheduler(epoch, lr):
#     if epoch < 30 and epoch <= 50:
#         lr = lr
#     elif epoch >= 30 and epoch <= 50:
#         lr = lr * tf.math.exp(-0.15)
#     elif epoch > 50:
#         lr = lr * tf.math.exp(-0.20)
#     wandb.log({'lr': lr})
#     return lr

def scheduler(epoch, lr):
    # if epoch < 10:
    #     lr = lr

    if epoch < 10:
        lr = lr * tf.math.exp(-0.10)
    wandb.log({'lr': lr})
    return lr

lr_scheduler = LearningRateScheduler(scheduler)

# the WandbModelCheckpoint saves things locally in the desired format
callbacks_list = [CustomCallback(), earlystopper, wandb_callback, lr_scheduler, \
    WandbModelCheckpoint(filepath = model_location, save_format='tf' )]

# callbacks_list = [earlystopper, lr_scheduler, CustomCallback()]

# TRAINING SECTION 

# get important training variables 
# from configuration file 

num_epochs = config_params['epochs']
batch_size = config_params['batch_size']

# CREATE DATA GENERATOR

# Create an instance of your data generator

train_generator = data.MyDataGenerator_CrazyNN(x=X_train_real_unique, y=y_train_real, batch_size=batch_size, \
    gau_noise=config_params['gau_noise'], \
    inputs_prepro = inputs_prepro, transform_input = transform_input, \
    outputs_prepro = outputs_prepro, transform_output = transform_output,
    min_max_scale = config_params['min_max_scale'],
    rec3_class = config_params['rec3_class'])


# FIRST STAGE OF TRAINING 
second_round = False

# print(y_val_regression.shape)

# print(y_val_classification.shape)

# union = np.concatenate([y_val_regression, y_val_classification.reshape(-1, 1)], axis = 1)
# print(union.shape)

history = model.fit(train_generator, batch_size = batch_size, verbose=2, \
        validation_data = (X_val, np.concatenate([y_val_regression, y_val_classification.reshape(-1, 1)], axis = 1)), epochs=num_epochs, \
            validation_batch_size = 5000, callbacks=callbacks_list)



config_params['wand_run_id'] = run.id

# Save config_params to the model local folder

with open(model_location + '/config_params.pkl', 'wb') as f:
    
        pickle.dump(config_params, f)

wandb.finish()